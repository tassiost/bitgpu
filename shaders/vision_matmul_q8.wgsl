// Q8_0 in-shader dequant matmul for vision tower:
//   out[m,n] = A[m,k] × W[n,k] + bias[n]
//
// Weights are stored as packed Q8_0 (two buffers):
//   w_packed: array<u32>  — [N, K/4] u32 words, each containing 4 int8 values
//   w_scales: array<f32>  — [N, K/32] f32 block scales
// Dequant: value = f32(int8_value) * scale
//
// This is the same pattern as the LLM engine's q8 KV cache (attention_sg_kv8.wgsl),
// adapted for vision matmul. Weight bandwidth: 1.125K bytes/row vs 4K for f32 = 3.56x reduction.
//
// Supports split outputs (QKV): routes output columns to out0/out1/out2.
struct Params {
  M: u32,        // number of input rows (num_patches)
  N: u32,        // total output features
  K: u32,        // input features (must be divisible by 32)
  N0: u32,       // first split (Q or full N)
  N1: u32,       // second split (K, or 0)
  N2: u32,       // third split (V, or 0)
  hasBias: u32,  // 1 = add bias, 0 = no bias
};

const TK: u32 = 1024u;

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w_packed: array<u32>;   // [N, K/4] packed int8
@group(0) @binding(3) var<storage, read> w_scales: array<f32>;   // [N, K/32] block scales
@group(0) @binding(4) var<storage, read> bias: array<f32>;
@group(0) @binding(5) var<storage, read_write> out0: array<f32>;
@group(0) @binding(6) var<storage, read_write> out1: array<f32>;
@group(0) @binding(7) var<storage, read_write> out2: array<f32>;

var<workgroup> shared_x: array<f32, 1024>;

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let n = wg.x * 64u + lid.x;
  let valid = n < p.N;

  let w_words_base = n * (p.K / 4u);    // word offset for this output row
  let w_scales_base = n * (p.K / 32u);  // scale offset for this output row
  let b = select(0.0, bias[n], p.hasBias != 0u);

  for (var m = 0u; m < p.M; m++) {
    let x_base = m * p.K;
    var acc = b;

    for (var kt = 0u; kt < p.K; kt = kt + TK) {
      // Cooperative load — ALL threads must participate (uniform control flow)
      for (var k = lid.x; k < TK; k = k + 64u) {
        let gk = kt + k;
        if (gk < p.K) {
          shared_x[k] = x[x_base + gk];
        }
      }
      workgroupBarrier();

      if (valid) {
        let klen = min(TK, p.K - kt);
        // Process 4 elements per iteration (one u32 word = 4 int8 values)
        for (var k = 0u; k < klen; k = k + 4u) {
          let gk = kt + k;
          let kw = gk / 4u;           // word index in K dimension
          let blk = gk / 32u;         // block index in K dimension
          let packed = w_packed[w_words_base + kw];
          let scale = w_scales[w_scales_base + blk];
          // Extract 4 signed int8 values from u32 (little-endian byte order)
          let bits = i32(packed);
          let v0 = f32((bits << 24) >> 24) * scale;
          let v1 = f32((bits << 16) >> 24) * scale;
          let v2 = f32((bits << 8) >> 24) * scale;
          let v3 = f32(bits >> 24) * scale;
          // Dot product with 4 X values from shared memory
          let xv = vec4<f32>(shared_x[k], shared_x[k + 1u], shared_x[k + 2u], shared_x[k + 3u]);
          let wv = vec4<f32>(v0, v1, v2, v3);
          acc = acc + dot(xv, wv);
        }
      }
      workgroupBarrier();
    }

    // Route to correct output buffer (only valid threads write)
    if (valid) {
      if (n < p.N0) {
        out0[m * p.N0 + n] = acc;
      } else if (n < p.N0 + p.N1) {
        out1[m * p.N1 + (n - p.N0)] = acc;
      } else {
        out2[m * p.N2 + (n - p.N0 - p.N1)] = acc;
      }
    }
  }
}
