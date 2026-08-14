// Q8_0 in-shader dequant matmul + residual add for vision:
//   out[m,n] = residual[m,n] + A[m,k] × W[n,k] + bias[n]
//
// Same as vision_matmul_q8 but with fused residual add.
// Single output only (N1=N2=0).
struct Params {
  M: u32,        // number of input rows (num_patches)
  N: u32,        // output features
  K: u32,        // input features (must be divisible by 32)
  hasBias: u32,  // 1 = add bias
  _p0: u32, _p1: u32, _p2: u32,
};

const TK: u32 = 1024u;

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w_packed: array<u32>;
@group(0) @binding(3) var<storage, read> w_scales: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@group(0) @binding(6) var<storage, read> residual: array<f32>;  // [M, N]

var<workgroup> shared_x: array<f32, 1024>;

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let n = wg.x * 64u + lid.x;
  let valid = n < p.N;

  let w_words_base = n * (p.K / 4u);
  let w_scales_base = n * (p.K / 32u);
  let b = select(0.0, bias[n], p.hasBias != 0u);

  for (var m = 0u; m < p.M; m++) {
    let x_base = m * p.K;
    var acc = b;

    for (var kt = 0u; kt < p.K; kt = kt + TK) {
      for (var k = lid.x; k < TK; k = k + 64u) {
        let gk = kt + k;
        if (gk < p.K) { shared_x[k] = x[x_base + gk]; }
      }
      workgroupBarrier();
      if (valid) {
        let klen = min(TK, p.K - kt);
        for (var k = 0u; k < klen; k = k + 4u) {
          let gk = kt + k;
          let kw = gk / 4u;
          let blk = gk / 32u;
          let packed = w_packed[w_words_base + kw];
          let scale = w_scales[w_scales_base + blk];
          let bits = i32(packed);
          let v0 = f32((bits << 24) >> 24) * scale;
          let v1 = f32((bits << 16) >> 24) * scale;
          let v2 = f32((bits << 8) >> 24) * scale;
          let v3 = f32(bits >> 24) * scale;
          let xv = vec4<f32>(shared_x[k], shared_x[k + 1u], shared_x[k + 2u], shared_x[k + 3u]);
          let wv = vec4<f32>(v0, v1, v2, v3);
          acc = acc + dot(xv, wv);
        }
      }
      workgroupBarrier();
    }

    if (valid) {
      // Fused residual add: out = residual + matmul
      out[m * p.N + n] = residual[m * p.N + n] + acc;
    }
  }
}
