// F16 weight matmul + residual add for vision FFN down:
//   out[m,n] = residual[m,n] + A[m,k] × W[n,k] + bias[n]
//
// Weights are stored as f16 (array<f16>), widened to f32 at read time.
// This halves weight bandwidth vs f32: 19.8 MB → 9.9 MB per layer for ffn_down.
//
// Pattern follows the LLM engine's f16 KV cache (attention_sg_kv16.wgsl):
//   let val = f32(weights[idx]);  // f16 → f32 widening at read
//
// Requires shader-f16 feature. Falls back to vision_matmul_tiled_add if unavailable.
enable f16;

struct Params {
  M: u32,        // number of input rows (num_patches)
  N: u32,        // output features (1152)
  K: u32,        // input features (4304)
  hasBias: u32,  // 1 = add bias
  _p0: u32, _p1: u32, _p2: u32,
};

const TK: u32 = 1024u;

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;          // [M, K]
@group(0) @binding(2) var<storage, read> w: array<f16>;          // [N, K] f16 weights
@group(0) @binding(3) var<storage, read> bias: array<f32>;       // [N]
@group(0) @binding(4) var<storage, read_write> out: array<f32>;  // [M, N]
@group(0) @binding(5) var<storage, read> residual: array<f32>;   // [M, N]

var<workgroup> shared_x: array<f32, 1024>;

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let n = wg.x * 64u + lid.x;
  let valid = n < p.N;

  let w_base = n * p.K;
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
        for (var k = 0u; k < klen; k = k + 1u) {
          // Read f16 weight and widen to f32 at read time
          acc = acc + shared_x[k] * f32(w[w_base + kt + k]);
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
