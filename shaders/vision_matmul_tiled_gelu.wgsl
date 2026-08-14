// Tiled f32 matmul + GELU fusion for vision FFN up:
//   out[m,n] = GELU(A[m,k] × W[n,k] + bias[n])
// Fuses the matmul and GELU into one dispatch, eliminating the separate GELU pass.
// Uses workgroup shared memory for A (X), same as vision_matmul_tiled.
// Single output only (N1=N2=0).
// NOTE: No early return before workgroupBarrier — guards with `valid` flag.
struct Params {
  M: u32,        // number of input rows (num_patches)
  N: u32,        // output features (intermediate_size = 4304)
  K: u32,        // input features (hidden_size = 1152)
  hasBias: u32,  // 1 = add bias
  _p0: u32, _p1: u32, _p2: u32,
};

const TK: u32 = 1024u;
const GELU_COEF: f32 = 0.7978845608028654;

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

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
      for (var k = lid.x; k < TK; k = k + 64u) {
        let gk = kt + k;
        if (gk < p.K) { shared_x[k] = x[x_base + gk]; }
      }
      workgroupBarrier();
      if (valid) {
        let klen = min(TK, p.K - kt);
        for (var k = 0u; k < klen; k = k + 1u) {
          acc = acc + shared_x[k] * w[w_base + kt + k];
        }
      }
      workgroupBarrier();
    }

    if (valid) {
      // Fused GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
      let inner = GELU_COEF * (acc + 0.044715 * acc * acc * acc);
      let clamped = clamp(inner, -15.0, 15.0);
      let gelu = 0.5 * acc * (1.0 + tanh(clamped));
      out[m * p.N + n] = gelu;
    }
  }
}
