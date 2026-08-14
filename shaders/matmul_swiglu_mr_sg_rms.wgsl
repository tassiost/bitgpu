// Fused RMSNorm + multi-row gate/up GEMV + SwiGLU for decode (M=1). Eliminates the
// separate rmsnorm_sg dispatch before the MLP matmul. Each workgroup redundantly
// computes the sum-of-squares of x, but saves one kernel launch per layer.
//
// Replaces: rmsnorm_sg → matmul_swiglu_mr_sg (2 dispatches → 1)
//
// Bindings follow the engine's setup() convention: uniform, then all inputs, then outputs.
// ins: [x, signbits, scales, gamma]  outs: [y]
enable subgroups;
override SG: u32 = 32u;
override ROWS: u32 = 4u;
struct Params { K: u32, nb: u32, F: u32, gridX: u32, D: u32, _p0: u32, _p1: u32, _p2: u32, eps: f32, _p3: u32, _p4: u32, _p5: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [K/4] — RAW input (pre-norm)
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [2F, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [2F, nb]
@group(0) @binding(4) var<storage, read> gamma: array<f32>;     // [D] — RMSNorm weights
@group(0) @binding(5) var<storage, read_write> y: array<f32>;   // [F]

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let nBase = (wg.y * p.gridX + wg.x) * ROWS;
  let Kvec = p.K / 4u;
  let wStride = p.K / 32u;

  // Phase 1: compute RMSNorm sum-of-squares (redundant per workgroup, but saves a dispatch)
  var sos = 0.0;
  for (var gi = lane; gi < Kvec; gi = gi + SG) {
    let xv = x[gi];
    sos = sos + dot(xv, xv);
  }
  let total = subgroupAdd(sos);
  let inv = inverseSqrt(total / f32(p.D) + p.eps);

  // Phase 2: gate/up matmul with on-the-fly normalization + SwiGLU
  var g: array<f32, 8>;                            // ROWS <= 8
  var u: array<f32, 8>;
  for (var r = 0u; r < ROWS; r = r + 1u) { g[r] = 0.0; u[r] = 0.0; }
  for (var gi = lane; gi < Kvec; gi = gi + SG) {
    let k = gi * 4u;
    let xv = x[gi];
    let nv = xv * inv * vec4<f32>(gamma[k], gamma[k + 1u], gamma[k + 2u], gamma[k + 3u]);
    let widx = k >> 5u;
    let sh = k & 31u;
    let sc = k / 128u;
    for (var r = 0u; r < ROWS; r = r + 1u) {
      let n = nBase + r;
      if (n < p.F) {
        let gw = (signbits[n * wStride + widx] >> sh) & 0xfu;
        let gv = vec4<f32>(select(-1.0, 1.0, (gw & 1u) != 0u), select(-1.0, 1.0, (gw & 2u) != 0u),
                           select(-1.0, 1.0, (gw & 4u) != 0u), select(-1.0, 1.0, (gw & 8u) != 0u));
        g[r] = g[r] + dot(nv, gv) * scales[n * p.nb + sc];
        let uw = (signbits[(p.F + n) * wStride + widx] >> sh) & 0xfu;
        let uv = vec4<f32>(select(-1.0, 1.0, (uw & 1u) != 0u), select(-1.0, 1.0, (uw & 2u) != 0u),
                           select(-1.0, 1.0, (uw & 4u) != 0u), select(-1.0, 1.0, (uw & 8u) != 0u));
        u[r] = u[r] + dot(nv, uv) * scales[(p.F + n) * p.nb + sc];
      }
    }
  }
  for (var r = 0u; r < ROWS; r = r + 1u) {
    let n = nBase + r;
    let gt = subgroupAdd(g[r]);
    let ut = subgroupAdd(u[r]);
    if (lane == 0u && n < p.F) { y[n] = (gt / (1.0 + exp(-gt))) * ut; }
  }
}
