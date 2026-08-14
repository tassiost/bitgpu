// Fused RMSNorm + split-K GEMV for decode (M=1). Eliminates the separate rmsnorm_sg
// dispatch by computing the norm inside each matmul workgroup. Each workgroup (one
// per output column) redundantly computes the sum-of-squares of x, but saves one
// kernel launch per layer. Net win when launch overhead > redundant compute.
//
// Replaces: rmsnorm_sg → matmul_split_sg (2 dispatches → 1)
//
// Bindings follow the engine's setup() convention: uniform, then all inputs, then outputs.
// ins: [x, signbits, scales, gamma]  outs: [out0, out1, out2]
enable subgroups;
override SG: u32 = 32u;
struct Params { K: u32, nb: u32, N0: u32, N1: u32, N2: u32, gridX: u32, D: u32, _pad: u32, eps: f32, _p1: u32, _p2: u32, _p3: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [K/4] — RAW input (pre-norm)
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [N0+N1+N2, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N0+N1+N2, nb]
@group(0) @binding(4) var<storage, read> gamma: array<f32>;     // [D] — RMSNorm weights
@group(0) @binding(5) var<storage, read_write> out0: array<f32>;
@group(0) @binding(6) var<storage, read_write> out1: array<f32>;
@group(0) @binding(7) var<storage, read_write> out2: array<f32>;

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let Ntot = p.N0 + p.N1 + p.N2;
  let n = wg.y * p.gridX + wg.x;
  if (n >= Ntot) { return; }
  let wRow = n * (p.K / 32u);
  let sbase = n * p.nb;
  let Kvec = p.K / 4u;

  // Phase 1: compute RMSNorm sum-of-squares (redundant per workgroup, but saves a dispatch)
  var sos = 0.0;
  for (var gi = lane; gi < Kvec; gi = gi + SG) {
    let xv = x[gi];
    sos = sos + dot(xv, xv);
  }
  let total = subgroupAdd(sos);
  let inv = inverseSqrt(total / f32(p.D) + p.eps);

  // Phase 2: matmul with on-the-fly normalization
  var acc = 0.0;
  for (var gi = lane; gi < Kvec; gi = gi + SG) {
    let k = gi * 4u;
    let xv = x[gi];
    let nv = xv * inv * vec4<f32>(gamma[k], gamma[k + 1u], gamma[k + 2u], gamma[k + 3u]);
    let word = signbits[wRow + (k >> 5u)];
    let bits4 = (word >> (k & 31u)) & 0xfu;
    let sv = vec4<f32>(select(-1.0, 1.0, (bits4 & 1u) != 0u), select(-1.0, 1.0, (bits4 & 2u) != 0u),
                       select(-1.0, 1.0, (bits4 & 4u) != 0u), select(-1.0, 1.0, (bits4 & 8u) != 0u));
    acc = acc + dot(nv, sv) * scales[sbase + (k / 128u)];
  }
  let result = subgroupAdd(acc);
  if (lane == 0u) {
    if (n < p.N0) { out0[n] = result; }
    else if (n < p.N0 + p.N1) { out1[n - p.N0] = result; }
    else { out2[n - p.N0 - p.N1] = result; }
  }
}
