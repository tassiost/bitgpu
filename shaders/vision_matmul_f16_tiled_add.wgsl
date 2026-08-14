// 2D tiled register-blocked F16 matmul + residual add for vision FFN down:
//   out[m,n] = residual[m,n] + A[m,k] × W[n,k] + bias[n]
//
// 2D tiling (BM=64, BN=64, BK=16) with 4x4 register tiles per thread.
// F16 weights are loaded from global memory as f16 and widened to f32
// during the cooperative W tile load into shared memory.
//
// This combines the 2D tiling pattern (matmul_split_tiled.wgsl) with
// f16 weight storage (attention_sg_kv16.wgsl) for maximum bandwidth savings:
// - 2D tiling eliminates redundant W reads across M iterations
// - f16 storage halves W bandwidth vs f32
//
// Requires shader-f16 feature. Falls back to vision_matmul_tiled_add if unavailable.
enable f16;

const BM: u32 = 64u;
const BN: u32 = 64u;
const BKV: u32 = 4u;  // BK / 4 (BK = 16)

struct Params {
  M: u32,        // number of input rows (num_patches)
  N: u32,        // output features (1152)
  K: u32,        // input features (4304, must be divisible by 4)
  hasBias: u32,  // 1 = add bias
  _p0: u32, _p1: u32, _p2: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;          // [M, K]
@group(0) @binding(2) var<storage, read> w: array<f16>;          // [N, K] f16 weights
@group(0) @binding(3) var<storage, read> bias: array<f32>;       // [N]
@group(0) @binding(4) var<storage, read_write> out: array<f32>;  // [M, N]
@group(0) @binding(5) var<storage, read> residual: array<f32>;   // [M, N]

var<workgroup> xs: array<vec4<f32>, 256>;  // BM * BKV
var<workgroup> ws: array<vec4<f32>, 256>;  // BN * BKV

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let tileM = wg.y * BM;
  let tileN = wg.x * BN;
  let tr = (tid / 16u) * 4u;
  let tc = (tid % 16u) * 4u;
  let Kv = p.K / 4u;

  // Initialize accumulator with bias
  var acc: array<f32, 16>;
  for (var i = 0u; i < 16u; i = i + 1u) {
    let tn = i % 4u;
    let gn = tileN + tc + tn;
    acc[i] = select(0.0, bias[gn], p.hasBias != 0u && gn < p.N);
  }

  let Ksteps = Kv / BKV;
  for (var ks = 0u; ks < Ksteps; ks = ks + 1u) {
    let k0v = ks * BKV;

    // Cooperative load X tile [BM, BKV] — pack f32 into vec4
    for (var e = tid; e < BM * BKV; e = e + 256u) {
      let m = e / BKV; let kv = e % BKV; let gm = tileM + m; let k = (k0v + kv) * 4u;
      if (gm < p.M && k + 3u < p.K) {
        let xb = gm * p.K + k;
        xs[e] = vec4<f32>(x[xb], x[xb + 1u], x[xb + 2u], x[xb + 3u]);
      } else { xs[e] = vec4<f32>(0.0); }
    }

    // Cooperative load W tile [BN, BKV] — widen f16 to f32 at read
    for (var e = tid; e < BN * BKV; e = e + 256u) {
      let n = e / BKV; let kv = e % BKV; let gn = tileN + n; let k = (k0v + kv) * 4u;
      var wv = vec4<f32>(0.0);
      if (gn < p.N && k + 3u < p.K) {
        let wb = gn * p.K + k;
        // Read 4 f16 values and widen to f32
        wv = vec4<f32>(f32(w[wb]), f32(w[wb + 1u]), f32(w[wb + 2u]), f32(w[wb + 3u]));
      }
      ws[e] = wv;
    }

    workgroupBarrier();

    // Compute 4x4 register tile
    for (var kv = 0u; kv < BKV; kv = kv + 1u) {
      var xr: array<vec4<f32>, 4>;
      for (var tm = 0u; tm < 4u; tm = tm + 1u) { xr[tm] = xs[(tr + tm) * BKV + kv]; }
      for (var tn = 0u; tn < 4u; tn = tn + 1u) {
        let wv = ws[(tc + tn) * BKV + kv];
        for (var tm = 0u; tm < 4u; tm = tm + 1u) {
          acc[tm * 4u + tn] = acc[tm * 4u + tn] + dot(xr[tm], wv);
        }
      }
    }

    workgroupBarrier();
  }

  // Write output with fused residual add
  for (var tm = 0u; tm < 4u; tm = tm + 1u) {
    let gm = tileM + tr + tm;
    if (gm >= p.M) { continue; }
    for (var tn = 0u; tn < 4u; tn = tn + 1u) {
      let gn = tileN + tc + tn;
      if (gn >= p.N) { continue; }
      out[gm * p.N + gn] = residual[gm * p.N + gn] + acc[tm * 4u + tn];
    }
  }
}
