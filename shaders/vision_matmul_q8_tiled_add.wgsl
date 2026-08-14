// 2D tiled Q8_0 matmul + residual add for vision attn output:
//   out[m,n] = residual[m,n] + A[m,k] × W[n,k] + bias[n]
//
// Same as vision_matmul_q8_tiled but with fused residual add.
// Single output only (N1=N2=0).
const BM: u32 = 64u;
const BN: u32 = 64u;
const BKV: u32 = 4u;

struct Params {
  M: u32,        // number of input rows (num_patches)
  N: u32,        // output features
  K: u32,        // input features (must be divisible by 4)
  hasBias: u32,  // 1 = add bias
  _p0: u32, _p1: u32, _p2: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w_packed: array<u32>;
@group(0) @binding(3) var<storage, read> w_scales: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@group(0) @binding(6) var<storage, read> residual: array<f32>;  // [M, N]

var<workgroup> xs: array<vec4<f32>, 256>;
var<workgroup> ws: array<vec4<f32>, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let tileM = wg.y * BM;
  let tileN = wg.x * BN;
  let tr = (tid / 16u) * 4u;
  let tc = (tid % 16u) * 4u;
  let Kv = p.K / 4u;

  var acc: array<f32, 16>;
  for (var i = 0u; i < 16u; i = i + 1u) {
    let tn = i % 4u;
    let gn = tileN + tc + tn;
    acc[i] = select(0.0, bias[gn], p.hasBias != 0u && gn < p.N);
  }

  let Ksteps = Kv / BKV;
  for (var ks = 0u; ks < Ksteps; ks = ks + 1u) {
    let k0v = ks * BKV;
    for (var e = tid; e < BM * BKV; e = e + 256u) {
      let m = e / BKV; let kv = e % BKV; let gm = tileM + m; let k = (k0v + kv) * 4u;
      if (gm < p.M && k + 3u < p.K) {
        let xb = gm * p.K + k;
        xs[e] = vec4<f32>(x[xb], x[xb + 1u], x[xb + 2u], x[xb + 3u]);
      } else { xs[e] = vec4<f32>(0.0); }
    }
    for (var e = tid; e < BN * BKV; e = e + 256u) {
      let n = e / BKV; let kv = e % BKV; let gn = tileN + n; let k = (k0v + kv) * 4u;
      var wv = vec4<f32>(0.0);
      if (gn < p.N && k + 3u < p.K) {
        let kw = k / 4u; let blk = k / 32u;
        let packed = w_packed[gn * (p.K / 4u) + kw];
        let scale = w_scales[gn * (p.K / 32u) + blk];
        let bits = i32(packed);
        wv = vec4<f32>(f32((bits << 24) >> 24) * scale, f32((bits << 16) >> 24) * scale,
                       f32((bits << 8) >> 24) * scale, f32(bits >> 24) * scale);
      }
      ws[e] = wv;
    }
    workgroupBarrier();
    for (var kv = 0u; kv < BKV; kv = kv + 1u) {
      var xr: array<vec4<f32>, 4>;
      for (var tm = 0u; tm < 4u; tm = tm + 1u) { xr[tm] = xs[(tr + tm) * BKV + kv]; }
      for (var tn = 0u; tn < 4u; tn = tn + 1u) {
        let w = ws[(tc + tn) * BKV + kv];
        for (var tm = 0u; tm < 4u; tm = tm + 1u) { acc[tm * 4u + tn] = acc[tm * 4u + tn] + dot(xr[tm], w); }
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
