// 2D tiled Q8_0 matmul + GELU for vision FFN up:
//   out[m,n] = GELU(A[m,k] × W[n,k] + bias[n])
//
// Same as vision_matmul_q8_tiled but with fused GELU activation.
// Single output only (N1=N2=0).
// Inner compute loops manually unrolled for Apple Silicon.
const BM: u32 = 64u;
const BN: u32 = 64u;
const BKV: u32 = 8u;
const GELU_COEF: f32 = 0.7978845608028654;

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

var<workgroup> xs: array<vec4<f32>, 512>;
var<workgroup> ws: array<vec4<f32>, 512>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let tileM = wg.y * BM;
  let tileN = wg.x * BN;
  let tr = (tid / 16u) * 4u;
  let tc = (tid % 16u) * 4u;
  let Kv = p.K / 4u;

  // Initialize accumulator with bias (unrolled)
  var acc: array<f32, 16>;
  acc[ 0] = select(0.0, bias[tileN + tc + 0u], p.hasBias != 0u && tileN + tc + 0u < p.N);
  acc[ 1] = select(0.0, bias[tileN + tc + 1u], p.hasBias != 0u && tileN + tc + 1u < p.N);
  acc[ 2] = select(0.0, bias[tileN + tc + 2u], p.hasBias != 0u && tileN + tc + 2u < p.N);
  acc[ 3] = select(0.0, bias[tileN + tc + 3u], p.hasBias != 0u && tileN + tc + 3u < p.N);
  acc[ 4] = acc[ 0]; acc[ 5] = acc[ 1]; acc[ 6] = acc[ 2]; acc[ 7] = acc[ 3];
  acc[ 8] = acc[ 0]; acc[ 9] = acc[ 1]; acc[10] = acc[ 2]; acc[11] = acc[ 3];
  acc[12] = acc[ 0]; acc[13] = acc[ 1]; acc[14] = acc[ 2]; acc[15] = acc[ 3];

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
    // Compute 4x4 register tile — manually unrolled
    for (var kv = 0u; kv < BKV; kv = kv + 1u) {
      let xr0 = xs[(tr + 0u) * BKV + kv];
      let xr1 = xs[(tr + 1u) * BKV + kv];
      let xr2 = xs[(tr + 2u) * BKV + kv];
      let xr3 = xs[(tr + 3u) * BKV + kv];
      let w0 = ws[(tc + 0u) * BKV + kv];
      let w1 = ws[(tc + 1u) * BKV + kv];
      let w2 = ws[(tc + 2u) * BKV + kv];
      let w3 = ws[(tc + 3u) * BKV + kv];
      acc[ 0] = acc[ 0] + xr0.x*w0.x + xr0.y*w0.y + xr0.z*w0.z + xr0.w*w0.w;
      acc[ 1] = acc[ 1] + xr0.x*w1.x + xr0.y*w1.y + xr0.z*w1.z + xr0.w*w1.w;
      acc[ 2] = acc[ 2] + xr0.x*w2.x + xr0.y*w2.y + xr0.z*w2.z + xr0.w*w2.w;
      acc[ 3] = acc[ 3] + xr0.x*w3.x + xr0.y*w3.y + xr0.z*w3.z + xr0.w*w3.w;
      acc[ 4] = acc[ 4] + xr1.x*w0.x + xr1.y*w0.y + xr1.z*w0.z + xr1.w*w0.w;
      acc[ 5] = acc[ 5] + xr1.x*w1.x + xr1.y*w1.y + xr1.z*w1.z + xr1.w*w1.w;
      acc[ 6] = acc[ 6] + xr1.x*w2.x + xr1.y*w2.y + xr1.z*w2.z + xr1.w*w2.w;
      acc[ 7] = acc[ 7] + xr1.x*w3.x + xr1.y*w3.y + xr1.z*w3.z + xr1.w*w3.w;
      acc[ 8] = acc[ 8] + xr2.x*w0.x + xr2.y*w0.y + xr2.z*w0.z + xr2.w*w0.w;
      acc[ 9] = acc[ 9] + xr2.x*w1.x + xr2.y*w1.y + xr2.z*w1.z + xr2.w*w1.w;
      acc[10] = acc[10] + xr2.x*w2.x + xr2.y*w2.y + xr2.z*w2.z + xr2.w*w2.w;
      acc[11] = acc[11] + xr2.x*w3.x + xr2.y*w3.y + xr2.z*w3.z + xr2.w*w3.w;
      acc[12] = acc[12] + xr3.x*w0.x + xr3.y*w0.y + xr3.z*w0.z + xr3.w*w0.w;
      acc[13] = acc[13] + xr3.x*w1.x + xr3.y*w1.y + xr3.z*w1.z + xr3.w*w1.w;
      acc[14] = acc[14] + xr3.x*w2.x + xr3.y*w2.y + xr3.z*w2.z + xr3.w*w2.w;
      acc[15] = acc[15] + xr3.x*w3.x + xr3.y*w3.y + xr3.z*w3.z + xr3.w*w3.w;
    }
    workgroupBarrier();
  }

  // Write output with fused GELU (unrolled)
  {
    let gm = tileM + tr + 0u;
    if (gm < p.M) {
      let gn0 = tileN + tc + 0u; if (gn0 < p.N) { let v = acc[ 0]; let inner = GELU_COEF * (v + 0.044715 * v * v * v); out[gm * p.N + gn0] = 0.5 * v * (1.0 + tanh(clamp(inner, -15.0, 15.0))); }
      let gn1 = tileN + tc + 1u; if (gn1 < p.N) { let v = acc[ 1]; let inner = GELU_COEF * (v + 0.044715 * v * v * v); out[gm * p.N + gn1] = 0.5 * v * (1.0 + tanh(clamp(inner, -15.0, 15.0))); }
      let gn2 = tileN + tc + 2u; if (gn2 < p.N) { let v = acc[ 2]; let inner = GELU_COEF * (v + 0.044715 * v * v * v); out[gm * p.N + gn2] = 0.5 * v * (1.0 + tanh(clamp(inner, -15.0, 15.0))); }
      let gn3 = tileN + tc + 3u; if (gn3 < p.N) { let v = acc[ 3]; let inner = GELU_COEF * (v + 0.044715 * v * v * v); out[gm * p.N + gn3] = 0.5 * v * (1.0 + tanh(clamp(inner, -15.0, 15.0))); }
    }
  }
  {
    let gm = tileM + tr + 1u;
    if (gm < p.M) {
      let gn0 = tileN + tc + 0u; if (gn0 < p.N) { let v = acc[ 4]; let inner = GELU_COEF * (v + 0.044715 * v * v * v); out[gm * p.N + gn0] = 0.5 * v * (1.0 + tanh(clamp(inner, -15.0, 15.0))); }
      let gn1 = tileN + tc + 1u; if (gn1 < p.N) { let v = acc[ 5]; let inner = GELU_COEF * (v + 0.044715 * v * v * v); out[gm * p.N + gn1] = 0.5 * v * (1.0 + tanh(clamp(inner, -15.0, 15.0))); }
      let gn2 = tileN + tc + 2u; if (gn2 < p.N) { let v = acc[ 6]; let inner = GELU_COEF * (v + 0.044715 * v * v * v); out[gm * p.N + gn2] = 0.5 * v * (1.0 + tanh(clamp(inner, -15.0, 15.0))); }
      let gn3 = tileN + tc + 3u; if (gn3 < p.N) { let v = acc[ 7]; let inner = GELU_COEF * (v + 0.044715 * v * v * v); out[gm * p.N + gn3] = 0.5 * v * (1.0 + tanh(clamp(inner, -15.0, 15.0))); }
    }
  }
  {
    let gm = tileM + tr + 2u;
    if (gm < p.M) {
      let gn0 = tileN + tc + 0u; if (gn0 < p.N) { let v = acc[ 8]; let inner = GELU_COEF * (v + 0.044715 * v * v * v); out[gm * p.N + gn0] = 0.5 * v * (1.0 + tanh(clamp(inner, -15.0, 15.0))); }
      let gn1 = tileN + tc + 1u; if (gn1 < p.N) { let v = acc[ 9]; let inner = GELU_COEF * (v + 0.044715 * v * v * v); out[gm * p.N + gn1] = 0.5 * v * (1.0 + tanh(clamp(inner, -15.0, 15.0))); }
      let gn2 = tileN + tc + 2u; if (gn2 < p.N) { let v = acc[10]; let inner = GELU_COEF * (v + 0.044715 * v * v * v); out[gm * p.N + gn2] = 0.5 * v * (1.0 + tanh(clamp(inner, -15.0, 15.0))); }
      let gn3 = tileN + tc + 3u; if (gn3 < p.N) { let v = acc[11]; let inner = GELU_COEF * (v + 0.044715 * v * v * v); out[gm * p.N + gn3] = 0.5 * v * (1.0 + tanh(clamp(inner, -15.0, 15.0))); }
    }
  }
  {
    let gm = tileM + tr + 3u;
    if (gm < p.M) {
      let gn0 = tileN + tc + 0u; if (gn0 < p.N) { let v = acc[12]; let inner = GELU_COEF * (v + 0.044715 * v * v * v); out[gm * p.N + gn0] = 0.5 * v * (1.0 + tanh(clamp(inner, -15.0, 15.0))); }
      let gn1 = tileN + tc + 1u; if (gn1 < p.N) { let v = acc[13]; let inner = GELU_COEF * (v + 0.044715 * v * v * v); out[gm * p.N + gn1] = 0.5 * v * (1.0 + tanh(clamp(inner, -15.0, 15.0))); }
      let gn2 = tileN + tc + 2u; if (gn2 < p.N) { let v = acc[14]; let inner = GELU_COEF * (v + 0.044715 * v * v * v); out[gm * p.N + gn2] = 0.5 * v * (1.0 + tanh(clamp(inner, -15.0, 15.0))); }
      let gn3 = tileN + tc + 3u; if (gn3 < p.N) { let v = acc[15]; let inner = GELU_COEF * (v + 0.044715 * v * v * v); out[gm * p.N + gn3] = 0.5 * v * (1.0 + tanh(clamp(inner, -15.0, 15.0))); }
    }
  }
}
