// 2D tiled register-blocked Q8_0 matmul for vision tower:
//   out[m,n] = A[m,k] × W[n,k] + bias[n]
//
// 2D tiling (BM=64, BN=64, BK=16) with 4x4 register tiles per thread.
// Both X and W tiles are loaded into shared memory, so W is read from global
// memory only K/16 times (vs M * K/4 in the 1D tiled version). For M=234
// patches, this eliminates ~234× redundant W reads.
//
// Q8_0 weights are dequantized during the cooperative W tile load:
//   value = f32(int8(byte)) * block_scale
// Then stored as vec4<f32> in shared memory for the dot product loop.
//
// Supports split outputs (QKV): routes output columns to out0/out1/out2.
// Pattern follows matmul_split_tiled.wgsl from the LLM engine.
struct Params {
  M: u32,        // number of input rows (num_patches)
  N: u32,        // total output features
  K: u32,        // input features (must be divisible by 4)
  N0: u32,       // first split (Q or full N)
  N1: u32,       // second split (K, or 0)
  N2: u32,       // third split (V, or 0)
  hasBias: u32,  // 1 = add bias, 0 = no bias
};

const BM: u32 = 64u;
const BN: u32 = 64u;
const BKV: u32 = 4u;  // BK / 4 (BK = 16, processed as 4 vec4 steps)

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;          // [M, K]
@group(0) @binding(2) var<storage, read> w_packed: array<u32>;   // [N, K/4] packed int8
@group(0) @binding(3) var<storage, read> w_scales: array<f32>;   // [N, K/32] block scales
@group(0) @binding(4) var<storage, read> bias: array<f32>;       // [N]
@group(0) @binding(5) var<storage, read_write> out0: array<f32>;
@group(0) @binding(6) var<storage, read_write> out1: array<f32>;
@group(0) @binding(7) var<storage, read_write> out2: array<f32>;

var<workgroup> xs: array<vec4<f32>, 256>;  // BM * BKV = 64 * 4 = 256
var<workgroup> ws: array<vec4<f32>, 256>;  // BN * BKV = 64 * 4 = 256

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let Ntot = p.N0 + p.N1 + p.N2;
  let tid = lid.x;
  let tileM = wg.y * BM;
  let tileN = wg.x * BN;
  let tr = (tid / 16u) * 4u;  // Thread row in 4x4 tile
  let tc = (tid % 16u) * 4u;  // Thread col in 4x4 tile
  let Kv = p.K / 4u;          // K in vec4 units

  // Initialize accumulator with bias
  var acc: array<f32, 16>;
  for (var i = 0u; i < 16u; i = i + 1u) {
    let tn = i % 4u;
    let gn = tileN + tc + tn;
    acc[i] = select(0.0, bias[gn], p.hasBias != 0u && gn < Ntot);
  }

  let Ksteps = Kv / BKV;
  for (var ks = 0u; ks < Ksteps; ks = ks + 1u) {
    let k0v = ks * BKV;

    // Cooperative load X tile [BM, BKV] — pack f32 into vec4
    for (var e = tid; e < BM * BKV; e = e + 256u) {
      let m = e / BKV;
      let kv = e % BKV;
      let gm = tileM + m;
      let k = (k0v + kv) * 4u;
      if (gm < p.M && k + 3u < p.K) {
        let xb = gm * p.K + k;
        xs[e] = vec4<f32>(x[xb], x[xb + 1u], x[xb + 2u], x[xb + 3u]);
      } else {
        xs[e] = vec4<f32>(0.0);
      }
    }

    // Cooperative load W tile [BN, BKV] — dequantize Q8_0 inline
    for (var e = tid; e < BN * BKV; e = e + 256u) {
      let n = e / BKV;
      let kv = e % BKV;
      let gn = tileN + n;
      let k = (k0v + kv) * 4u;
      var wv = vec4<f32>(0.0);
      if (gn < Ntot && k + 3u < p.K) {
        let kw = k / 4u;           // word index in K dimension
        let blk = k / 32u;         // Q8_0 block index
        let packed = w_packed[gn * (p.K / 4u) + kw];
        let scale = w_scales[gn * (p.K / 32u) + blk];
        let bits = i32(packed);
        wv = vec4<f32>(
          f32((bits << 24) >> 24) * scale,
          f32((bits << 16) >> 24) * scale,
          f32((bits << 8) >> 24) * scale,
          f32(bits >> 24) * scale,
        );
      }
      ws[e] = wv;
    }

    workgroupBarrier();

    // Compute 4x4 register tile
    for (var kv = 0u; kv < BKV; kv = kv + 1u) {
      var xr: array<vec4<f32>, 4>;
      for (var tm = 0u; tm < 4u; tm = tm + 1u) {
        xr[tm] = xs[(tr + tm) * BKV + kv];
      }
      for (var tn = 0u; tn < 4u; tn = tn + 1u) {
        let w = ws[(tc + tn) * BKV + kv];
        for (var tm = 0u; tm < 4u; tm = tm + 1u) {
          acc[tm * 4u + tn] = acc[tm * 4u + tn] + dot(xr[tm], w);
        }
      }
    }

    workgroupBarrier();
  }

  // Write output with split routing
  for (var tm = 0u; tm < 4u; tm = tm + 1u) {
    let gm = tileM + tr + tm;
    if (gm >= p.M) { continue; }
    for (var tn = 0u; tn < 4u; tn = tn + 1u) {
      let gn = tileN + tc + tn;
      if (gn >= Ntot) { continue; }
      let v = acc[tm * 4u + tn];
      if (gn < p.N0) {
        out0[gm * p.N0 + gn] = v;
      } else if (gn < p.N0 + p.N1) {
        out1[gm * p.N1 + (gn - p.N0)] = v;
      } else {
        out2[gm * p.N2 + (gn - p.N0 - p.N1)] = v;
      }
    }
  }
}
