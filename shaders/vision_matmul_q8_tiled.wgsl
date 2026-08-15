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
// Inner compute loops are MANUALLY UNROLLED — the WGSL→Metal compiler
// doesn't always unroll even with known bounds (nuss-and-bolts study
// showed ~3x from manual unrolling on Apple Silicon).
//
// Supports split outputs (QKV): routes output columns to out0/out1/out2.
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
const BKV: u32 = 8u;  // BK / 4 (BK = 32, processed as 8 vec4 steps)

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;          // [M, K]
@group(0) @binding(2) var<storage, read> w_packed: array<u32>;   // [N, K/4] packed int8
@group(0) @binding(3) var<storage, read> w_scales: array<f32>;   // [N, K/32] block scales
@group(0) @binding(4) var<storage, read> bias: array<f32>;       // [N]
@group(0) @binding(5) var<storage, read_write> out0: array<f32>;
@group(0) @binding(6) var<storage, read_write> out1: array<f32>;
@group(0) @binding(7) var<storage, read_write> out2: array<f32>;

var<workgroup> xs: array<vec4<f32>, 512>;  // BM * BKV = 64 * 8 = 512
var<workgroup> ws: array<vec4<f32>, 512>;  // BN * BKV = 64 * 8 = 512

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let Ntot = p.N0 + p.N1 + p.N2;
  let tid = lid.x;
  let tileM = wg.y * BM;
  let tileN = wg.x * BN;
  let tr = (tid / 16u) * 4u;  // Thread row in 4x4 tile
  let tc = (tid % 16u) * 4u;  // Thread col in 4x4 tile
  let Kv = p.K / 4u;          // K in vec4 units

  // Initialize accumulator with bias (unrolled)
  var acc: array<f32, 16>;
  acc[ 0] = select(0.0, bias[tileN + tc + 0u], p.hasBias != 0u && tileN + tc + 0u < Ntot);
  acc[ 1] = select(0.0, bias[tileN + tc + 1u], p.hasBias != 0u && tileN + tc + 1u < Ntot);
  acc[ 2] = select(0.0, bias[tileN + tc + 2u], p.hasBias != 0u && tileN + tc + 2u < Ntot);
  acc[ 3] = select(0.0, bias[tileN + tc + 3u], p.hasBias != 0u && tileN + tc + 3u < Ntot);
  acc[ 4] = acc[ 0]; acc[ 5] = acc[ 1]; acc[ 6] = acc[ 2]; acc[ 7] = acc[ 3];
  acc[ 8] = acc[ 0]; acc[ 9] = acc[ 1]; acc[10] = acc[ 2]; acc[11] = acc[ 3];
  acc[12] = acc[ 0]; acc[13] = acc[ 1]; acc[14] = acc[ 2]; acc[15] = acc[ 3];

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

    // Compute 4x4 register tile — manually unrolled tm/tn loops
    // (WGSL→Metal compiler doesn't always unroll even with known bounds)
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

  // Write output with split routing (unrolled)
  {
    let gm0 = tileM + tr + 0u;
    if (gm0 < p.M) {
      let gn0 = tileN + tc + 0u; if (gn0 < Ntot) { let v = acc[ 0]; if (gn0 < p.N0) { out0[gm0 * p.N0 + gn0] = v; } else if (gn0 < p.N0 + p.N1) { out1[gm0 * p.N1 + (gn0 - p.N0)] = v; } else { out2[gm0 * p.N2 + (gn0 - p.N0 - p.N1)] = v; } }
      let gn1 = tileN + tc + 1u; if (gn1 < Ntot) { let v = acc[ 1]; if (gn1 < p.N0) { out0[gm0 * p.N0 + gn1] = v; } else if (gn1 < p.N0 + p.N1) { out1[gm0 * p.N1 + (gn1 - p.N0)] = v; } else { out2[gm0 * p.N2 + (gn1 - p.N0 - p.N1)] = v; } }
      let gn2 = tileN + tc + 2u; if (gn2 < Ntot) { let v = acc[ 2]; if (gn2 < p.N0) { out0[gm0 * p.N0 + gn2] = v; } else if (gn2 < p.N0 + p.N1) { out1[gm0 * p.N1 + (gn2 - p.N0)] = v; } else { out2[gm0 * p.N2 + (gn2 - p.N0 - p.N1)] = v; } }
      let gn3 = tileN + tc + 3u; if (gn3 < Ntot) { let v = acc[ 3]; if (gn3 < p.N0) { out0[gm0 * p.N0 + gn3] = v; } else if (gn3 < p.N0 + p.N1) { out1[gm0 * p.N1 + (gn3 - p.N0)] = v; } else { out2[gm0 * p.N2 + (gn3 - p.N0 - p.N1)] = v; } }
    }
  }
  {
    let gm1 = tileM + tr + 1u;
    if (gm1 < p.M) {
      let gn0 = tileN + tc + 0u; if (gn0 < Ntot) { let v = acc[ 4]; if (gn0 < p.N0) { out0[gm1 * p.N0 + gn0] = v; } else if (gn0 < p.N0 + p.N1) { out1[gm1 * p.N1 + (gn0 - p.N0)] = v; } else { out2[gm1 * p.N2 + (gn0 - p.N0 - p.N1)] = v; } }
      let gn1 = tileN + tc + 1u; if (gn1 < Ntot) { let v = acc[ 5]; if (gn1 < p.N0) { out0[gm1 * p.N0 + gn1] = v; } else if (gn1 < p.N0 + p.N1) { out1[gm1 * p.N1 + (gn1 - p.N0)] = v; } else { out2[gm1 * p.N2 + (gn1 - p.N0 - p.N1)] = v; } }
      let gn2 = tileN + tc + 2u; if (gn2 < Ntot) { let v = acc[ 6]; if (gn2 < p.N0) { out0[gm1 * p.N0 + gn2] = v; } else if (gn2 < p.N0 + p.N1) { out1[gm1 * p.N1 + (gn2 - p.N0)] = v; } else { out2[gm1 * p.N2 + (gn2 - p.N0 - p.N1)] = v; } }
      let gn3 = tileN + tc + 3u; if (gn3 < Ntot) { let v = acc[ 7]; if (gn3 < p.N0) { out0[gm1 * p.N0 + gn3] = v; } else if (gn3 < p.N0 + p.N1) { out1[gm1 * p.N1 + (gn3 - p.N0)] = v; } else { out2[gm1 * p.N2 + (gn3 - p.N0 - p.N1)] = v; } }
    }
  }
  {
    let gm2 = tileM + tr + 2u;
    if (gm2 < p.M) {
      let gn0 = tileN + tc + 0u; if (gn0 < Ntot) { let v = acc[ 8]; if (gn0 < p.N0) { out0[gm2 * p.N0 + gn0] = v; } else if (gn0 < p.N0 + p.N1) { out1[gm2 * p.N1 + (gn0 - p.N0)] = v; } else { out2[gm2 * p.N2 + (gn0 - p.N0 - p.N1)] = v; } }
      let gn1 = tileN + tc + 1u; if (gn1 < Ntot) { let v = acc[ 9]; if (gn1 < p.N0) { out0[gm2 * p.N0 + gn1] = v; } else if (gn1 < p.N0 + p.N1) { out1[gm2 * p.N1 + (gn1 - p.N0)] = v; } else { out2[gm2 * p.N2 + (gn1 - p.N0 - p.N1)] = v; } }
      let gn2 = tileN + tc + 2u; if (gn2 < Ntot) { let v = acc[10]; if (gn2 < p.N0) { out0[gm2 * p.N0 + gn2] = v; } else if (gn2 < p.N0 + p.N1) { out1[gm2 * p.N1 + (gn2 - p.N0)] = v; } else { out2[gm2 * p.N2 + (gn2 - p.N0 - p.N1)] = v; } }
      let gn3 = tileN + tc + 3u; if (gn3 < Ntot) { let v = acc[11]; if (gn3 < p.N0) { out0[gm2 * p.N0 + gn3] = v; } else if (gn3 < p.N0 + p.N1) { out1[gm2 * p.N1 + (gn3 - p.N0)] = v; } else { out2[gm2 * p.N2 + (gn3 - p.N0 - p.N1)] = v; } }
    }
  }
  {
    let gm3 = tileM + tr + 3u;
    if (gm3 < p.M) {
      let gn0 = tileN + tc + 0u; if (gn0 < Ntot) { let v = acc[12]; if (gn0 < p.N0) { out0[gm3 * p.N0 + gn0] = v; } else if (gn0 < p.N0 + p.N1) { out1[gm3 * p.N1 + (gn0 - p.N0)] = v; } else { out2[gm3 * p.N2 + (gn0 - p.N0 - p.N1)] = v; } }
      let gn1 = tileN + tc + 1u; if (gn1 < Ntot) { let v = acc[13]; if (gn1 < p.N0) { out0[gm3 * p.N0 + gn1] = v; } else if (gn1 < p.N0 + p.N1) { out1[gm3 * p.N1 + (gn1 - p.N0)] = v; } else { out2[gm3 * p.N2 + (gn1 - p.N0 - p.N1)] = v; } }
      let gn2 = tileN + tc + 2u; if (gn2 < Ntot) { let v = acc[14]; if (gn2 < p.N0) { out0[gm3 * p.N0 + gn2] = v; } else if (gn2 < p.N0 + p.N1) { out1[gm3 * p.N1 + (gn2 - p.N0)] = v; } else { out2[gm3 * p.N2 + (gn2 - p.N0 - p.N1)] = v; } }
      let gn3 = tileN + tc + 3u; if (gn3 < Ntot) { let v = acc[15]; if (gn3 < p.N0) { out0[gm3 * p.N0 + gn3] = v; } else if (gn3 < p.N0 + p.N1) { out1[gm3 * p.N1 + (gn3 - p.N0)] = v; } else { out2[gm3 * p.N2 + (gn3 - p.N0 - p.N1)] = v; } }
    }
  }
}
