requires packed_4x8_integer_dot_product;

// DP4a-accelerated Q8_0 matmul for vision tower:
//   out[m,n] = A[m,k] × W[n,k] + bias[n]
//
// Uses dot4I8Packed to do 4 int8 dot products in one instruction.
// X (activations) are quantized to int8 on the fly with a per-tile scale.
// W (weights) are kept as packed int8 in shared memory (no dequant).
// The dot product result is rescaled by (x_scale * w_scale) at the end.
//
// Requires: packed_4x8_integer_dot_product WGSL feature.
//
// 2D tiling (BM=64, BN=64, BK=32) with 4x4 register tiles per thread.
struct Params {
  M: u32,        // number of input rows (num_patches)
  N: u32,        // total output features
  K: u32,        // input features (must be divisible by 32 for Q8_0 blocks)
  N0: u32,       // first split (Q or full N)
  N1: u32,       // second split (K, or 0)
  N2: u32,       // third split (V, or 0)
  hasBias: u32,  // 1 = add bias, 0 = no bias
};

const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 32u;   // Q8_0 block size = 32

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;          // [M, K]
@group(0) @binding(2) var<storage, read> w_packed: array<u32>;   // [N, K/4] packed int8
@group(0) @binding(3) var<storage, read> w_scales: array<f32>;   // [N, K/32] block scales
@group(0) @binding(4) var<storage, read> bias: array<f32>;       // [N]
@group(0) @binding(5) var<storage, read_write> out0: array<f32>;
@group(0) @binding(6) var<storage, read_write> out1: array<f32>;
@group(0) @binding(7) var<storage, read_write> out2: array<f32>;

// Shared memory: packed int8 for W (no dequant), packed int8 for X (quantized on load)
var<workgroup> xs: array<u32, 512>;   // BM * BK/4 = 64 * 8 = 512 packed u32
var<workgroup> ws: array<u32, 512>;   // BN * BK/4 = 64 * 8 = 512 packed u32
var<workgroup> x_scales: array<f32, 64>;  // per-row X scale for this tile (BM rows)

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let Ntot = p.N0 + p.N1 + p.N2;
  let tid = lid.x;
  let tileM = wg.y * BM;
  let tileN = wg.x * BN;
  let tr = (tid / 16u) * 4u;  // Thread row in 4x4 tile
  let tc = (tid % 16u) * 4u;  // Thread col in 4x4 tile
  let Kw = p.K / 4u;          // K in packed u32 units
  let Kblocks = p.K / BK;     // K in Q8_0 block units

  // Initialize accumulator with bias (unrolled)
  var acc: array<f32, 16>;
  acc[ 0] = select(0.0, bias[tileN + tc + 0u], p.hasBias != 0u && tileN + tc + 0u < Ntot);
  acc[ 1] = select(0.0, bias[tileN + tc + 1u], p.hasBias != 0u && tileN + tc + 1u < Ntot);
  acc[ 2] = select(0.0, bias[tileN + tc + 2u], p.hasBias != 0u && tileN + tc + 2u < Ntot);
  acc[ 3] = select(0.0, bias[tileN + tc + 3u], p.hasBias != 0u && tileN + tc + 3u < Ntot);
  acc[ 4] = acc[ 0]; acc[ 5] = acc[ 1]; acc[ 6] = acc[ 2]; acc[ 7] = acc[ 3];
  acc[ 8] = acc[ 0]; acc[ 9] = acc[ 1]; acc[10] = acc[ 2]; acc[11] = acc[ 3];
  acc[12] = acc[ 0]; acc[13] = acc[ 1]; acc[14] = acc[ 2]; acc[15] = acc[ 3];

  let Ksteps = Kblocks;  // each step processes one Q8_0 block (32 elements = 8 packed u32)
  for (var ks = 0u; ks < Ksteps; ks = ks + 1u) {
    let k0 = ks * BK;        // base K offset for this block
    let k0w = ks * 8u;       // base packed-u32 offset (32/4 = 8)

    // Cooperative load X tile [BM, BK] — quantize f32 to int8 on the fly
    // Each thread loads some elements, finds the row max, packs to int8
    // First pass: load f32 and compute per-row scale
    for (var m = 0u; m < BM; m = m + 1u) {
      let gm = tileM + m;
      if (tid == m) {
        var amax = 0.0;
        for (var k = 0u; k < BK; k = k + 1u) {
          let gk = k0 + k;
          if (gm < p.M && gk < p.K) {
            let v = abs(x[gm * p.K + gk]);
            if (v > amax) { amax = v; }
          }
        }
        let scale = amax / 127.0;
        x_scales[m] = select(1.0, scale, scale > 0.0);
      }
    }
    workgroupBarrier();

    // Second pass: pack X into int8 using the per-row scale
    for (var e = tid; e < BM * 8u; e = e + 256u) {
      let m = e / 8u;
      let kw = e % 8u;       // packed u32 index within this block
      let gm = tileM + m;
      let k = k0 + kw * 4u;  // first f32 element for this packed u32
      var packed_val = 0u;
      if (gm < p.M && k + 3u < p.K) {
        let scale = x_scales[m];
        let v0 = i32(clamp(round(x[gm * p.K + k + 0u] / scale), -128.0, 127.0));
        let v1 = i32(clamp(round(x[gm * p.K + k + 1u] / scale), -128.0, 127.0));
        let v2 = i32(clamp(round(x[gm * p.K + k + 2u] / scale), -128.0, 127.0));
        let v3 = i32(clamp(round(x[gm * p.K + k + 3u] / scale), -128.0, 127.0));
        // Pack as little-endian bytes: v0 in bits 0-7, v1 in 8-15, v2 in 16-23, v3 in 24-31
        packed_val = pack4xI8(vec4<i32>(v0, v1, v2, v3));
      }
      xs[m * 8u + kw] = packed_val;
    }

    // Cooperative load W tile [BN, BK] — keep as packed int8 (no dequant)
    for (var e = tid; e < BN * 8u; e = e + 256u) {
      let n = e / 8u;
      let kw = e % 8u;
      let gn = tileN + n;
      var packed_val = 0u;
      if (gn < Ntot) {
        packed_val = w_packed[gn * Kw + k0w + kw];
      }
      ws[n * 8u + kw] = packed_val;
    }

    workgroupBarrier();

    // Compute 4x4 register tile using dot4I8Packed
    // For each (tr, tc) pair, dot4I8Packed(xs_packed, ws_packed) gives 4 int8 dot products
    // accumulated as i32. We then multiply by (x_scale * w_scale) and add to f32 accumulator.
    let x_scale_r0 = x_scales[tr + 0u];
    let x_scale_r1 = x_scales[tr + 1u];
    let x_scale_r2 = x_scales[tr + 2u];
    let x_scale_r3 = x_scales[tr + 3u];
    let w_scale_0 = w_scales[(tileN + tc + 0u) * Kblocks + ks];
    let w_scale_1 = w_scales[(tileN + tc + 1u) * Kblocks + ks];
    let w_scale_2 = w_scales[(tileN + tc + 2u) * Kblocks + ks];
    let w_scale_3 = w_scales[(tileN + tc + 3u) * Kblocks + ks];

    // 8 dot4I8Packed per (row, col) pair (32 elements = 8 packed u32)
    for (var kw = 0u; kw < 8u; kw = kw + 1u) {
      let xv0 = xs[(tr + 0u) * 8u + kw];
      let xv1 = xs[(tr + 1u) * 8u + kw];
      let xv2 = xs[(tr + 2u) * 8u + kw];
      let xv3 = xs[(tr + 3u) * 8u + kw];
      let wv0 = ws[(tc + 0u) * 8u + kw];
      let wv1 = ws[(tc + 1u) * 8u + kw];
      let wv2 = ws[(tc + 2u) * 8u + kw];
      let wv3 = ws[(tc + 3u) * 8u + kw];

      // dot4I8Packed returns i32 = sum of 4 int8 products
      let d00 = f32(dot4I8Packed(xv0, wv0)) * x_scale_r0 * w_scale_0;
      let d01 = f32(dot4I8Packed(xv0, wv1)) * x_scale_r0 * w_scale_1;
      let d02 = f32(dot4I8Packed(xv0, wv2)) * x_scale_r0 * w_scale_2;
      let d03 = f32(dot4I8Packed(xv0, wv3)) * x_scale_r0 * w_scale_3;
      let d10 = f32(dot4I8Packed(xv1, wv0)) * x_scale_r1 * w_scale_0;
      let d11 = f32(dot4I8Packed(xv1, wv1)) * x_scale_r1 * w_scale_1;
      let d12 = f32(dot4I8Packed(xv1, wv2)) * x_scale_r1 * w_scale_2;
      let d13 = f32(dot4I8Packed(xv1, wv3)) * x_scale_r1 * w_scale_3;
      let d20 = f32(dot4I8Packed(xv2, wv0)) * x_scale_r2 * w_scale_0;
      let d21 = f32(dot4I8Packed(xv2, wv1)) * x_scale_r2 * w_scale_1;
      let d22 = f32(dot4I8Packed(xv2, wv2)) * x_scale_r2 * w_scale_2;
      let d23 = f32(dot4I8Packed(xv2, wv3)) * x_scale_r2 * w_scale_3;
      let d30 = f32(dot4I8Packed(xv3, wv0)) * x_scale_r3 * w_scale_0;
      let d31 = f32(dot4I8Packed(xv3, wv1)) * x_scale_r3 * w_scale_1;
      let d32 = f32(dot4I8Packed(xv3, wv2)) * x_scale_r3 * w_scale_2;
      let d33 = f32(dot4I8Packed(xv3, wv3)) * x_scale_r3 * w_scale_3;

      acc[ 0] = acc[ 0] + d00; acc[ 1] = acc[ 1] + d01; acc[ 2] = acc[ 2] + d02; acc[ 3] = acc[ 3] + d03;
      acc[ 4] = acc[ 4] + d10; acc[ 5] = acc[ 5] + d11; acc[ 6] = acc[ 6] + d12; acc[ 7] = acc[ 7] + d13;
      acc[ 8] = acc[ 8] + d20; acc[ 9] = acc[ 9] + d21; acc[10] = acc[10] + d22; acc[11] = acc[11] + d23;
      acc[12] = acc[12] + d30; acc[13] = acc[13] + d31; acc[14] = acc[14] + d32; acc[15] = acc[15] + d33;
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
