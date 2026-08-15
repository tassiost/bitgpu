// SubgroupMatrix Q8_0 matmul for vision tower:
//   out[m,n] = residual[m,n] + A[m,k] × W[n,k] + bias[n]
//
// Same as vision_matmul_q8_sgmat but with fused residual add. Single output.
//
// Uses hardware tensor cores (Metal simdgroup_matrix) via WebGPU's
// chromium_experimental_subgroup_matrix extension. On Apple Silicon,
// each subgroup (32 threads) computes an 8×8×8 matmul tile.
//
// Workgroup: 8 subgroups × 32 threads = 256 threads total.
// Each subgroup handles one 8×8 output tile within a 16×32 workgroup tile.
// K dimension is tiled in 8-element steps (loaded from shared memory).
//
// Layout: SUBGROUP_M=2 (rows), SUBGROUP_N=4 (cols) = 8 subgroups.
// Workgroup tile: 16 rows × 32 cols = 512 output elements.

enable f16;
enable subgroups;
enable chromium_experimental_subgroup_matrix;
diagnostic(off, chromium.subgroup_matrix_uniformity);

struct Params {
  M: u32,        // number of input rows (num_patches)
  N: u32,        // output features
  K: u32,        // input features (must be divisible by 8)
  hasBias: u32,  // 1 = add bias
  _p0: u32, _p1: u32, _p2: u32,
};

// Matrix tile dimensions (Apple Silicon simdgroup_matrix)
const SGM_M: u32 = 8u;
const SGM_N: u32 = 8u;
const SGM_K: u32 = 8u;

// Workgroup tile: 2×4 subgroups = 8 subgroups
const SUBGROUP_M: u32 = 2u;
const SUBGROUP_N: u32 = 4u;
const NUM_SUBGROUPS: u32 = SUBGROUP_M * SUBGROUP_N;  // 8
const SUBGROUP_SIZE: u32 = 32u;
const WG_SIZE: u32 = NUM_SUBGROUPS * SUBGROUP_SIZE;  // 256

// Workgroup output tile dimensions
const WG_M: u32 = SUBGROUP_M * SGM_M;  // 16
const WG_N: u32 = SUBGROUP_N * SGM_N;  // 32

// K tile size (loaded into shared memory per iteration)
// 4 SubgroupMatrix K-steps per load → 4x fewer barriers
const TILE_K: u32 = 32u;  // 4 * SGM_K
const K_STEPS_PER_TILE: u32 = TILE_K / SGM_K;  // 4

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;          // [M, K]
@group(0) @binding(2) var<storage, read> w_packed: array<u32>;   // [N, K/4] packed int8
@group(0) @binding(3) var<storage, read> w_scales: array<f32>;   // [N, K/32] block scales
@group(0) @binding(4) var<storage, read> bias: array<f32>;       // [N]
@group(0) @binding(5) var<storage, read> residual: array<f32>;  // [M, N]
@group(0) @binding(6) var<storage, read_write> out: array<f32>;

// Shared memory for f16 tiles:
//   A tile: [WG_M, TILE_K] = 16×8 f16 = 128 bytes
//   B tile: [WG_N, TILE_K] = 32×8 f16 = 256 bytes
// Total: 384 bytes (tiny!)
var<workgroup> shmem_a: array<f16, 512u>;   // 128
var<workgroup> shmem_b: array<f16, 1024u>;   // 256
// f32 result storage for subgroupMatrixStore (f32 result can't store to f16)
// [WG_M, WG_N] = 16×32 f32 = 2048 bytes
var<workgroup> shmem_result: array<f32, 512u>;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_id) sg_id: u32) {
  let tid = lid.x;
  let tileM = wg.y * WG_M;
  let tileN = wg.x * WG_N;

  // Map subgroup_id to 2D position in the workgroup tile
  let sg_m = sg_id % SUBGROUP_M;
  let sg_n = sg_id / SUBGROUP_M;

  // Each subgroup's output tile position
  let sg_tileM = tileM + sg_m * SGM_M;
  let sg_tileN = tileN + sg_n * SGM_N;

  var acc = subgroup_matrix_result<f32, SGM_N, SGM_M>();

  let Ksteps = p.K / TILE_K;

  for (var ks = 0u; ks < Ksteps; ks = ks + 1u) {
    let k0 = ks * TILE_K;

    // Cooperative load A tile [WG_M, TILE_K] as f16 (128 elements, 256 threads)
    for (var e = tid; e < WG_M * TILE_K; e = e + WG_SIZE) {
      let m = e / TILE_K;
      let k = e % TILE_K;
      let gm = tileM + m;
      let gk = k0 + k;
      if (gm < p.M && gk < p.K) {
        shmem_a[e] = f16(x[gm * p.K + gk]);
      } else {
        shmem_a[e] = f16(0.0);
      }
    }

    // Cooperative load B tile [WG_N, TILE_K] as f16 — dequantize Q8_0 inline (256 elements)
    for (var e = tid; e < WG_N * TILE_K; e = e + WG_SIZE) {
      let n = e / TILE_K;
      let k = e % TILE_K;
      let gn = tileN + n;
      let gk = k0 + k;
      var val = 0.0;
      if (gn < p.N && gk < p.K) {
        let kw = gk / 4u;
        let blk = gk / 32u;
        let packed = w_packed[gn * (p.K / 4u) + kw];
        let scale = w_scales[gn * (p.K / 32u) + blk];
        let bits = i32(packed);
        let byte_idx = gk % 4u;
        let byte_val = select(
          select(
            select(
              f32((bits << 24) >> 24),
              f32((bits << 16) >> 24), byte_idx == 1u),
            f32((bits << 8) >> 24), byte_idx == 2u),
          f32(bits >> 24), byte_idx == 3u);
        val = byte_val * scale;
      }
      shmem_b[e] = f16(val);
    }

    workgroupBarrier();

    // Each subgroup does 4 SubgroupMatrix 8×8×8 multiplies from the K tile
    let a_base = sg_m * SGM_M * TILE_K;  // row offset in shmem_a
    let b_base = sg_n * SGM_N * TILE_K;  // row offset in shmem_b

    for (var ki = 0u; ki < K_STEPS_PER_TILE; ki = ki + 1u) {
      let a_off = a_base + ki * SGM_K;
      let b_off = b_base + ki * SGM_K;
      let a_mat = subgroupMatrixLoad<subgroup_matrix_left<f16, SGM_K, SGM_M>>(&shmem_a, a_off, false, TILE_K);
      let b_mat = subgroupMatrixLoad<subgroup_matrix_right<f16, SGM_N, SGM_K>>(&shmem_b, b_off, false, TILE_K);
      acc = subgroupMatrixMultiplyAccumulate(a_mat, b_mat, acc);
    }

    workgroupBarrier();
  }

  // Store result to shared memory
  let result_offset = sg_m * SGM_M * WG_N + sg_n * SGM_N;
  subgroupMatrixStore(&shmem_result, result_offset, acc, false, WG_N);

  workgroupBarrier();

  // Cooperative write: each thread writes one element of the 16×32 result
  for (var e = tid; e < WG_M * WG_N; e = e + WG_SIZE) {
    let m = e / WG_N;
    let n = e % WG_N;
    let gm = tileM + m;
    let gn = tileN + n;
    if (gm < p.M && gn < p.N) {
      let val = residual[gm * p.N + gn] + shmem_result[m * WG_N + n] + select(0.0, bias[gn], p.hasBias != 0u);
      out[gm * p.N + gn] = val;
    }
  }
}
