// Tiled f32 matmul for vision tower: C[m,n] = A[m,k] × W[n,k] + bias[n]
// Uses workgroup shared memory for A (X) to reduce global memory traffic 64×.
// One workgroup per 64 output columns; each thread handles one N column.
// K is processed in tiles of TK=1024 to keep shared memory at 4KB.
//
// Supports split outputs (QKV): routes output columns to out0/out1/out2.
// NOTE: No early return before workgroupBarrier — guards with `valid` flag instead.
struct Params {
  M: u32,        // number of input rows (num_patches)
  N: u32,        // total output features
  K: u32,        // input features
  N0: u32,       // first split (Q or full N)
  N1: u32,       // second split (K, or 0)
  N2: u32,       // third split (V, or 0)
  hasBias: u32,  // 1 = add bias, 0 = no bias
};

const TK: u32 = 1024u;

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out0: array<f32>;
@group(0) @binding(5) var<storage, read_write> out1: array<f32>;
@group(0) @binding(6) var<storage, read_write> out2: array<f32>;

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
      // Cooperative load — ALL threads must participate (uniform control flow)
      for (var k = lid.x; k < TK; k = k + 64u) {
        let gk = kt + k;
        if (gk < p.K) {
          shared_x[k] = x[x_base + gk];
        }
      }
      workgroupBarrier();

      // Dot product — only valid threads compute
      if (valid) {
        let klen = min(TK, p.K - kt);
        for (var k = 0u; k < klen; k = k + 1u) {
          acc = acc + shared_x[k] * w[w_base + kt + k];
        }
      }
      workgroupBarrier();
    }

    // Route to correct output buffer (only valid threads write)
    if (valid) {
      if (n < p.N0) {
        out0[m * p.N0 + n] = acc;
      } else if (n < p.N0 + p.N1) {
        out1[m * p.N1 + (n - p.N0)] = acc;
      } else {
        out2[m * p.N2 + (n - p.N0 - p.N1)] = acc;
      }
    }
  }
}
