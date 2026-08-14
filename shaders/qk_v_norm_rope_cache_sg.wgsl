// Merged post-QKV-matmul kernel for decode (S=1, kv8 cache, subgroup path).
// Replaces 3 separate dispatches (rmsnorm_rope_sg for Q, rmsnorm_rope_sg_kv8 for K,
// copy_kv8 for V) with one. Dispatches H+KV workgroups: the first H handle Q heads
// (RMSNorm + RoPE → qr), the next KV handle K heads (RMSNorm + RoPE + q8 quantize →
// Kc+Ksc) AND the matching V row (q8 quantize → Vc+Vsc, no norm/rope).
//
// This is a consumer merge — no redundant compute. Each workgroup does exactly the
// same work as the separate kernels did, just in a single dispatch.
//
// Sink mode (roll): K is stored UNROPED. The kRope uniform flag controls this:
// kRope=1 → apply RoPE to K; kRope=0 → skip RoPE (store normed K only).
enable subgroups;
override SG: u32 = 32u;
struct Params { H: u32, KV: u32, D: u32, eps: f32, kOutRow0: u32, kRope: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> qbuf: array<f32>;       // Q [H*D]
@group(0) @binding(2) var<storage, read> kbuf: array<f32>;       // K [KV*D]
@group(0) @binding(3) var<storage, read> vbuf: array<f32>;       // V [KV*D]
@group(0) @binding(4) var<storage, read> qgamma: array<f32>;     // q_norm [D]
@group(0) @binding(5) var<storage, read> kgamma: array<f32>;     // k_norm [D]
@group(0) @binding(6) var<storage, read> cos: array<f32>;        // [D]
@group(0) @binding(7) var<storage, read> sin: array<f32>;        // [D]
@group(0) @binding(8) var<storage, read_write> qr: array<f32>;   // Q output [H*D]
@group(0) @binding(9) var<storage, read_write> kcq: array<u32>;  // K cache packed snorm8
@group(0) @binding(10) var<storage, read_write> kcs: array<f32>; // K cache block scales
@group(0) @binding(11) var<storage, read_write> vcq: array<u32>; // V cache packed snorm8
@group(0) @binding(12) var<storage, read_write> vcs: array<f32>; // V cache block scales

var<workgroup> wabs: array<f32, 32>; // per-word abs max (D <= 128 -> at most 32 words)
var<workgroup> wblk: array<f32, 4>;  // per-block scale (D/32 <= 4 blocks)

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let head = wg.x;
  let half = p.D / 2u;
  let W4 = p.D / 4u;

  if (head < p.H) {
    // ── Q: RMSNorm + RoPE → qr ──
    let base = head * p.D;
    var s = 0.0;
    for (var i = lane; i < p.D; i = i + SG) { let v = qbuf[base + i]; s = s + v * v; }
    let inv = inverseSqrt(subgroupAdd(s) / f32(p.D) + p.eps);
    for (var i = lane; i < p.D; i = i + SG) {
      let nd = qbuf[base + i] * inv * qgamma[i];
      var pd: u32; var sgn: f32;
      if (i < half) { pd = i + half; sgn = -1.0; } else { pd = i - half; sgn = 1.0; }
      let rot = sgn * (qbuf[base + pd] * inv * qgamma[pd]);
      qr[base + i] = nd * cos[i] + rot * sin[i];
    }
  } else {
    let krow = head - p.H;
    if (krow >= p.KV) { return; }

    // ── K: RMSNorm + RoPE + q8 quantize → Kc + Ksc ──
    let kbase = krow * p.D;
    var s = 0.0;
    for (var i = lane; i < p.D; i = i + SG) { let v = kbuf[kbase + i]; s = s + v * v; }
    let inv = inverseSqrt(subgroupAdd(s) / f32(p.D) + p.eps);

    var vals: array<vec4<f32>, 8>;  // words per lane: W4/SG <= 8 for SG >= 4
    var wi = 0u;
    for (var w = lane; w < W4; w = w + SG) {
      var vv = vec4<f32>(0.0);
      for (var e = 0u; e < 4u; e = e + 1u) {
        let i = w * 4u + e;
        let nd = kbuf[kbase + i] * inv * kgamma[i];
        if (p.kRope == 1u) {
          var pd: u32; var sgn: f32;
          if (i < half) { pd = i + half; sgn = -1.0; } else { pd = i - half; sgn = 1.0; }
          let rot = sgn * (kbuf[kbase + pd] * inv * kgamma[pd]);
          vv[e] = nd * cos[i] + rot * sin[i];
        } else {
          vv[e] = nd;  // sink mode: store normed K unroped
        }
      }
      vals[wi] = vv;
      wi = wi + 1u;
      wabs[w] = max(max(abs(vv.x), abs(vv.y)), max(abs(vv.z), abs(vv.w)));
    }
    workgroupBarrier();
    if (lane < p.D / 32u) {
      var m = 0.0;
      for (var i = 0u; i < 8u; i = i + 1u) { m = max(m, wabs[lane * 8u + i]); }
      let sc = max(m, 1e-30);
      wblk[lane] = sc;
      kcs[(p.kOutRow0 + krow) * (p.D / 32u) + lane] = sc;
    }
    workgroupBarrier();
    wi = 0u;
    for (var w = lane; w < W4; w = w + SG) {
      kcq[(p.kOutRow0 + krow) * W4 + w] = pack4x8snorm(vals[wi] / wblk[w >> 3u]);
      wi = wi + 1u;
    }

    // ── V: q8 quantize → Vc + Vsc (no norm, no rope) ──
    let vbase = krow * p.D;
    wi = 0u;
    for (var w = lane; w < W4; w = w + SG) {
      let vv = vec4<f32>(vbuf[vbase + w * 4u], vbuf[vbase + w * 4u + 1u],
                         vbuf[vbase + w * 4u + 2u], vbuf[vbase + w * 4u + 3u]);
      vals[wi] = vv;
      wi = wi + 1u;
      wabs[w] = max(max(abs(vv.x), abs(vv.y)), max(abs(vv.z), abs(vv.w)));
    }
    workgroupBarrier();
    if (lane < p.D / 32u) {
      var m = 0.0;
      for (var i = 0u; i < 8u; i = i + 1u) { m = max(m, wabs[lane * 8u + i]); }
      let sc = max(m, 1e-30);
      wblk[lane] = sc;
      vcs[(p.kOutRow0 + krow) * (p.D / 32u) + lane] = sc;
    }
    workgroupBarrier();
    wi = 0u;
    for (var w = lane; w < W4; w = w + SG) {
      vcq[(p.kOutRow0 + krow) * W4 + w] = pack4x8snorm(vals[wi] / wblk[w >> 3u]);
      wi = wi + 1u;
    }
  }
}
