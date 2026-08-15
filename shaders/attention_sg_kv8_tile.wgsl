// Tiled flash-attention: scans ONE tile [tileStart, tileEnd) of the q8 KV cache.
enable subgroups;
override SG: u32 = 32u;
struct Params { S: u32, H: u32, KV: u32, D: u32, posBase: u32, Ltot: u32, tileStart: u32, tileEnd: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> Kq: array<u32>;
@group(0) @binding(3) var<storage, read> Vq: array<u32>;
@group(0) @binding(4) var<storage, read> Ks: array<f32>;
@group(0) @binding(5) var<storage, read> Vs: array<f32>;
@group(0) @binding(6) var<storage, read> m_in: array<f32>;
@group(0) @binding(7) var<storage, read> l_in: array<f32>;
@group(0) @binding(8) var<storage, read> acc_in: array<f32>;
@group(0) @binding(9) var<storage, read_write> m_out: array<f32>;
@group(0) @binding(10) var<storage, read_write> l_out: array<f32>;
@group(0) @binding(11) var<storage, read_write> acc_out: array<f32>;
@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let idx = wg.x;
  if (idx >= p.S * p.H) { return; }
  let h = idx % p.H;
  let qi = idx / p.H;
  let pos = p.posBase + qi;
  let kvh = h / (p.H / p.KV);
  let qb = (qi * p.H + h) * p.D;
  let inv = 1.0 / sqrt(f32(p.D));
  let W4 = p.D / 4u;
  let B32 = p.D / 32u;
  var m = m_in[idx];
  var l = l_in[idx];
  let end = min(p.tileEnd, pos + 1u);
  for (var w = lane; w < W4; w = w + SG) {
    let ab = idx * p.D + w * 4u;
    acc_out[ab] = acc_in[ab];
    acc_out[ab + 1u] = acc_in[ab + 1u];
    acc_out[ab + 2u] = acc_in[ab + 2u];
    acc_out[ab + 3u] = acc_in[ab + 3u];
  }
  for (var j = p.tileStart; j < end; j = j + 1u) {
    let rowQ = (j * p.KV + kvh) * W4;
    let rowS = (j * p.KV + kvh) * B32;
    var part = 0.0;
    for (var w = lane; w < W4; w = w + SG) {
      let kw = unpack4x8snorm(Kq[rowQ + w]) * Ks[rowS + (w >> 3u)];
      let qv = vec4<f32>(q[qb + w * 4u], q[qb + w * 4u + 1u], q[qb + w * 4u + 2u], q[qb + w * 4u + 3u]);
      part = part + dot(qv, kw);
    }
    let score = subgroupAdd(part) * inv;
    let mnew = max(m, score);
    let corr = exp(m - mnew);
    let wgt = exp(score - mnew);
    l = l * corr + wgt;
    for (var w = lane; w < W4; w = w + SG) {
      let vw = unpack4x8snorm(Vq[rowQ + w]) * Vs[rowS + (w >> 3u)];
      let ab = idx * p.D + w * 4u;
      acc_out[ab] = acc_out[ab] * corr + wgt * vw.x;
      acc_out[ab + 1u] = acc_out[ab + 1u] * corr + wgt * vw.y;
      acc_out[ab + 2u] = acc_out[ab + 2u] * corr + wgt * vw.z;
      acc_out[ab + 3u] = acc_out[ab + 3u] * corr + wgt * vw.w;
    }
    m = mnew;
  }
  m_out[idx] = m;
  l_out[idx] = l;
}
