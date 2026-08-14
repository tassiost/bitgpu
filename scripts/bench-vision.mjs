#!/usr/bin/env node
// Vision tower benchmark: runs the existing verify-e2e-vision.mjs multiple times,
// collects results, and prints theoretical bottleneck analysis.
//
// Usage: node scripts/bench-vision.mjs [screenshot|red] [runs]

import { execSync } from 'node:child_process'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const imageType = process.argv[2] || 'screenshot'
const runs = parseInt(process.argv[3] || '3')

// Vision tower architecture constants (Qwen3-VL Bonsai-27B)
const ARCH = {
  depth: 27, hidden: 1152, inter: 4304, heads: 16, headDim: 72,
  mergedDim: 4608, projDim: 5120,
}
const MEM_BW = 100e9  // M3 unified memory ~100 GB/s
const GPU_TFLOPS = 5e12  // M3 ~5 TFLOPS f32 sustained

function analyze(numPatches) {
  const { depth, hidden: H, inter, heads, headDim: hd } = ARCH
  const M = numPatches
  const q8 = (n) => Math.ceil(n / 32) * 2 + n
  const f16 = (n) => n * 2

  const w = {
    qkv: q8(3 * H * H), attnOut: q8(H * H), ffnUp: q8(H * inter),
    ffnDown: f16(inter * H), ln: H * 4 * 2 * 2,
  }
  w.total = w.qkv + w.attnOut + w.ffnUp + w.ffnDown + w.ln

  const a = {
    ln1: M * H * 4 * 2, qkv: M * H * 4 + M * 3 * H * 4,
    rope: M * 3 * H * 4 * 2, attn: M * 2 * H * 4 + M * H * 4 + M * H * 4,
    attnOut: M * H * 4 * 3, ln2: M * H * 4 * 2,
    ffnUp: M * H * 4 + M * inter * 4, ffnDown: M * inter * 4 + M * H * 4 * 2,
  }
  a.total = a.ln1 + a.qkv + a.rope + a.attn + a.attnOut + a.ln2 + a.ffnUp + a.ffnDown

  const f = {
    qkv: 2 * M * 3 * H * H, attnOut: 2 * M * H * H,
    qk: 2 * M * M * H * heads, sv: 2 * M * M * H * heads,
    ffnUp: 2 * M * H * inter, ffnDown: 2 * M * inter * H,
  }
  f.total = f.qkv + f.attnOut + f.qk + f.sv + f.ffnUp + f.ffnDown

  const wT = w.total * depth, aT = a.total * depth, fT = f.total * depth
  const tW = wT / MEM_BW, tA = aT / MEM_BW, tF = fT / GPU_TFLOPS
  const tMem = tW + tA

  return { w, a, f, wT, aT, fT, tW, tA, tF, tMem, tRoof: Math.max(tMem, tF), M, depth }
}

function printReport(results) {
  const patches = results[0].patches
  const times = results.map(r => r.ms)
  const minMs = Math.min(...times)
  const maxMs = Math.max(...times)
  const avgMs = times.reduce((a, b) => a + b, 0) / times.length

  const A = analyze(patches)

  console.log(`\n${'═'.repeat(70)}`)
  console.log(`  VISION TOWER BENCHMARK — ${imageType} (${patches} patches, ${runs} runs)`)
  console.log(`${'═'.repeat(70)}`)
  console.log(`  Times: ${times.join('ms, ')}ms`)
  console.log(`  Min: ${minMs}ms  Max: ${maxMs}ms  Avg: ${avgMs.toFixed(0)}ms`)
  console.log(`  Per-layer: ${(minMs / ARCH.depth).toFixed(1)}ms/layer`)

  console.log(`\n── Weight Bandwidth (per layer, Q8+F16) ──`)
  const w = A.w
  console.log(`  QKV (Q8):       ${(w.qkv / 1024).toFixed(0)} KB  (${(w.qkv / w.total * 100).toFixed(0)}%)`)
  console.log(`  AttnOut (Q8):   ${(w.attnOut / 1024).toFixed(0)} KB  (${(w.attnOut / w.total * 100).toFixed(0)}%)`)
  console.log(`  FFN up (Q8):    ${(w.ffnUp / 1024).toFixed(0)} KB  (${(w.ffnUp / w.total * 100).toFixed(0)}%)`)
  console.log(`  FFN down (F16): ${(w.ffnDown / 1024).toFixed(0)} KB  (${(w.ffnDown / w.total * 100).toFixed(0)}%)`)
  console.log(`  LayerNorms:     ${(w.ln / 1024).toFixed(0)} KB  (${(w.ln / w.total * 100).toFixed(0)}%)`)
  console.log(`  Total/layer:    ${(w.total / 1e6).toFixed(2)} MB`)
  console.log(`  Total/all:      ${(A.wT / 1e6).toFixed(1)} MB`)

  console.log(`\n── Activation Bandwidth (per layer, f32) ──`)
  const a = A.a
  console.log(`  LN1:        ${(a.ln1 / 1024).toFixed(0)} KB`)
  console.log(`  QKV proj:   ${(a.qkv / 1024).toFixed(0)} KB`)
  console.log(`  RoPE:       ${(a.rope / 1024).toFixed(0)} KB`)
  console.log(`  Attention:  ${(a.attn / 1024).toFixed(0)} KB`)
  console.log(`  AttnOut:    ${(a.attnOut / 1024).toFixed(0)} KB`)
  console.log(`  LN2:        ${(a.ln2 / 1024).toFixed(0)} KB`)
  console.log(`  FFN up:     ${(a.ffnUp / 1024).toFixed(0)} KB`)
  console.log(`  FFN down:   ${(a.ffnDown / 1024).toFixed(0)} KB`)
  console.log(`  Total/layer:${(a.total / 1e6).toFixed(2)} MB`)
  console.log(`  Total/all:  ${(A.aT / 1e6).toFixed(1)} MB`)

  console.log(`\n── Compute (per layer) ──`)
  const f = A.f
  console.log(`  QKV matmul:     ${(f.qkv / 1e6).toFixed(1)} MFLOP`)
  console.log(`  AttnOut matmul: ${(f.attnOut / 1e6).toFixed(1)} MFLOP`)
  console.log(`  Q·K^T:          ${(f.qk / 1e6).toFixed(1)} MFLOP`)
  console.log(`  Softmax·V:      ${(f.sv / 1e6).toFixed(1)} MFLOP`)
  console.log(`  FFN up:         ${(f.ffnUp / 1e6).toFixed(1)} MFLOP`)
  console.log(`  FFN down:       ${(f.ffnDown / 1e6).toFixed(1)} MFLOP`)
  console.log(`  Total/layer:    ${(f.total / 1e6).toFixed(1)} MFLOP`)
  console.log(`  Total/all:      ${(A.fT / 1e9).toFixed(2)} GFLOP`)

  console.log(`\n── Theoretical Minimum (@ 100 GB/s, 5 TFLOPS) ──`)
  console.log(`  Weight reads:    ${(A.tW * 1000).toFixed(1)} ms`)
  console.log(`  Activation R/W:  ${(A.tA * 1000).toFixed(1)} ms`)
  console.log(`  Total memory:    ${(A.tMem * 1000).toFixed(1)} ms`)
  console.log(`  Compute:         ${(A.tF * 1000).toFixed(1)} ms`)
  console.log(`  Roofline:        ${(A.tRoof * 1000).toFixed(1)} ms`)
  const bottleneck = A.tMem > A.tF ? 'MEMORY-BOUND' : 'COMPUTE-BOUND'
  console.log(`  Bottleneck:      ${bottleneck}`)

  console.log(`\n── Efficiency ──`)
  console.log(`  Actual (min):    ${minMs}ms`)
  console.log(`  Roofline:        ${(A.tRoof * 1000).toFixed(1)}ms`)
  console.log(`  Efficiency:      ${(A.tRoof * 1000 / minMs * 100).toFixed(1)}% of roofline`)
  console.log(`  Gap:             ${(minMs / (A.tRoof * 1000)).toFixed(1)}× slower than theoretical min`)

  const wPct = (A.tW / A.tMem * 100).toFixed(0)
  const aPct = (A.tA / A.tMem * 100).toFixed(0)
  console.log(`\n── Bandwidth Split ──`)
  console.log(`  Weights:     ${wPct}% (${(A.tW * 1000).toFixed(1)}ms)`)
  console.log(`  Activations: ${aPct}% (${(A.tA * 1000).toFixed(1)}ms)`)
  console.log(`  Compute:     ${(A.tF * 1000).toFixed(1)}ms (${(A.tF * 1000 / minMs * 100).toFixed(0)}% of actual)`)

  // What-if: f16 activations
  const aF16 = A.aT / 2  // halve activation bandwidth
  const tMemF16 = A.tW + aF16 / MEM_BW
  console.log(`\n── What-if: F16 Activations ──`)
  console.log(`  Activation BW:  ${(A.aT / 1e6).toFixed(1)} MB → ${(aF16 / 1e6).toFixed(1)} MB`)
  console.log(`  Memory time:    ${(A.tMem * 1000).toFixed(1)}ms → ${(tMemF16 * 1000).toFixed(1)}ms`)
  console.log(`  Speedup:        ${(A.tMem / tMemF16).toFixed(2)}×`)

  // What-if: f16 weights for Q8 layers
  const wF16 = (A.wT - w.ffnDown * ARCH.depth) + (w.qkv + w.attnOut + w.ffnUp) * 2 * ARCH.depth / 1.0625
  // Actually Q8 is already smaller than f16, so f16 weights would be BIGGER. Skip this.

  // What-if: Q4 weights for Q8 layers
  const q4 = (n) => Math.ceil(n / 32) * 2 + n / 2  // 4-bit: half the packed bytes + same scales
  const wQ4 = {
    qkv: q4(3 * ARCH.hidden * ARCH.hidden),
    attnOut: q4(ARCH.hidden * ARCH.hidden),
    ffnUp: q4(ARCH.hidden * ARCH.inter),
    ffnDown: w.ffnDown,  // already f16
    ln: w.ln,
  }
  wQ4.total = wQ4.qkv + wQ4.attnOut + wQ4.ffnUp + wQ4.ffnDown + wQ4.ln
  const wQ4T = wQ4.total * ARCH.depth
  const tMemQ4 = wQ4T / MEM_BW + A.aT / MEM_BW
  console.log(`\n── What-if: Q4 Weights (Q8→Q4 for matmul layers) ──`)
  console.log(`  Weight BW:      ${(A.wT / 1e6).toFixed(1)} MB → ${(wQ4T / 1e6).toFixed(1)} MB`)
  console.log(`  Memory time:    ${(A.tMem * 1000).toFixed(1)}ms → ${(tMemQ4 * 1000).toFixed(1)}ms`)
  console.log(`  Speedup:        ${(A.tMem / tMemQ4).toFixed(2)}×`)

  console.log(`${'═'.repeat(70)}\n`)
}

// Run the existing verify script multiple times and parse results
async function main() {
  const results = []

  // Set image type by editing the verify script temporarily
  const verifyScript = join(root, 'scripts/verify-e2e-vision.mjs')
  const { readFileSync, writeFileSync } = await import('node:fs')
  const original = readFileSync(verifyScript, 'utf8')

  for (let run = 0; run < runs; run++) {
    // Set image type
    const imgLine = imageType === 'screenshot'
      ? '    `&image=file` +\n    `&imageUrl=http://localhost:${port}/test-screenshot.png` +\n    `&run=1`'
      : '    `&image=synthetic-red` +\n    `&run=1`'
    const modified = original.replace(
      /    `&image=[^`]+`[\s\S]*?`&run=1`/,
      imgLine
    )
    writeFileSync(verifyScript, modified)

    console.log(`\n── Run ${run + 1}/${runs} (${imageType}) ──`)

    try {
      const output = execSync(
        `pkill -9 -f "Google Chrome" 2>/dev/null; sleep 3; cd "${root}" && node scripts/verify-e2e-vision.mjs 2>&1`,
        { timeout: 300000, encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'] }
      )
      console.log(output)

      const visionMatch = output.match(/Vision tower: (\d+) image embeddings in (\d+)ms/)
      if (visionMatch) {
        results.push({ patches: parseInt(visionMatch[1]), ms: parseInt(visionMatch[2]) })
      }
    } catch (e) {
      console.log(`  Run failed: ${e.message?.substring(0, 200)}`)
      if (e.stdout) console.log(e.stdout.substring(0, 500))
    }

    // Cooldown between runs
    if (run < runs - 1) {
      console.log('  Cooling down 90s...')
      await new Promise(r => setTimeout(r, 90000))
    }
  }

  // Restore original script
  writeFileSync(verifyScript, original)

  if (results.length === 0) {
    console.log('\nNo results collected!')
    return
  }

  printReport(results)
}

main().catch(e => { console.error(e); process.exit(1) })
