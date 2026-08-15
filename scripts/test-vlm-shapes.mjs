// VLM shape test with text-before-image ordering (matching llama.cpp)
import { createReadStream, existsSync, statSync, writeFileSync, unlinkSync } from 'node:fs'
import { join, dirname, normalize, sep, extname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { homedir } from 'node:os'
import { createServer } from 'node:http'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const downloads = join(homedir(), 'Downloads')
const MIME = { '.html':'text/html', '.js':'text/javascript', '.json':'application/json', '.gguf':'application/octet-stream', '.bin':'application/octet-stream', '.map':'application/json', '.d.ts':'text/plain', '.css':'text/css', '.wgsl':'text/plain', '.png':'image/png' }

function resolvePath(rel) {
  const r = join(root, rel.replace(/^\//, '')); if (existsSync(r)) return r
  const d = join(downloads, rel.replace(/^\//, '')); if (existsSync(d)) return d
  return null
}

function serve(port = 8888) {
  return new Promise((resolve) => {
    const server = createServer((req, res) => {
      const url = new URL(req.url, 'http://localhost')
      const rel = decodeURIComponent(normalize(url.pathname).replace(/^([/\\])+/, ''))
      if (rel.split(sep).includes('..')) { res.writeHead(403).end(); return }
      const path = resolvePath(rel)
      if (!path || !existsSync(path) || !statSync(path).isFile()) { res.writeHead(404).end(); return }
      const sz = statSync(path).size
      const range = req.headers.range
      if (range) {
        const m = /bytes=(\d+)-(\d*)/.exec(range)
        if (m) {
          const start = parseInt(m[1], 10), end = m[2] ? parseInt(m[2], 10) : sz - 1
          if (start >= sz || end >= sz || start > end) { res.writeHead(416).end(); return }
          res.writeHead(206, { 'Content-Range': `bytes ${start}-${end}/${sz}`, 'Accept-Ranges': 'bytes', 'Content-Length': String(end-start+1), 'Content-Type': MIME[extname(path).toLowerCase()] || 'application/octet-stream' })
          createReadStream(path, { start, end }).pipe(res); return
        }
      }
      res.writeHead(200, { 'Content-Type': MIME[extname(path).toLowerCase()] || 'application/octet-stream', 'Accept-Ranges': 'bytes', 'Content-Length': String(sz) })
      createReadStream(path).pipe(res)
    })
    server.listen(port, () => resolve(server))
  })
}

function findChrome() {
  const paths = ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', '/Applications/Chromium.app/Contents/MacOS/Chromium']
  for (const p of paths) if (existsSync(p)) return p
  throw new Error('Chrome not found')
}

async function main() {
  const port = 8888
  const server = await serve(port)
  const modelFile = 'Bonsai-27B-Q1_0.gguf'
  const mmprojFile = 'Bonsai-27B-mmproj-Q8_0.gguf'
  const tokenizerFile = 'tokenizer.json'
  const tokenizerConfigFile = 'tokenizer_config.json'

  for (const f of [modelFile, mmprojFile, tokenizerFile, tokenizerConfigFile]) {
    const p = resolvePath(f)
    if (!p) { console.error(`File not found: ${f}`); process.exit(1) }
  }

  const html = `<!doctype html>
<html><head><meta charset="utf-8"><title>VLM Shape Test</title></head>
<body><pre id="out">idle</pre>
<script type="module">
  const out = document.getElementById('out')
  const log = (s) => { out.textContent += '\\n' + s }

  function makeShapeImage() {
    const w = 256, h = 256
    const canvas = document.createElement('canvas')
    canvas.width = w; canvas.height = h
    const ctx = canvas.getContext('2d')
    ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, w, h)
    const cx = 64, cy = 64, r = 40
    // Top-left: red circle
    ctx.fillStyle = '#ff0000'; ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill()
    // Top-right: blue square
    ctx.fillStyle = '#0000ff'; ctx.fillRect(w - cx - r, cy - r, r * 2, r * 2)
    // Bottom-left: green triangle
    ctx.fillStyle = '#00aa00'; ctx.beginPath()
    ctx.moveTo(cx - r, h - cy + r); ctx.lineTo(cx + r, h - cy + r); ctx.lineTo(cx, h - cy - r); ctx.closePath(); ctx.fill()
    // Bottom-right: yellow star
    ctx.fillStyle = '#ffcc00'
    const sx = w - cx, sy = h - cy
    ctx.beginPath()
    for (let i = 0; i < 10; i++) {
      const angle = (i * Math.PI) / 5 - Math.PI / 2
      const radius = i % 2 === 0 ? r : r * 0.4
      const x = sx + Math.cos(angle) * radius, y = sy + Math.sin(angle) * radius
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y)
    }
    ctx.closePath(); ctx.fill()
    const imageData = ctx.getImageData(0, 0, w, h)
    const rgb = new Float32Array(3 * w * h)
    for (let i = 0; i < w * h; i++) { rgb[i*3] = imageData.data[i*4]/255; rgb[i*3+1] = imageData.data[i*4+1]/255; rgb[i*3+2] = imageData.data[i*4+2]/255 }
    return { rgb, width: w, height: h, frames: 1 }
  }

  async function run() {
    out.textContent = ''
    try {
      const { createEngine } = await import('/dist/index.js')
      const { fromGguf } = await import('/dist/gguf.js')
      const { createChat } = await import('/dist/chat.js')

      log('1. Generating shape image (512x512)...')
      const image = makeShapeImage()

      log('2. Creating engine...')
      const t0 = performance.now()
      const parsed = await fromGguf('http://localhost:${port}/${modelFile}')
      const engine = await createEngine({ ...parsed, visionMmprojUrl: 'http://localhost:${port}/${mmprojFile}', warmShaders: false })
      log('   Engine created in ' + (performance.now() - t0).toFixed(0) + 'ms')

      log('3. Loading tokenizer...')
      const chat = await createChat(engine, { tokenizerJsonUrl: 'http://localhost:${port}/${tokenizerFile}', tokenizerConfigUrl: 'http://localhost:${port}/${tokenizerConfigFile}' })
      const tk = chat.tokenizer

      log('4. Running vision tower...')
      const t2 = performance.now()
      const visResult = await engine.visionForward([image])
      const numMerged = visResult.numPatches
      log('   Vision tower: ' + numMerged + ' patches in ' + (performance.now() - t2).toFixed(0) + 'ms')

      log('5. Building prompt (text-before-image, matching llama.cpp)...')
      const promptText = 'What do you see in this image? Describe everything you can identify.'

      // TEXT FIRST, then image — matches llama.cpp ordering
      // Add system prompt to match llama.cpp's Qwen25VLChatHandler
      const messages = [
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: [
          { type: 'text', text: promptText },
          { type: 'image', image },
        ] },
      ]

      const imageTokenId = 248056
      const promptString = tk.applyChatTemplate(messages, { addGenerationPrompt: true, enableThinking: true })
      log('   Prompt: ' + JSON.stringify(promptString))

      const promptIds = tk.encode(promptString, false)
      log('   Encoded: ' + promptIds.length + ' tokens')

      const imgPositions = []
      for (let i = 0; i < promptIds.length; i++) { if (promptIds[i] === imageTokenId) imgPositions.push(i) }
      log('   Image positions: ' + imgPositions.length + ' (need ' + numMerged + ')')

      let finalIds = promptIds, finalImgPositions = imgPositions
      if (imgPositions.length === 1 && numMerged > 1) {
        const padPos = imgPositions[0]
        finalIds = [...promptIds.slice(0, padPos), ...new Array(numMerged).fill(imageTokenId), ...promptIds.slice(padPos + 1)]
        finalImgPositions = []; for (let i = 0; i < numMerged; i++) finalImgPositions.push(padPos + i)
        log('   Expanded to ' + finalIds.length + ' tokens')
      } else if (imgPositions.length !== numMerged) {
        log('FATAL: position mismatch'); log('TEST_DONE'); return
      }

      log('6. Generating...')
      const t3 = performance.now()
      const result = await engine.generateWithImages(finalIds, finalImgPositions, [image], {
        maxTokens: 512, temperature: 0.1, repetitionPenalty: 1.1, noRepeatNgramSize: 3,
      })
      log('   ' + result.tokens.length + ' tokens in ' + (performance.now() - t3).toFixed(0) + 'ms')
      log('   Prefill: ' + (result.prefillMs?.toFixed(0) || '?') + 'ms, Decode: ' + (result.decodeMs?.toFixed(0) || '?') + 'ms')

      const response = tk.decode(result.tokens, true)
      log('\\n=== VLM RESPONSE ===')
      log(response)
      log('=== END ===')
      log('\\nGround truth: top-left=red circle, top-right=blue square, bottom-left=green triangle, bottom-right=yellow star')
      log('TEST_DONE')
    } catch (e) {
      log('FATAL: ' + e.message); log(e.stack || ''); log('TEST_DONE')
    }
  }
  run()
</script>
</body></html>`

  const htmlPath = join(root, 'examples', 'test-vlm-shapes.html')
  writeFileSync(htmlPath, html)

  const { chromium } = await import('playwright-core')
  const browser = await chromium.launch({ headless: false, executablePath: findChrome(), args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--disable-gpu-sandbox', '--enable-webgpu-developer-features'] })
  const page = await browser.newPage()
  page.on('console', msg => console.log('  [console]', msg.text()))
  page.on('pageerror', err => console.log('  [pageerror]', err.message))
  await page.goto(`http://localhost:${port}/examples/test-vlm-shapes.html`, { waitUntil: 'domcontentloaded' })

  const deadline = Date.now() + 600000
  let lastContent = ''
  while (Date.now() < deadline) {
    await new Promise(r => setTimeout(r, 10000))
    const content = await page.evaluate(() => document.getElementById('out')?.textContent || '')
    if (content !== lastContent) { console.log('  [progress]', content.slice(-200)); lastContent = content }
    if (content.includes('TEST_DONE')) break
  }
  const output = await page.evaluate(() => document.getElementById('out').textContent)
  console.log(output)
  await browser.close(); server.close()
  try { unlinkSync(htmlPath) } catch {}
  if (output.includes('FATAL')) process.exit(1)
  process.exit(0)
}

main().catch(e => { console.error(e); process.exit(1) })
