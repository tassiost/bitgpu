// Headless driver for GPU vs CPU vision tower verification.
// Compares the GPU dispatch output against the CPU reference implementation.
import { existsSync, statSync, createReadStream } from 'node:fs'
import { join, dirname, extname, normalize, sep } from 'node:path'
import { fileURLToPath } from 'node:url'
import { homedir } from 'node:os'
import { createServer } from 'node:http'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const downloads = join(homedir(), 'Downloads')

const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.json': 'application/json', '.map': 'application/json',
  '.bin': 'application/octet-stream', '.gguf': 'application/octet-stream',
  '.css': 'text/css', '.wgsl': 'text/plain', '.d.ts': 'text/plain',
}

function resolvePath(rel) {
  const repoPath = join(root, rel.replace(/^\//, ''))
  if (existsSync(repoPath)) return repoPath
  const dlPath = join(downloads, rel.replace(/^\//, ''))
  if (existsSync(dlPath)) return dlPath
  return null
}

function serve(port = 8888) {
  return new Promise((resolve) => {
    const server = createServer((req, res) => {
      const url = new URL(req.url, 'http://localhost')
      const rel = decodeURIComponent(normalize(url.pathname).replace(/^([/\\])+/, ''))
      if (rel.split(sep).includes('..')) { res.writeHead(403).end(); return }
      const path = resolvePath(rel)
      if (!path || !existsSync(path) || !statSync(path).isFile()) {
        res.writeHead(404).end(`Not found: ${rel}`)
        return
      }
      const size = statSync(path).size
      const range = req.headers.range
      if (range) {
        const m = /bytes=(\d+)-(\d*)/.exec(range)
        if (m) {
          const start = parseInt(m[1])
          const end = m[2] ? parseInt(m[2]) : size - 1
          const len = end - start + 1
          res.writeHead(206, {
            'content-type': MIME[extname(path)] ?? 'application/octet-stream',
            'content-range': `bytes ${start}-${end}/${size}`,
            'content-length': len,
            'accept-ranges': 'bytes',
            'Access-Control-Allow-Origin': '*',
          })
          createReadStream(path, { start, end }).pipe(res)
          return
        }
      }
      res.writeHead(200, {
        'Content-Type': MIME[extname(path)] || 'application/octet-stream',
        'Access-Control-Allow-Origin': '*',
        'Content-Length': size,
        'accept-ranges': 'bytes',
      })
      createReadStream(path).pipe(res)
    })
    server.listen(port, () => resolve(server))
  })
}

async function main() {
  const port = 8888
  const server = await serve(port)
  console.log(`Server on :${port}`)

  const modelFile = 'Bonsai-27B-Q1_0.gguf'
  const mmprojFile = 'Bonsai-27B-mmproj-Q8_0.gguf'

  for (const f of [modelFile, mmprojFile]) {
    const p = resolvePath(f)
    if (!p) { console.error(`File not found: ${f}`); process.exit(1) }
    console.log(`  ${f}: ${p} (${(statSync(p).size / 1e6).toFixed(1)} MB)`)
  }

  const imageType = process.argv[2] || 'synthetic-red'
  const url = `http://localhost:${port}/examples/verify-vision.html` +
    `?model=http://localhost:${port}/${modelFile}` +
    `&mmproj=http://localhost:${port}/${mmprojFile}` +
    `&image=${imageType}` +
    `&run=1`

  const { chromium } = await import('playwright-core')
  const fs = await import('node:fs')
  const exe = [
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Chromium.app/Contents/MacOS/Chromium',
  ].find(p => fs.existsSync(p))
  if (!exe) { console.error('Chrome not found'); process.exit(1) }

  const browser = await chromium.launch({
    headless: false,
    executablePath: exe,
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--disable-gpu-sandbox', '--enable-webgpu-developer-features'],
  })
  const page = await browser.newPage()
  page.on('console', msg => console.log('  [console]', msg.text()))
  page.on('pageerror', err => console.log('  [pageerror]', err.message))

  console.log(`\nNavigating to ${url.slice(0, 80)}...`)
  await page.goto(url, { waitUntil: 'domcontentloaded' })

  console.log('Waiting for verification result...')
  let done = false
  let output = ''
  const start = Date.now()
  while (!done && Date.now() - start < 300000) {
    await page.waitForTimeout(5000)
    output = await page.evaluate(() => document.getElementById('out')?.textContent || '')
    if (output.includes('PASSED') || output.includes('FAILED') || output.includes('ERROR')) {
      done = true
    }
    if (output.length > 0) {
      const lines = output.split('\n').filter(l => l.trim())
      if (lines.length > 0) console.log(`  [${((Date.now()-start)/1000).toFixed(0)}s] ${lines[lines.length-1].slice(0,100)}`)
    }
  }

  console.log('\n' + output)
  await browser.close()
  server.close()

  if (output.includes('VISION VERIFICATION PASSED')) {
    console.log('\n✅ GPU vs CPU VERIFICATION PASSED')
    process.exit(0)
  } else {
    console.log('\n❌ GPU vs CPU VERIFICATION FAILED')
    process.exit(1)
  }
}

main().catch(e => { console.error(e); process.exit(1) })
