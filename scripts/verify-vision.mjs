// Headless driver for vision tower GPU vs CPU verification.
// Serves the repo root + ~/Downloads (for the GGUF files), launches Chrome with
// WebGPU, loads verify-vision.html, clicks Run, waits for the result.
//
//   node scripts/verify-vision.mjs
//
// Exits non-zero if the verification fails (cosine < 0.999).
import { chromium } from 'playwright-core'
import { createServer } from 'node:http'
import { existsSync, statSync, createReadStream } from 'node:fs'
import { join, dirname, extname, normalize, sep } from 'node:path'
import { fileURLToPath } from 'node:url'
import { homedir } from 'node:os'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const downloads = join(homedir(), 'Downloads')

const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.json': 'application/json', '.map': 'application/json',
  '.bin': 'application/octet-stream', '.gguf': 'application/octet-stream',
  '.css': 'text/css', '.wgsl': 'text/plain', '.d.ts': 'text/plain',
}

// Resolve a path from either the repo root or ~/Downloads
function resolvePath(rel) {
  // Try repo root first
  const repoPath = join(root, rel.replace(/^\//, ''))
  if (existsSync(repoPath)) return repoPath
  // Try ~/Downloads
  const dlPath = join(downloads, rel.replace(/^\//, ''))
  if (existsSync(dlPath)) return dlPath
  return null
}

function serve(port = 8888) {
  return new Promise((resolve) => {
    const server = createServer((req, res) => {
      const url = new URL(req.url, 'http://localhost')
      let rel = normalize(decodeURIComponent(url.pathname)).replace(/^([/\\])+/, '')
      // Strip query
      rel = rel.split('?')[0]

      if (rel.split(sep).includes('..')) {
        res.writeHead(403).end()
        return
      }

      const path = resolvePath(rel)
      if (!path || !existsSync(path) || !statSync(path).isFile()) {
        res.writeHead(404).end(`Not found: ${rel}`)
        return
      }

      // Range support (critical for GGUF range requests)
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
          })
          createReadStream(path, { start, end }).pipe(res)
          return
        }
      }

      // Full file (stream — never buffer multi-GB files)
      res.writeHead(200, {
        'content-type': MIME[extname(path)] ?? 'application/octet-stream',
        'content-length': size,
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

  // Check files exist
  const modelPath = resolvePath(modelFile)
  const mmprojPath = resolvePath(mmprojFile)
  if (!modelPath) {
    console.error(`Model not found: ${modelFile} (looked in ${root} and ${downloads})`)
    process.exit(1)
  }
  if (!mmprojPath) {
    console.error(`Mmproj not found: ${mmprojFile} (looked in ${root} and ${downloads})`)
    process.exit(1)
  }
  console.log(`Model: ${modelPath} (${(statSync(modelPath).size / 1e9).toFixed(2)} GB)`)
  console.log(`Mmproj: ${mmprojPath} (${(statSync(mmprojPath).size / 1e6).toFixed(0)} MB)`)

  const url = `http://localhost:${port}/examples/verify-vision.html` +
    `?model=http://localhost:${port}/${modelFile}` +
    `&mmproj=http://localhost:${port}/${mmprojFile}` +
    `&run=1`

  // Find Chrome
  const chromePaths = [
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/usr/bin/google-chrome',
    '/usr/bin/chromium-browser',
  ]
  const chromePath = chromePaths.find(p => existsSync(p))
  if (!chromePath) {
    console.error('Chrome not found')
    process.exit(1)
  }

  console.log(`Launching Chrome: ${chromePath}`)
  const browser = await chromium.launch({
    executablePath: chromePath,
    headless: false,  // WebGPU needs headed mode on most platforms
    args: [
      '--enable-unsafe-webgpu',
      '--enable-features=Vulkan',
      '--disable-gpu-sandbox',
      '--enable-webgpu-developer-features',
    ],
  })

  const page = await browser.newPage()
  page.on('console', msg => {
    const text = msg.text()
    console.log(`  [browser] ${text}`)
  })
  page.on('pageerror', err => console.error(`  [pageerror] ${err.message}`))

  console.log(`Navigating to ${url}`)
  await page.goto(url, { waitUntil: 'domcontentloaded' })

  // Fill in the URLs from query params
  const params = new URL(url).searchParams
  await page.fill('#modelUrl', params.get('model'))
  await page.fill('#mmprojUrl', params.get('mmproj'))

  // Click Run
  console.log('Clicking Run...')
  await page.click('#run')

  // Wait for completion (look for PASS or FAIL in the output)
  console.log('Waiting for verification to complete...')
  let result = null
  for (let i = 0; i < 600; i++) {  // 10 min timeout
    await page.waitForTimeout(1000)
    const text = await page.textContent('#out')
    if (text.includes('VISION VERIFICATION PASSED')) {
      result = 'PASS'
      break
    }
    if (text.includes('VISION VERIFICATION FAILED') || text.includes('ERROR:')) {
      result = 'FAIL'
      break
    }
    // Progress
    const lines = text.split('\n').filter(l => l.trim())
    if (lines.length > 0 && i % 5 === 0) {
      console.log(`  [progress] ${lines[lines.length - 1]}`)
    }
  }

  // Print full output
  const fullText = await page.textContent('#out')
  console.log('\n=== Full output ===')
  console.log(fullText)

  await browser.close()
  server.close()

  if (result === 'PASS') {
    console.log('\n✅ Vision verification PASSED')
    process.exit(0)
  } else if (result === 'FAIL') {
    console.log('\n❌ Vision verification FAILED')
    process.exit(1)
  } else {
    console.log('\n⏱ Vision verification TIMED OUT')
    process.exit(2)
  }
}

main().catch(e => { console.error(e); process.exit(1) })
