#!/usr/bin/env python3
"""Reference VLM test using llama-cpp-python with Qwen2.5-VL chat handler.
Generates a 512x512 image with 4 shapes in each corner, then asks the VLM
to describe what it sees — using the SAME Q1_0 model + Q8_0 mmproj.
This isolates whether hallucination is from our WebGPU pipeline or the quantization.
"""
from PIL import Image, ImageDraw
import sys, os, time, math, base64, io

# 1. Generate the same 512x512 shape image
W, H = 512, 512
img = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(img)

cx, cy, r = 128, 128, 80

# Top-left: red circle
draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill="#ff0000")
# Top-right: blue square
draw.rectangle([W - cx - r, cy - r, W - cx + r, cy + r], fill="#0000ff")
# Bottom-left: green triangle
draw.polygon([(cx - r, H - cy + r), (cx + r, H - cy + r), (cx, H - cy - r)], fill="#00aa00")
# Bottom-right: yellow star
sx, sy = W - cx, H - cy
star_points = []
for i in range(10):
    angle = (i * math.pi) / 5 - math.pi / 2
    radius = r if i % 2 == 0 else r * 0.4
    star_points.append((sx + math.cos(angle) * radius, sy + math.sin(angle) * radius))
draw.polygon(star_points, fill="#ffcc00")

img_path = "/tmp/vlm_shapes_test.png"
img.save(img_path)
print(f"Image saved to {img_path} ({W}x{H})")

# Encode as base64 data URI
buf = io.BytesIO()
img.save(buf, format="PNG")
img_b64 = base64.b64encode(buf.getvalue()).decode()
data_uri = f"data:image/png;base64,{img_b64}"

# 2. Load model with VLM chat handler
model_path = os.path.expanduser("~/Downloads/Bonsai-27B-Q1_0.gguf")
mmproj_path = os.path.expanduser("~/Downloads/Bonsai-27B-mmproj-Q8_0.gguf")

for p in [model_path, mmproj_path]:
    if not os.path.exists(p):
        print(f"ERROR: Not found: {p}")
        sys.exit(1)

print(f"\nLoading model: {model_path}")
print(f"Vision projector: {mmproj_path}")

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler

chat_handler = Qwen25VLChatHandler(clip_model_path=mmproj_path, verbose=False)

t0 = time.time()
llm = Llama(
    model_path=model_path,
    n_gpu_layers=99,
    n_ctx=4096,
    verbose=False,
    chat_handler=chat_handler,
    chat_format="qwen2.5-vl",  # may need adjustment
)
print(f"Model loaded in {time.time() - t0:.1f}s")

# 3. Run inference with the image
prompt = "What do you see in this image? Describe everything you can identify."
print(f"\nPrompt: {prompt}")

t1 = time.time()
try:
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
        max_tokens=512,
        temperature=0.1,
    )
    gen_time = time.time() - t1
    text = response["choices"][0]["message"]["content"]
    print(f"\nGeneration took {gen_time:.1f}s")
    print(f"\n=== LLAMA.CPP RESPONSE ===")
    print(text)
    print(f"=== END ===")
    print(f"\nGround truth: top-left=red circle, top-right=blue square, bottom-left=green triangle, bottom-right=yellow star")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
