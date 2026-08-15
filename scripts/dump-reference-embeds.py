#!/usr/bin/env python3
"""Dump vision embeddings from llama.cpp reference for comparison."""
from PIL import Image, ImageDraw
import sys, os, time, math, base64, io, json
import numpy as np

# Generate the same 512x512 shape image
W, H = 512, 512
img = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(img)
cx, cy, r = 128, 128, 80
draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill="#ff0000")
draw.rectangle([W - cx - r, cy - r, W - cx + r, cy + r], fill="#0000ff")
draw.polygon([(cx - r, H - cy + r), (cx + r, H - cy + r), (cx, H - cy - r)], fill="#00aa00")
sx, sy = W - cx, H - cy
star_points = []
for i in range(10):
    angle = (i * math.pi) / 5 - math.pi / 2
    radius = r if i % 2 == 0 else r * 0.4
    star_points.append((sx + math.cos(angle) * radius, sy + math.sin(angle) * radius))
draw.polygon(star_points, fill="#ffcc00")

img_path = "/tmp/vlm_shapes_test.png"
img.save(img_path)

# Encode as base64
buf = io.BytesIO()
img.save(buf, format="PNG")
img_b64 = base64.b64encode(buf.getvalue()).decode()
data_uri = f"data:image/png;base64,{img_b64}"

model_path = os.path.expanduser("~/Downloads/Bonsai-27B-Q1_0.gguf")
mmproj_path = os.path.expanduser("~/Downloads/Bonsai-27B-mmproj-Q8_0.gguf")

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler

chat_handler = Qwen25VLChatHandler(clip_model_path=mmproj_path, verbose=True)

llm = Llama(
    model_path=model_path,
    n_gpu_layers=99,
    n_ctx=4096,
    verbose=False,
    chat_handler=chat_handler,
    chat_format="qwen2.5-vl",
)

# Access the CLIP model to get embeddings directly
clip = chat_handler.clip
print(f"CLIP model loaded: {clip}")

# Preprocess the image
from llama_cpp.llama_cpp import clip_image_f32, clip_image_u8, clip_image_preprocess

# Create clip_image_u8 from PIL image
clip_img = clip_image_u8()
clip_img.nx = W
clip_img.ny = H
# PIL image to RGB bytes
rgb_img = img.convert("RGB")
pixels = list(rgb_img.getdata())
flat = bytearray()
for r, g, b in pixels:
    flat.extend([r, g, b])
clip_img.buf = bytes(flat)

# Preprocess
clip_f32 = clip_image_f32()
clip_image_preprocess(clip, clip_img, clip_f32, pad_to_square=False)
print(f"Preprocessed: nx={clip_f32.nx}, ny={clip_f32.ny}")

# Encode to get embeddings
from llama_cpp.llama_cpp import clip_image_batch_encode
import ctypes

# Allocate output buffer: [5120, 256, 1] = 5120 * 256 floats
n_patches = 256
hidden_size = 5120
embeds = np.zeros((n_patches, hidden_size), dtype=np.float32)

# Use the clip_image_batch_encode function
result = clip_image_batch_encode(clip, 1, [clip_f32], embeds.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
print(f"Encode result: {result}")
print(f"Embeddings shape: {embeds.shape}")
print(f"Embeddings stats: min={embeds.min():.6f}, max={embeds.max():.6f}, mean={embeds.mean():.6f}")

# Save first few embeddings for comparison
out_path = "/tmp/llama_cpp_vision_embeds.json"
data = {
    "shape": list(embeds.shape),
    "first_5_rows_first_10_vals": embeds[:5, :10].tolist(),
    "all_first_10_vals": embeds[:10, :10].tolist(),
    "stats": {
        "min": float(embeds.min()),
        "max": float(embeds.max()),
        "mean": float(embeds.mean()),
        "std": float(embeds.std()),
    },
    # Save full embeddings as binary for direct comparison
    "embeds_b64": base64.b64encode(embeds.tobytes()).decode(),
}
with open(out_path, "w") as f:
    json.dump(data, f)
print(f"\nEmbeddings saved to {out_path}")
print(f"First 5 rows, first 10 values:")
for i in range(5):
    print(f"  row[{i}]: {embeds[i, :10].tolist()}")
