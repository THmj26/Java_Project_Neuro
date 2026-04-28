# resize.py
# 功能：把 raw/ 下的12张原图缩放到 64x64，按编号命名输出到 images/
# 用法：在项目根目录跑 python resize.py

from PIL import Image
import os

INPUT_DIR  = "raw"
OUTPUT_DIR = "images"
SIZE = (64, 64)

# 类别顺序（跟 train.java 里的标签对应）
NAME_MAP = {
    "rat.jpg":     "01_rat.jpg",
    "ox.jpg":      "02_ox.jpg",
    "tiger.jpg":   "03_tiger.jpg",
    "rabbit.jpg":  "04_rabbit.jpg",
    "dragon.jpg":  "05_dragon.jpg",
    "snake.jpg":   "06_snake.jpg",
    "horse.jpg":   "07_horse.jpg",
    "goat.jpg":    "08_goat.jpg",
    "monkey.jpg":  "09_monkey.jpg",
    "rooster.jpg": "10_rooster.jpg",
    "dog.jpg":     "11_dog.jpg",
    "pig.jpg":     "12_pig.jpg",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
for src, dst in NAME_MAP.items():
    src_path = os.path.join(INPUT_DIR, src)
    if not os.path.exists(src_path):
        print(f"[SKIP] 找不到: {src_path}")
        continue
    img = Image.open(src_path).convert("RGB")
    img = img.resize(SIZE, Image.LANCZOS)
    img.save(os.path.join(OUTPUT_DIR, dst), "JPEG", quality=95)
    print(f"OK: {src} -> {dst}")

print("完成")
