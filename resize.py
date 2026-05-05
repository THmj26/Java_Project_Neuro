# resize.py
# 功能：把 raw/ 下的12张原图做数据增强，每张生成多个变体，全部缩放到 64x64 输出到 images/
# 用法：在项目根目录跑 python resize.py
# 每张原图生成 10 个变体，共 12 × 10 = 120 张训练图片

from PIL import Image, ImageEnhance
import os

INPUT_DIR  = "raw"
OUTPUT_DIR = "images"
SIZE = (64, 64)

# 原图文件名 → (输出前缀编号, 类别名)
# 编号从01开始，与 DataLoader 的 -1 偏移对应，最终标签为 0~11
NAME_MAP = {
    "rat.jpg":     ("01", "rat"),
    "ox.jpg":      ("02", "ox"),
    "tiger.jpg":   ("03", "tiger"),
    "rabbit.jpg":  ("04", "rabbit"),
    "dragon.jpg":  ("05", "dragon"),
    "snake.jpg":   ("06", "snake"),
    "horse.jpg":   ("07", "horse"),
    "goat.jpg":    ("08", "goat"),
    "monkey.jpg":  ("09", "monkey"),
    "rooster.jpg": ("10", "rooster"),
    "dog.jpg":     ("11", "dog"),
    "pig.jpg":     ("12", "pig"),
}

def save(img, prefix, name, tag):
    out = img.resize(SIZE, Image.LANCZOS)
    path = os.path.join(OUTPUT_DIR, f"{prefix}_{name}_{tag}.jpg")
    out.save(path, "JPEG", quality=95)
    print(f"  -> {os.path.basename(path)}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

for src, (prefix, name) in NAME_MAP.items():
    src_path = os.path.join(INPUT_DIR, src)
    if not os.path.exists(src_path):
        print(f"[SKIP] 找不到: {src_path}")
        continue

    print(f"处理: {src}")
    img = Image.open(src_path).convert("RGB")
    w, h = img.size

    # 1. 原图
    save(img, prefix, name, "orig")

    # 2. 水平翻转
    save(img.transpose(Image.FLIP_LEFT_RIGHT), prefix, name, "flipH")

    # 3. 旋转 +15°
    save(img.rotate(15, expand=False), prefix, name, "rot15")

    # 4. 旋转 -15°
    save(img.rotate(-15, expand=False), prefix, name, "rot-15")

    # 5. 旋转 +30°
    save(img.rotate(30, expand=False), prefix, name, "rot30")

    # 6. 旋转 -30°
    save(img.rotate(-30, expand=False), prefix, name, "rot-30")

    # 7. 亮度增强（+50%）
    save(ImageEnhance.Brightness(img).enhance(1.5), prefix, name, "bright")

    # 8. 亮度降低（-30%）
    save(ImageEnhance.Brightness(img).enhance(0.7), prefix, name, "dark")

    # 9. 中心裁剪（取中间 80%，再缩放回 64x64）
    margin_w, margin_h = int(w * 0.1), int(h * 0.1)
    save(img.crop((margin_w, margin_h, w - margin_w, h - margin_h)), prefix, name, "crop")

    # 10. 对比度增强（+40%）
    save(ImageEnhance.Contrast(img).enhance(1.4), prefix, name, "contrast")

print(f"\n完成，共生成 {len(NAME_MAP) * 10} 张图片")
