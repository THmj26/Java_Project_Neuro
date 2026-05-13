#!/usr/bin/env python3
"""
camera.py — Real-time zodiac animal recognition via webcam
Requires: pip install opencv-python
Usage: python camera.py
       SPACE → capture & predict    Q → quit
"""

import cv2
import subprocess
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JAVA_CP    = os.path.join(SCRIPT_DIR, "out", "production", "source")
CAPTURE    = os.path.join(SCRIPT_DIR, "capture.jpg")

LABELS_ZH = {
    "Rat": "鼠", "Ox": "牛", "Tiger": "虎", "Rabbit": "兔",
    "Dragon": "龙", "Snake": "蛇", "Horse": "马", "Goat": "羊",
    "Monkey": "猴", "Rooster": "鸡", "Dog": "狗", "Pig": "猪",
}


def run_predict(image_path: str) -> str:
    try:
        r = subprocess.run(
            ["java", "-cp", JAVA_CP, "predict", image_path],
            capture_output=True, text=True, cwd=SCRIPT_DIR, timeout=30,
        )
        if r.returncode != 0:
            return f"ERROR: {r.stderr.strip()}"
        return r.stdout.strip()
    except subprocess.TimeoutExpired:
        return "ERROR: java timeout"
    except FileNotFoundError:
        return "ERROR: java not found in PATH"


def parse_label(output: str) -> str:
    """Extract animal name from predict.java output, e.g. 'Rat'."""
    m = re.search(r'预测结果:\s*([A-Za-z]+)', output)
    if m:
        en = m.group(1)
        zh = LABELS_ZH.get(en, "")
        return f"{en}  {zh}"
    return "Unknown"


def draw_text(img, text, pos, scale=0.9, color=(0, 60, 255), thickness=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def main():
    if not os.path.isdir(JAVA_CP):
        print(f"[WARNING] Compiled classes not found: {JAVA_CP}")
        print("          Please run Build -> Build Project in IntelliJ first.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check device connection.")
        sys.exit(1)

    print("Webcam ready.")
    print("  SPACE → capture & predict")
    print("  Q     → quit\n")

    frozen   = None   # non-None while frame is frozen
    label    = ""     # latest prediction label shown on screen
    raw_out  = ""     # full predict.java output printed to terminal

    while True:
        if frozen is None:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame from webcam.")
                break
            display = frame.copy()
        else:
            display = frozen.copy()

        hint = "SPACE: capture | Q: quit" if frozen is None else "SPACE: resume | Q: quit"
        draw_text(display, hint, (10, 30), scale=0.65, color=(0, 200, 0))

        if label:
            draw_text(display, label, (10, 75), scale=1.2, color=(0, 50, 255), thickness=3)

        cv2.imshow("Zodiac Recognizer", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord(' '):
            if frozen is None:
                resized = cv2.resize(frame, (64, 64))
                cv2.imwrite(CAPTURE, resized)
                frozen = display.copy()

                print("Recognizing...", end="", flush=True)
                raw_out = run_predict(CAPTURE)
                label   = parse_label(raw_out)
                print(f"\r{'─' * 40}")
                print(raw_out)
                print(f"{'─' * 40}\n")
            else:
                frozen = None
                label  = ""

    cap.release()
    cv2.destroyAllWindows()
    if os.path.exists(CAPTURE):
        os.remove(CAPTURE)


if __name__ == "__main__":
    main()
