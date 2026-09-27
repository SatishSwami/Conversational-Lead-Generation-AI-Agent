#!/usr/bin/env python3
"""Create a 4-panel comparison: Original, Log transform, Gamma (0.5), High-pass filter."""
import sys
import os
import cv2
import numpy as np


def log_transform(img):
    img_f = img.astype(np.float32)
    c = 255.0 / np.log(1.0 + img_f.max())
    log_img = c * np.log(1.0 + img_f)
    return np.clip(log_img, 0, 255).astype(np.uint8)


def gamma_transform(img, gamma=0.5):
    inv = 1.0 / float(gamma)
    table = np.array([((i / 255.0) ** inv) * 255 for i in np.arange(256)]).astype('uint8')
    return cv2.LUT(img, table)


def high_pass(img):
    # Use unsharp mask style high-pass: original - blurred, add 128 to center
    blurred = cv2.GaussianBlur(img, (21, 21), 0)
    hp = cv2.subtract(img, blurred)
    # scale and shift for visibility
    hp = cv2.normalize(hp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return hp.astype(np.uint8)


def to_three_channel(img):
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def make_panel(img):
    # pad to square-ish area if needed (keep proportions)
    return img


def main():
    if len(sys.argv) < 2:
        print('Usage: enhancement_compare.py input.jpg [output.jpg]')
        sys.exit(1)
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(inp), 'outputs', 'enhancement_comparison.jpg')
    os.makedirs(os.path.dirname(out), exist_ok=True)

    img = cv2.imread(inp)
    if img is None:
        print('Failed to load', inp)
        sys.exit(2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    log_img = log_transform(gray)
    gamma_img = gamma_transform(gray, gamma=0.5)
    hp_img = high_pass(gray)

    # Convert to BGR for consistent concatenation and add labels
    orig_b = to_three_channel(gray)
    log_b = to_three_channel(log_img)
    gamma_b = to_three_channel(gamma_img)
    hp_b = to_three_channel(hp_img)

    # Resize all panes to same size (use original scaled to 600x600 or smaller)
    h, w = gray.shape[:2]
    target_w = min(600, w)
    target_h = int(h * (target_w / w))
    panes = [orig_b, log_b, gamma_b, hp_b]
    panes = [cv2.resize(p, (target_w, target_h)) for p in panes]

    # Create labels on top of panes
    titles = ['Original', 'Log Transform', 'Gamma (0.5)', 'High Pass Filter']
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, p in enumerate(panes):
        cv2.putText(p, titles[i], (10, 30), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Concatenate horizontally and save
    composite = np.hstack(panes)
    cv2.imwrite(out, composite)
    print('Saved', out)


if __name__ == '__main__':
    main()
