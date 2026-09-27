#!/usr/bin/env python3
"""Simple CLI Image Processor using OpenCV.

Usage examples:
    python tools/image_processor.py "C:/Users/Satish/OneDrive/Desktop/SAtish.jpeg" --resize 800x600 --format PNG --output-dir outputs
    python tools/image_processor.py img1.jpg img2.jpg --blur 5 --brightness 1.2
    python tools/image_processor.py photo.jpg --face-blur --haar path/to/haarcascade_frontalface_default.xml

This script accepts one or more input images and applies the chosen operations.
"""
import argparse
import os
import cv2
import numpy as np


def adjust_brightness_contrast(img, brightness=1.0, contrast=0.0):
    img = img.astype(np.float32) * float(brightness) + float(contrast)
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def blur_faces(img, face_cascade_path, ksize=23):
    if not face_cascade_path or not os.path.exists(face_cascade_path):
        print("haar cascade not found; skipping face blur")
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        roi = img[y : y + h, x : x + w]
        k = ksize if ksize % 2 == 1 else ksize + 1
        roi = cv2.GaussianBlur(roi, (k, k), 0)
        img[y : y + h, x : x + w] = roi
    return img


def process_image(path, args):
    img = cv2.imread(path)
    if img is None:
        print(f"Failed to load {path}")
        return

    if args.resize:
        try:
            w, h = map(int, args.resize.split('x'))
            img = cv2.resize(img, (w, h))
        except Exception as e:
            print('Invalid --resize format, expected WIDTHxHEIGHT', e)

    if args.crop:
        try:
            x, y, w, h = map(int, args.crop.split(','))
            img = img[y : y + h, x : x + w]
        except Exception as e:
            print('Invalid --crop format, expected x,y,w,h', e)

    if args.rotate:
        (h_img, w_img) = img.shape[:2]
        M = cv2.getRotationMatrix2D((w_img / 2, h_img / 2), float(args.rotate), 1.0)
        img = cv2.warpAffine(img, M, (w_img, h_img))

    if args.brightness is not None or args.contrast is not None:
        b = args.brightness if args.brightness is not None else 1.0
        c = args.contrast if args.contrast is not None else 0.0
        img = adjust_brightness_contrast(img, b, c)

    if args.blur:
        k = int(args.blur)
        if k % 2 == 0:
            k += 1
        img = cv2.GaussianBlur(img, (k, k), 0)

    if args.face_blur:
        img = blur_faces(img, args.haar, ksize=args.face_blur_ksize)

    out_dir = args.output_dir or os.path.dirname(path) or '.'
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    out_ext = '.' + (args.format.lower() if args.format else os.path.splitext(path)[1].lstrip('.'))
    out_name = f"{base}_processed{out_ext}"
    out_path = os.path.join(out_dir, out_name)

    params = []
    if out_ext.lower() in ('.jpg', '.jpeg') and args.quality:
        params = [cv2.IMWRITE_JPEG_QUALITY, int(args.quality)]

    ok = cv2.imwrite(out_path, img, params)
    if ok:
        print(f"Saved {out_path}")
    else:
        print(f"Failed to save {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Apply common image operations to one or more images')
    parser.add_argument('inputs', nargs='+', help='Input image path(s)')
    parser.add_argument('--output-dir', help='Directory to save processed images')
    parser.add_argument('--resize', help='WIDTHxHEIGHT (e.g. 800x600)')
    parser.add_argument('--crop', help='x,y,w,h')
    parser.add_argument('--rotate', type=float, help='degrees')
    parser.add_argument('--format', help='PNG or JPEG (file extension)')
    parser.add_argument('--brightness', type=float, help='Multiplier (1.0 = no change)')
    parser.add_argument('--contrast', type=float, help='Additive contrast (0 = no change)')
    parser.add_argument('--blur', help='Gaussian blur kernel size (odd integer)')
    parser.add_argument('--face-blur', action='store_true', help='Detect faces and blur them (requires --haar)')
    parser.add_argument('--haar', help='Path to Haar cascade XML (e.g., haarcascade_frontalface_default.xml)')
    parser.add_argument('--face-blur-ksize', type=int, default=23)
    parser.add_argument('--quality', type=int, default=95, help='JPEG quality')
    args = parser.parse_args()

    for p in args.inputs:
        process_image(p, args)


if __name__ == '__main__':
    main()
