#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crop ROI CLI - نسخه خط فرمان برای انتخاب ناحیه Crop

استفاده:
  python3 tools/crop_roi_cli.py --weighbridge 1 --x1 100 --y1 150 --x2 1800 --y2 900
"""

import cv2
import json
import argparse
from pathlib import Path


def create_crop_config(weighbridge_id, x1, y1, x2, y2, base_image_path):
    """ایجاد config با crop ROI"""

    # بارگذاری تصویر برای بررسی اندازه
    img = cv2.imread(base_image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {base_image_path}")

    h, w = img.shape[:2]

    # بررسی محدوده
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        raise ValueError(f"Crop ROI خارج از محدوده تصویر است! Image size: {w}x{h}")

    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"مختصات نامعتبر: x1 < x2 و y1 < y2 باید باشد")

    crop_width = x2 - x1
    crop_height = y2 - y1

    print(f"\n📐 اندازه‌ها:")
    print(f"   تصویر اصلی: {w}x{h}")
    print(f"   Crop ROI: [{x1}, {y1}, {x2}, {y2}]")
    print(f"   ابعاد Crop: {crop_width}x{crop_height}")
    print(f"   کاهش: {100 - (crop_width*crop_height)/(w*h)*100:.1f}%")

    # ایجاد config
    config = {
        'weighbridge_id': weighbridge_id,
        'weighbridge_name': f'باسکول {weighbridge_id}',
        'camera': 'UP',
        'base_image': base_image_path,
        'original_image_size': [w, h],
        'crop_roi': {
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'width': crop_width,
            'height': crop_height,
            'description': 'Region to crop before processing'
        },
        'areas': {
            'scale_area': {
                'name': 'Scale Area',
                'name_fa': 'ناحیه باسکول',
                'description': 'Yellow boundary area (coordinates relative to cropped image)',
                'num_points': 4,
                'points': [
                    [10, 10],
                    [crop_width - 10, 10],
                    [crop_width - 10, crop_height - 10],
                    [10, crop_height - 10]
                ]
            },
            'prohibited_zone': {
                'name': 'Prohibited Zone',
                'name_fa': 'منطقه ممنوع',
                'description': 'Larger boundary (coordinates relative to cropped image)',
                'num_points': 4,
                'points': [
                    [5, 5],
                    [crop_width - 5, 5],
                    [crop_width - 5, crop_height - 5],
                    [5, crop_height - 5]
                ]
            }
        }
    }

    return config, img


def save_crop_preview(img, x1, y1, x2, y2, output_path):
    """ذخیره پیش‌نمایش ناحیه کراپ"""
    # رسم مستطیل
    img_preview = img.copy()
    cv2.rectangle(img_preview, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # نمایش ابعاد
    cv2.putText(img_preview, f"Crop: {x2-x1}x{y2-y1}", (x1, y1-10),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imwrite(output_path, img_preview)
    print(f"   پیش‌نمایش: {output_path}")

    # ذخیره تصویر کراپ شده
    cropped = img[y1:y2, x1:x2]
    cropped_path = output_path.replace('.jpg', '_cropped.jpg')
    cv2.imwrite(cropped_path, cropped)
    print(f"   کراپ شده: {cropped_path}")


def main():
    parser = argparse.ArgumentParser(description='Crop ROI CLI - انتخاب ناحیه کراپ')
    parser.add_argument('--weighbridge', '-w', type=int, required=True, choices=[1, 2],
                       help='شماره باسکول (1 یا 2)')
    parser.add_argument('--x1', type=int, required=True, help='مختصات X بالا-چپ')
    parser.add_argument('--y1', type=int, required=True, help='مختصات Y بالا-چپ')
    parser.add_argument('--x2', type=int, required=True, help='مختصات X پایین-راست')
    parser.add_argument('--y2', type=int, required=True, help='مختصات Y پایین-راست')
    parser.add_argument('--preview', action='store_true', help='ذخیره پیش‌نمایش')

    args = parser.parse_args()

    # تعیین مسیر تصویر پایه
    base_images = {
        1: 'baskol1-images/bask1_up_20251202_110425.jpg',
        2: 'baskol2-images/bask2_up_20251202_110500.jpg'
    }

    image_path = base_images[args.weighbridge]

    if not Path(image_path).exists():
        print(f"❌ خطا: تصویر پایه یافت نشد: {image_path}")
        return 1

    try:
        print("="*70)
        print(f"Crop ROI Configuration - Weighbridge {args.weighbridge}")
        print("="*70)

        # ایجاد config
        config, img = create_crop_config(
            args.weighbridge,
            args.x1, args.y1, args.x2, args.y2,
            image_path
        )

        # ذخیره config
        output_path = f'configs/wb{args.weighbridge}_single_cam_roi.json'
        Path('configs').mkdir(exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Config ذخیره شد: {output_path}")

        # ذخیره پیش‌نمایش
        if args.preview:
            preview_path = f'crop_preview_wb{args.weighbridge}.jpg'
            save_crop_preview(img, args.x1, args.y1, args.x2, args.y2, preview_path)

        print(f"\n⚠️  توجه:")
        print(f"   - Polygons نسبت به تصویر کراپ شده هستند")
        print(f"   - برای تنظیم دقیق، از dual_area_roi_picker استفاده کنید")

        print(f"\n🎉 موفقیت! حالا می‌توانید سیستم را تست کنید:")
        print(f"   python3 main_single_camera.py -w {args.weighbridge} -i test_images/")

        return 0

    except Exception as e:
        print(f"\n❌ خطا: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
