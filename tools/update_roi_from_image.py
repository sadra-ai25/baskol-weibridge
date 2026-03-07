#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
به‌روزرسانی ROI از تصاویر dual_area_roi
این اسکریپت کمک می‌کند تا نقاط مستطیل‌های scale_area و prohibited_zone را
از تصاویر wb1_up_dual_area_roi.jpg و wb2_up_dual_area_roi.jpg استخراج کنید.
"""

import cv2
import json
import sys
import argparse
from pathlib import Path

# متغیرهای سراسری
scale_area_points = []
prohibited_zone_points = []
current_mode = "scale"  # 'scale' or 'prohibited'
img_display = None
img_original = None


def mouse_callback(event, x, y, flags, param):
    """تابع callback برای کلیک روی تصویر"""
    global scale_area_points, prohibited_zone_points, current_mode, img_display, img_original

    if event == cv2.EVENT_LBUTTONDOWN:
        if current_mode == "scale":
            scale_area_points.append([x, y])
            print(f"Scale Area Point {len(scale_area_points)}: ({x}, {y})")
        else:
            prohibited_zone_points.append([x, y])
            print(f"Prohibited Zone Point {len(prohibited_zone_points)}: ({x, y})")

        # رسم نقطه
        img_display = img_original.copy()

        # رسم نقاط scale area
        for i, pt in enumerate(scale_area_points):
            cv2.circle(img_display, tuple(pt), 8, (0, 255, 255), -1)
            cv2.putText(img_display, f"S{i+1}", (pt[0]+10, pt[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # اتصال خطوط scale area
        if len(scale_area_points) > 1:
            for i in range(len(scale_area_points)):
                cv2.line(img_display,
                        tuple(scale_area_points[i]),
                        tuple(scale_area_points[(i+1) % len(scale_area_points)]),
                        (0, 255, 255), 2)

        # رسم نقاط prohibited zone
        for i, pt in enumerate(prohibited_zone_points):
            cv2.circle(img_display, tuple(pt), 8, (0, 0, 255), -1)
            cv2.putText(img_display, f"P{i+1}", (pt[0]+10, pt[1]+10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # اتصال خطوط prohibited zone
        if len(prohibited_zone_points) > 1:
            for i in range(len(prohibited_zone_points)):
                cv2.line(img_display,
                        tuple(prohibited_zone_points[i]),
                        tuple(prohibited_zone_points[(i+1) % len(prohibited_zone_points)]),
                        (0, 0, 255), 2)

        cv2.imshow('ROI Selector', img_display)


def main():
    global img_display, img_original, current_mode, scale_area_points, prohibited_zone_points

    parser = argparse.ArgumentParser(
        description='استخراج نقاط ROI از تصویر dual_area_roi',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
نحوه استفاده:
1. تصویر wb{N}_up_dual_area_roi.jpg را انتخاب کنید
2. ابتدا 4 نقطه برای Scale Area (مستطیل زرد) کلیک کنید
3. Enter بزنید
4. سپس 4 نقطه برای Prohibited Zone (مستطیل قرمز) کلیک کنید
5. Enter بزنید تا ذخیره شود

مثال:
  python3 tools/update_roi_from_image.py \\
    -i outputs/roi_visualizations/wb1_up_dual_area_roi.jpg \\
    -w 1
        """
    )

    parser.add_argument('--image', '-i', type=str, required=True,
                       help='مسیر تصویر dual_area_roi')
    parser.add_argument('--weighbridge', '-w', type=int, required=True,
                       choices=[1, 2], help='شماره باسکول')

    args = parser.parse_args()

    # بارگذاری تصویر
    img_original = cv2.imread(args.image)
    if img_original is None:
        print(f"❌ خطا: نمی‌توان تصویر را بارگذاری کرد: {args.image}")
        return 1

    img_display = img_original.copy()
    h, w = img_original.shape[:2]

    print("="*80)
    print("به‌روزرسانی ROI از تصویر")
    print("="*80)
    print(f"تصویر: {args.image}")
    print(f"اندازه: {w}x{h}")
    print(f"باسکول: {args.weighbridge}")
    print("="*80)
    print("\n📌 مرحله 1: Scale Area (مستطیل زرد)")
    print("   - 4 نقطه را کلیک کنید (گوشه‌های مستطیل)")
    print("   - بعد از انتخاب 4 نقطه، Enter بزنید\n")

    # نمایش تصویر
    cv2.namedWindow('ROI Selector', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('ROI Selector', mouse_callback)
    cv2.imshow('ROI Selector', img_display)

    current_mode = "scale"

    while True:
        key = cv2.waitKey(1) & 0xFF

        # Enter - تغییر مد یا ذخیره
        if key == 13:  # Enter
            if current_mode == "scale":
                if len(scale_area_points) >= 4:
                    print(f"\n✅ Scale Area: {len(scale_area_points)} نقطه ثبت شد")
                    print("\n📌 مرحله 2: Prohibited Zone (مستطیل قرمز)")
                    print("   - 4 نقطه را کلیک کنید (گوشه‌های مستطیل بزرگتر)")
                    print("   - بعد از انتخاب 4 نقطه، Enter بزنید\n")
                    current_mode = "prohibited"
                else:
                    print(f"⚠️  لطفاً حداقل 4 نقطه برای Scale Area انتخاب کنید (فعلاً: {len(scale_area_points)})")

            elif current_mode == "prohibited":
                if len(prohibited_zone_points) >= 4:
                    print(f"\n✅ Prohibited Zone: {len(prohibited_zone_points)} نقطه ثبت شد")
                    break
                else:
                    print(f"⚠️  لطفاً حداقل 4 نقطه برای Prohibited Zone انتخاب کنید (فعلاً: {len(prohibited_zone_points)})")

        # Escape - لغو
        elif key == 27:  # ESC
            print("\n❌ لغو شد")
            cv2.destroyAllWindows()
            return 1

        # u - Undo (حذف آخرین نقطه)
        elif key == ord('u'):
            if current_mode == "scale" and scale_area_points:
                removed = scale_area_points.pop()
                print(f"حذف شد: Scale Area Point {len(scale_area_points)+1}")
            elif current_mode == "prohibited" and prohibited_zone_points:
                removed = prohibited_zone_points.pop()
                print(f"حذف شد: Prohibited Zone Point {len(prohibited_zone_points)+1}")

            # رسم مجدد
            img_display = img_original.copy()
            for i, pt in enumerate(scale_area_points):
                cv2.circle(img_display, tuple(pt), 8, (0, 255, 255), -1)
            for i, pt in enumerate(prohibited_zone_points):
                cv2.circle(img_display, tuple(pt), 8, (0, 0, 255), -1)
            cv2.imshow('ROI Selector', img_display)

    cv2.destroyAllWindows()

    # ذخیره نقاط در config
    config_path = f"configs/wb{args.weighbridge}_single_cam_roi.json"

    print("\n" + "="*80)
    print("ذخیره نقاط در config...")
    print("="*80)

    # بارگذاری config فعلی
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"⚠️  فایل config یافت نشد. ایجاد config جدید...")
        config = {
            "weighbridge_id": args.weighbridge,
            "weighbridge_name": f"باسکول {args.weighbridge}",
            "camera": "UP",
            "original_image_size": [w, h],
            "crop_roi": None,
            "areas": {}
        }

    # به‌روزرسانی نقاط
    config["areas"]["scale_area"] = {
        "name": "Scale Area",
        "name_fa": "ناحیه باسکول",
        "description": "Yellow boundary area (coordinates relative to ORIGINAL image)",
        "num_points": len(scale_area_points),
        "points": scale_area_points
    }

    config["areas"]["prohibited_zone"] = {
        "name": "Prohibited Zone",
        "name_fa": "منطقه ممنوع",
        "description": "Larger boundary (coordinates relative to ORIGINAL image)",
        "num_points": len(prohibited_zone_points),
        "points": prohibited_zone_points
    }

    # ذخیره config
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"✅ Config ذخیره شد: {config_path}")
    print(f"\n📊 خلاصه:")
    print(f"   - Scale Area: {len(scale_area_points)} نقطه")
    print(f"   - Prohibited Zone: {len(prohibited_zone_points)} نقطه")
    print("\n✅ تمام!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
