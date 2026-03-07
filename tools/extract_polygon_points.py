#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
استخراج نقاط polygon از تصاویر dual_area_roi

این ابزار به شما کمک می‌کند تا نقاط دقیق polygons را از تصاویر
wb1_up_dual_area_roi.jpg و wb2_up_dual_area_roi.jpg استخراج کنید.
"""

import cv2
import json
import sys
import argparse
import numpy as np

# متغیرهای سراسری
scale_area_points = []
prohibited_zone_points = []
img_display = None
img_original = None
current_phase = "scale"  # 'scale' or 'prohibited'


def mouse_callback(event, x, y, flags, param):
    """تابع callback برای کلیک روی تصویر"""
    global scale_area_points, prohibited_zone_points, img_display, img_original, current_phase

    if event == cv2.EVENT_LBUTTONDOWN:
        if current_phase == "scale":
            scale_area_points.append([x, y])
            print(f"✅ Scale Area Point {len(scale_area_points)}: [{x}, {y}]")
            color = (0, 255, 255)  # زرد
        else:
            prohibited_zone_points.append([x, y])
            print(f"✅ Prohibited Zone Point {len(prohibited_zone_points)}: [{x}, {y}]")
            color = (0, 0, 255)  # قرمز

        # رسم مجدد
        draw_points()


def draw_points():
    """رسم همه نقاط روی تصویر"""
    global scale_area_points, prohibited_zone_points, img_display, img_original

    img_display = img_original.copy()

    # رسم prohibited zone (قرمز) - در پس‌زمینه
    if len(prohibited_zone_points) > 0:
        pts = np.array(prohibited_zone_points, dtype=np.int32)
        if len(prohibited_zone_points) > 2:
            cv2.polylines(img_display, [pts], True, (0, 0, 255), 3)
            # Fill با شفافیت
            overlay = img_display.copy()
            cv2.fillPoly(overlay, [pts], (0, 0, 255))
            cv2.addWeighted(overlay, 0.1, img_display, 0.9, 0, img_display)

        for i, pt in enumerate(prohibited_zone_points):
            cv2.circle(img_display, tuple(pt), 8, (0, 0, 255), -1)
            cv2.circle(img_display, tuple(pt), 12, (255, 255, 255), 2)
            cv2.putText(img_display, f"P{i+1}", (pt[0]+15, pt[1]-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # رسم scale area (زرد) - در جلو
    if len(scale_area_points) > 0:
        pts = np.array(scale_area_points, dtype=np.int32)
        if len(scale_area_points) > 2:
            cv2.polylines(img_display, [pts], True, (0, 255, 255), 3)
            # Fill با شفافیت
            overlay = img_display.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 255))
            cv2.addWeighted(overlay, 0.1, img_display, 0.9, 0, img_display)

        for i, pt in enumerate(scale_area_points):
            cv2.circle(img_display, tuple(pt), 8, (0, 255, 255), -1)
            cv2.circle(img_display, tuple(pt), 12, (255, 255, 255), 2)
            cv2.putText(img_display, f"S{i+1}", (pt[0]+15, pt[1]+15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # نمایش راهنما
    phase_text = "SCALE AREA (Yellow)" if current_phase == "scale" else "PROHIBITED ZONE (Red)"
    cv2.rectangle(img_display, (10, 10), (450, 70), (0, 0, 0), -1)
    cv2.putText(img_display, f"Phase: {phase_text}", (20, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img_display, "Press ENTER to switch | U=undo | ESC=cancel", (20, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow('Extract Polygon Points', img_display)


def main():
    global img_original, img_display, current_phase, scale_area_points, prohibited_zone_points

    parser = argparse.ArgumentParser(
        description='استخراج نقاط polygon از تصویر',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
نحوه استفاده:
1. تصویر dual_area_roi را باز می‌کند
2. روی نقاط مستطیل زرد (Scale Area) کلیک کنید - به ترتیب ساعتگرد
3. Enter بزنید
4. روی نقاط مستطیل قرمز (Prohibited Zone) کلیک کنید - به ترتیب ساعتگرد
5. Enter بزنید تا ذخیره شود

مثال:
  python3 tools/extract_polygon_points.py \\
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
    print("استخراج نقاط Polygon از تصویر")
    print("="*80)
    print(f"تصویر: {args.image}")
    print(f"اندازه: {w}x{h}")
    print(f"باسکول: {args.weighbridge}")
    print("="*80)
    print("\n📌 فاز 1: Scale Area (مستطیل زرد)")
    print("   - روی هر نقطه به ترتیب ساعتگرد کلیک کنید")
    print("   - بعد از تمام شدن، ENTER بزنید")
    print("   - برای حذف آخرین نقطه: U\n")

    # نمایش تصویر
    cv2.namedWindow('Extract Polygon Points', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Extract Polygon Points', 1400, 800)
    cv2.setMouseCallback('Extract Polygon Points', mouse_callback)

    draw_points()

    while True:
        key = cv2.waitKey(1) & 0xFF

        # Enter - تغییر فاز یا ذخیره
        if key == 13:  # Enter
            if current_phase == "scale":
                if len(scale_area_points) >= 3:
                    print(f"\n✅ Scale Area: {len(scale_area_points)} نقطه ثبت شد")
                    print(f"   نقاط: {scale_area_points}\n")
                    print("📌 فاز 2: Prohibited Zone (مستطیل قرمز)")
                    print("   - روی هر نقطه به ترتیب ساعتگرد کلیک کنید")
                    print("   - بعد از تمام شدن، ENTER بزنید\n")
                    current_phase = "prohibited"
                    draw_points()
                else:
                    print(f"⚠️  حداقل 3 نقطه نیاز است (فعلاً: {len(scale_area_points)})")

            elif current_phase == "prohibited":
                if len(prohibited_zone_points) >= 3:
                    print(f"\n✅ Prohibited Zone: {len(prohibited_zone_points)} نقطه ثبت شد")
                    print(f"   نقاط: {prohibited_zone_points}")
                    break
                else:
                    print(f"⚠️  حداقل 3 نقطه نیاز است (فعلاً: {len(prohibited_zone_points)})")

        # Escape - لغو
        elif key == 27:  # ESC
            print("\n❌ لغو شد")
            cv2.destroyAllWindows()
            return 1

        # u - Undo
        elif key == ord('u') or key == ord('U'):
            if current_phase == "scale" and scale_area_points:
                removed = scale_area_points.pop()
                print(f"↩️  حذف شد: Scale Area Point {len(scale_area_points)+1}")
                draw_points()
            elif current_phase == "prohibited" and prohibited_zone_points:
                removed = prohibited_zone_points.pop()
                print(f"↩️  حذف شد: Prohibited Zone Point {len(prohibited_zone_points)+1}")
                draw_points()

    cv2.destroyAllWindows()

    # ذخیره در config
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
    print(f"\n📋 Scale Area Points:")
    for i, pt in enumerate(scale_area_points):
        print(f"      {i+1}. {pt}")
    print(f"\n📋 Prohibited Zone Points:")
    for i, pt in enumerate(prohibited_zone_points):
        print(f"      {i+1}. {pt}")
    print("\n✅ تمام!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
