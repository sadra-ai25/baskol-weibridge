#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ماژول اعتبارسنجی تک دوربین با تشخیص هوشمند خطوط زرد

این ماژول با منطق هوشمند عمل می‌کند:
1. ابتدا خطوط زرد را در تصویر تشخیص می‌دهد
2. اگر خطوط زرد قابل مشاهده باشند → از Scale Area استفاده می‌کند
3. اگر خطوط زرد پوشیده شده باشند (وسیله بزرگ مثل تریلر) → از Prohibited Zone استفاده می‌کند
"""

import cv2
import numpy as np
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from ultralytics import YOLO
from image_utils import resize_with_padding, transform_bbox_from_padded


@dataclass
class Detection:
    """کلاس اطلاعات یک تشخیص"""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)


@dataclass
class SingleCameraValidationResult:
    """نتیجه اعتبارسنجی تک دوربین"""
    is_valid: bool
    message: str
    detections_in_active_area: List[Detection]
    vehicles_count: int
    forbidden_objects_count: int
    violations: List[str]
    used_area: str  # 'scale_area' or 'prohibited_zone'
    yellow_lines_visible: bool
    yellow_line_visibility_percent: float


class SingleCameraValidator:
    """کلاس اعتبارسنجی برای سیستم تک دوربین با تشخیص خطوط زرد"""

    # کلاس‌های مجاز و غیرمجاز
    ALLOWED_VEHICLE_CLASSES = {'truck', 'car', 'boat'}  # 'boat' often misclassified from top-down trucks
    FORBIDDEN_CLASSES = {'bicycle', 'motorcycle'}  # فقط وسایل نقلیه غیرمجاز
    IGNORED_CLASSES = {'person'}  # person = راننده داخل کابین - نادیده گرفته می‌شود

    # def __init__(self, model_path: str = "weights/yolo11l.pt", confidence_threshold: float = 0.25):
    def __init__(self, model_path: str = "weights/best.pt", confidence_threshold: float = 0.25):
        """
        Args:
            model_path: مسیر مدل YOLO
            confidence_threshold: آستانه اطمینان برای تشخیص‌ها
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def detect_yellow_lines(self, image_path: str, scale_polygon: List[List[int]], crop_roi: Dict = None) -> Tuple[bool, float]:
        """
        تشخیص اینکه آیا خطوط زرد در تصویر قابل مشاهده هستند یا خیر

        ✨ مهم: polygon ها در فضای 1920x1080 هستند (بعد از crop و padding)

        Args:
            image_path: مسیر تصویر
            scale_polygon: نقاط چندضلعی ناحیه باسکول (در فضای 1920x1080)
            crop_roi: ناحیه crop (اگر None باشد، تمام تصویر استفاده می‌شود)

        Returns:
            (is_visible, visibility_percent): آیا خطوط قابل مشاهده است و درصد دید
        """
        # خواندن تصویر اصلی
        img = cv2.imread(image_path)
        if img is None:
            return False, 0.0

        # اگر crop_roi داریم، ابتدا crop کنیم
        if crop_roi:
            x1 = crop_roi['x1']
            y1 = crop_roi['y1']
            x2 = crop_roi['x2']
            y2 = crop_roi['y2']
            cropped_image = img[y1:y2, x1:x2].copy()
        else:
            cropped_image = img.copy()

        # ✨ استفاده از padding برای رسیدن به 1920x1080
        padded_image, _ = resize_with_padding(cropped_image, target_size=(1920, 1080))

        # تبدیل به HSV برای تشخیص رنگ زرد
        hsv = cv2.cvtColor(padded_image, cv2.COLOR_BGR2HSV)

        # محدوده رنگ زرد در HSV
        # زرد کمرنگ تا زرد پررنگ
        lower_yellow1 = np.array([20, 50, 50])
        upper_yellow1 = np.array([40, 255, 255])

        # ماسک برای رنگ زرد
        mask_yellow = cv2.inRange(hsv, lower_yellow1, upper_yellow1)

        # ایجاد ماسک برای ناحیه scale_area
        # polygon ها در فضای 1920x1080 هستند، پس مستقیماً استفاده می‌کنیم
        h, w = padded_image.shape[:2]
        scale_mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(scale_polygon, dtype=np.int32)
        cv2.fillPoly(scale_mask, [pts], 255)

        # محاسبه پیکسل‌های زرد داخل ناحیه scale
        yellow_in_scale = cv2.bitwise_and(mask_yellow, scale_mask)

        # محاسبه درصد پوشش خطوط زرد
        scale_area_pixels = np.sum(scale_mask > 0)
        yellow_pixels = np.sum(yellow_in_scale > 0)

        if scale_area_pixels == 0:
            return False, 0.0

        visibility_percent = (yellow_pixels / scale_area_pixels) * 100

        # آستانه: اگر بیش از 5% از ناحیه scale زرد باشد، خطوط قابل مشاهده است
        # این مقدار قابل تنظیم است
        is_visible = bool(visibility_percent > 5.0)

        return is_visible, float(visibility_percent)

    def _calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """محاسبه IoU بین دو bbox"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # محاسبه ناحیه overlap
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def detect_objects(self, image_path: str, crop_roi: Dict = None) -> List[Detection]:
        """
        تشخیص اشیاء در تصویر با استفاده از YOLO

        ✨ مهم: از padding استفاده می‌شود برای حفظ aspect ratio

        Args:
            image_path: مسیر تصویر
            crop_roi: ناحیه crop (اگر None باشد، تمام تصویر استفاده می‌شود)

        Returns:
            لیست تشخیص‌های انجام شده (با مختصات در فضای 1920x1080)
        """
        import cv2

        # خواندن تصویر اصلی
        full_image = cv2.imread(image_path)
        if full_image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        # اگر crop_roi داریم، ابتدا crop کنیم
        if crop_roi:
            x1 = crop_roi['x1']
            y1 = crop_roi['y1']
            x2 = crop_roi['x2']
            y2 = crop_roi['y2']
            cropped_image = full_image[y1:y2, x1:x2].copy()
        else:
            cropped_image = full_image.copy()

        # ✨ استفاده از padding برای حفظ aspect ratio (به جای resize)
        padded_image, padding_metadata = resize_with_padding(cropped_image, target_size=(1920, 1080))

        # ذخیره metadata برای استفاده بعدی
        self._padding_metadata = padding_metadata

        # YOLO را روی تصویر padded اجرا کن (1920x1080)
        results = self.model(padded_image, conf=self.confidence_threshold, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # استخراج اطلاعات
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                class_name = result.names[cls_id]

                # مختصات در فضای 1920x1080 (همان فضایی که polygon ها تعریف شده‌اند)
                bbox = (int(x1), int(y1), int(x2), int(y2))

                detections.append(Detection(
                    class_name=class_name,
                    confidence=conf,
                    bbox=bbox
                ))

        # حذف detections تکراری (NMS برای کلاس‌های مجاز)
        # اگر دو detection از کلاس‌های مجاز overlap زیادی دارند، فقط یکی با conf بالاتر نگه داشته شود
        filtered_detections = []
        detections_sorted = sorted(detections, key=lambda d: d.confidence, reverse=True)

        for det in detections_sorted:
            # بررسی overlap با detections قبلی
            should_keep = True
            for kept_det in filtered_detections:
                # فقط برای vehicles بررسی کن
                if (det.class_name in self.ALLOWED_VEHICLE_CLASSES and
                    kept_det.class_name in self.ALLOWED_VEHICLE_CLASSES):
                    iou = self._calculate_iou(det.bbox, kept_det.bbox)
                    if iou > 0.5:  # overlap بیش از 50%
                        should_keep = False
                        break

            if should_keep:
                filtered_detections.append(det)

        return filtered_detections

    def is_bbox_completely_inside_polygon(self, bbox: Tuple[int, int, int, int],
                                         polygon: List[List[int]]) -> bool:
        """
        بررسی اینکه آیا bbox کاملاً داخل چندضلعی است (بدون همپوشانی با لبه‌ها)

        Args:
            bbox: مستطیل محدودکننده (x1, y1, x2, y2)
            polygon: لیست نقاط چندضلعی

        Returns:
            True اگر bbox کاملاً داخل polygon باشد
        """
        x1, y1, x2, y2 = bbox
        pts = np.array(polygon, dtype=np.int32)

        # بررسی همه گوشه‌های bbox
        corners = [
            (x1, y1),  # بالا چپ
            (x2, y1),  # بالا راست
            (x1, y2),  # پایین چپ
            (x2, y2),  # پایین راست
        ]

        # همه گوشه‌ها باید داخل polygon باشند
        for corner in corners:
            result = cv2.pointPolygonTest(pts, corner, False)
            if result < 0:  # خارج از polygon
                return False

        return True

    def is_bbox_overlapping_polygon(self, bbox: Tuple[int, int, int, int],
                                   polygon: List[List[int]],
                                   threshold: float = 0.05) -> bool:
        """
        بررسی اینکه آیا bbox با چندضلعی همپوشانی دارد
        (حداقل threshold درصد از bbox باید داخل polygon باشد)

        Args:
            bbox: مستطیل محدودکننده (x1, y1, x2, y2)
            polygon: لیست نقاط چندضلعی
            threshold: حداقل درصد همپوشانی (پیش‌فرض: 0.05 = 5%)

        Returns:
            True اگر bbox با polygon همپوشانی داشته باشد
        """
        x1, y1, x2, y2 = bbox
        pts = np.array(polygon, dtype=np.int32)

        # بررسی گوشه‌های bbox
        corners = [
            (x1, y1),  # بالا چپ
            (x2, y1),  # بالا راست
            (x1, y2),  # پایین چپ
            (x2, y2),  # پایین راست
        ]

        # بررسی نقاط میانی
        mid_points = [
            ((x1 + x2) // 2, y1),  # وسط بالا
            ((x1 + x2) // 2, y2),  # وسط پایین
            (x1, (y1 + y2) // 2),  # وسط چپ
            (x2, (y1 + y2) // 2),  # وسط راست
            ((x1 + x2) // 2, (y1 + y2) // 2),  # مرکز
        ]

        all_points = corners + mid_points
        inside_count = sum(1 for p in all_points if cv2.pointPolygonTest(pts, p, False) >= 0)

        overlap_ratio = inside_count / len(all_points)
        return overlap_ratio >= threshold

    def filter_detections_in_polygon(self, detections: List[Detection],
                                    polygon: List[List[int]],
                                    require_complete_inside: bool = False) -> List[Detection]:
        """
        فیلتر کردن تشخیص‌هایی که داخل چندضلعی هستند

        Args:
            detections: لیست تشخیص‌ها
            polygon: چندضلعی
            require_complete_inside: اگر True، فقط اشیاء کاملاً داخل polygon را برمی‌گرداند

        Returns:
            لیست تشخیص‌های داخل چندضلعی
        """
        filtered = []
        for det in detections:
            if require_complete_inside:
                if self.is_bbox_completely_inside_polygon(det.bbox, polygon):
                    filtered.append(det)
            else:
                # ✨ ابتدا overlap استاندارد را بررسی کن
                if self.is_bbox_overlapping_polygon(det.bbox, polygon):
                    filtered.append(det)
                else:
                    # ✨ اگر overlap نداشت، بررسی کن که آیا نزدیک به polygon است
                    # این برای مواردی است که polygon کوچک‌تر از واقعیت است
                    x1, y1, x2, y2 = det.bbox
                    pts = np.array(polygon, dtype=np.int32)

                    # بررسی فاصله مرکز bbox تا polygon
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    dist_center = cv2.pointPolygonTest(pts, (center_x, center_y), measureDist=True)

                    # اگر مرکز نزدیک به polygon است (در محدوده 200 پیکسل)
                    if abs(dist_center) < 200:
                        filtered.append(det)
        return filtered

    def validate(self, image_path: str, roi_config: Dict) -> SingleCameraValidationResult:
        """
        اعتبارسنجی تصویر با منطق هوشمند:
        1. تشخیص خطوط زرد
        2. اگر خطوط قابل مشاهده → استفاده از scale_area
        3. اگر خطوط پوشیده شده → استفاده از prohibited_zone

        Args:
            image_path: مسیر تصویر
            roi_config: تنظیمات ROI شامل دو ناحیه و crop_roi اختیاری

        Returns:
            نتیجه اعتبارسنجی
        """
        # استخراج crop_roi اگر موجود باشد
        crop_roi = roi_config.get('crop_roi', None)

        # تشخیص اشیاء (با استفاده از crop اگر مشخص شده)
        all_detections = self.detect_objects(image_path, crop_roi=crop_roi)

        # استخراج نواحی از config
        # نقاط همیشه نسبت به تصویر اصلی (1920x1080) هستند
        scale_area_points = roi_config['areas']['scale_area']['points']
        prohibited_zone_points = roi_config['areas']['prohibited_zone']['points']

        # تشخیص خطوط زرد
        yellow_lines_visible, visibility_percent = self.detect_yellow_lines(
            image_path, scale_area_points, crop_roi=crop_roi
        )

        # انتخاب ناحیه بر اساس قابلیت دید خطوط زرد
        if yellow_lines_visible:
            # حالت عادی: خطوط زرد قابل مشاهده است
            active_area = 'scale_area'
            active_polygon = scale_area_points
            require_complete = False  # ✨ overlap کافی است (ماشین‌های بزرگ کاملاً داخل نمی‌شوند)
        else:
            # حالت فال‌بک: خطوط زرد پوشیده شده (وسیله بزرگ)
            active_area = 'prohibited_zone'
            active_polygon = prohibited_zone_points
            require_complete = False  # ✨ overlap کافی است

        # فیلتر کردن تشخیص‌های داخل ناحیه فعال
        detections_in_area = self.filter_detections_in_polygon(
            all_detections, active_polygon, require_complete_inside=require_complete
        )

        # جداسازی وسایل نقلیه و اشیاء ممنوع
        vehicles = [
            d for d in detections_in_area
            if d.class_name in self.ALLOWED_VEHICLE_CLASSES
        ]
        forbidden = [
            d for d in detections_in_area
            if d.class_name in self.FORBIDDEN_CLASSES
        ]

        # قوانین اعتبارسنجی
        violations = []
        is_valid = True

        # قانون 0: اگر هیچ آبجکتی نیست → VALID (اسکیپ می‌شود، نه نامعتبر!)
        if len(vehicles) == 0 and len(forbidden) == 0:
            # این حالت معتبر است - ناحیه خالی است
            is_valid = True
            message = "✅ VALID - No objects detected (Empty area - Skipped)"
        else:
            # قانون 1: اگر وسیله‌ای هست، باید دقیقاً 1 عدد باشد
            if len(vehicles) > 1:
                violations.append(f"بیش از یک وسیله نقلیه در ناحیه ({len(vehicles)} وسیله)")
                is_valid = False

            # قانون 2: نباید شیء ممنوع در ناحیه باشد
            if len(forbidden) > 0:
                forbidden_names = [d.class_name for d in forbidden]
                violations.append(f"اشیاء ممنوع در ناحیه: {', '.join(set(forbidden_names))}")
                is_valid = False

            # قانون 3: نباید اشیاء دیگری (غیر از وسیله اصلی و ignored) در ناحیه باشد
            other_objects = [
                d for d in detections_in_area
                if d not in vehicles and d not in forbidden and d.class_name not in self.IGNORED_CLASSES
            ]
            if len(other_objects) > 0:
                other_names = [d.class_name for d in other_objects]
                violations.append(f"اشیاء اضافی در ناحیه: {', '.join(set(other_names))}")
                is_valid = False

            # ✨ قانون 4 (جدید): بررسی overlap با خطوط polygon فعال
            # اگر ماشین با خط های polygon overlap داشته باشد → INVALID
            if len(vehicles) > 0:
                for vehicle in vehicles:
                    # بررسی overlap با polygon
                    bbox = vehicle.bbox
                    x1, y1, x2, y2 = bbox

                    # ساخت contour برای bounding box
                    bbox_contour = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)

                    # بررسی intersection با خط polygon
                    # استفاده از cv2.intersectConvexConvex یا روش دیگر
                    # برای سادگی، بررسی می‌کنیم که آیا گوشه های bbox روی خط polygon هستند

                    # روش ساده: بررسی فاصله نقاط bbox تا خط polygon
                    # اگر هر کدام از گوشه های bbox روی خط polygon باشند → overlap

                    # برای این کار از cv2.pointPolygonTest استفاده می‌کنیم
                    # اگر نقطه دقیقاً روی لبه باشد، مقدار 0 برمی‌گردد
                    # اگر داخل باشد، مثبت است و اگر بیرون باشد، منفی است

                    overlap_detected = False
                    tolerance = 30  # تلرانس 30 پیکسل برای overlap

                    # بررسی گوشه های bounding box
                    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                    active_poly_np = np.array(active_polygon, dtype=np.int32)

                    for corner in corners:
                        dist = cv2.pointPolygonTest(active_poly_np, corner, measureDist=True)
                        # اگر فاصله نزدیک به صفر است (داخل تلرانس)، یعنی روی لبه است
                        if abs(dist) < tolerance:
                            overlap_detected = True
                            break

                    # همچنین بررسی کنیم که آیا لبه های bbox با polygon تقاطع دارند
                    # این کار پیچیده‌تر است، برای سادگی فقط گوشه ها را بررسی می‌کنیم

                    if overlap_detected:
                        area_name = "ناحیه باسکول (خط زرد)" if yellow_lines_visible else "منطقه ممنوع (خط قرمز)"
                        violations.append(f"وسیله با خطوط {area_name} overlap دارد")
                        is_valid = False

            # پیام نهایی برای حالت‌های دارای آبجکت
            if is_valid:
                if yellow_lines_visible:
                    message = "✅ VALID - Vehicle properly positioned in scale area"
                else:
                    message = "✅ VALID - Large vehicle in prohibited zone (yellow lines masked)"
            else:
                area_name = "ناحیه باسکول" if yellow_lines_visible else "منطقه بزرگ"
                message = f"❌ INVALID - {area_name}: " + " | ".join(violations)

        return SingleCameraValidationResult(
            is_valid=is_valid,
            message=message,
            detections_in_active_area=detections_in_area,
            vehicles_count=len(vehicles),
            forbidden_objects_count=len(forbidden),
            violations=violations,
            used_area=active_area,
            yellow_lines_visible=yellow_lines_visible,
            yellow_line_visibility_percent=visibility_percent
        )

    def draw_results(self, image_path: str, roi_config: Dict,
                    validation_result: SingleCameraValidationResult,
                    output_path: str) -> None:
        """
        رسم نتایج اعتبارسنجی روی تصویر

        ✨ مهم: از padding استفاده می‌شود برای حفظ aspect ratio

        Args:
            image_path: مسیر تصویر ورودی
            roi_config: تنظیمات ROI
            validation_result: نتیجه اعتبارسنجی
            output_path: مسیر ذخیره تصویر خروجی
        """
        # خواندن تصویر اصلی
        full_image = cv2.imread(image_path)
        if full_image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        # استخراج crop_roi (اگر وجود داشته باشد)
        crop_roi = roi_config.get('crop_roi')

        # اگر crop_roi داریم، ابتدا crop کنیم
        if crop_roi:
            x1 = crop_roi['x1']
            y1 = crop_roi['y1']
            x2 = crop_roi['x2']
            y2 = crop_roi['y2']
            cropped_image = full_image[y1:y2, x1:x2].copy()
        else:
            cropped_image = full_image.copy()

        # ✨ استفاده از padding برای رسیدن به 1920x1080
        img, _ = resize_with_padding(cropped_image, target_size=(1920, 1080))

        h, w = img.shape[:2]

        # استخراج نقاط polygon (نقاط در فضای 1920x1080 هستند)
        scale_pts = np.array(roi_config['areas']['scale_area']['points'], dtype=np.int32)
        prohibited_pts = np.array(roi_config['areas']['prohibited_zone']['points'], dtype=np.int32)

        # رسم هر دو مستطیل روی تصویر کراپ شده
        # 1. رسم Prohibited Zone (مستطیل بزرگتر) - با رنگ قرمز
        if validation_result.used_area == 'prohibited_zone':
            # فعال - رنگ پررنگ
            overlay = img.copy()
            cv2.fillPoly(overlay, [prohibited_pts], (0, 0, 255))
            cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
            cv2.polylines(img, [prohibited_pts], True, (0, 0, 255), 4)
        else:
            # غیرفعال - رنگ کمرنگ
            cv2.polylines(img, [prohibited_pts], True, (0, 0, 180), 2)

        # 2. رسم Scale Area (مستطیل کوچکتر با خطوط زرد) - با رنگ زرد
        if validation_result.used_area == 'scale_area':
            # فعال - رنگ پررنگ
            overlay = img.copy()
            cv2.fillPoly(overlay, [scale_pts], (0, 255, 255))
            cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
            cv2.polylines(img, [scale_pts], True, (0, 255, 255), 4)
        else:
            # غیرفعال (خطوط پوشیده شده) - رنگ خاکستری
            cv2.polylines(img, [scale_pts], True, (128, 128, 128), 2)

        # رسم تشخیص‌ها
        # چون تصویر img در فضای 1920x1080 است، باید detections را روی این تصویر بگیریم
        # برای این کار باید detections را از validation_result استفاده کنیم
        # یا detections را دوباره روی تصویر فعلی بگیریم

        # استفاده از detection های موجود در validation_result
        for det in validation_result.detections_in_active_area:
            # مختصات detection در فضای 1920x1080 است (crop شده + resize شده)
            x1, y1, x2, y2 = det.bbox

            # اگر crop داریم، باید مختصات را تبدیل کنیم
            # چون detection روی تصویر crop شده + resize شده اتفاق افتاده
            # و ما هم img را crop + resize کرده‌ایم، پس مختصات باید مستقیماً کار کنند
            # ✨ مختصات مستقیماً استفاده می‌شوند ✨

            # تعیین رنگ بر اساس نوع شیء
            if det.class_name in self.ALLOWED_VEHICLE_CLASSES:
                color = (0, 255, 0)  # سبز برای وسایل مجاز
                label_color = (0, 255, 0)
            elif det.class_name in self.FORBIDDEN_CLASSES:
                color = (0, 0, 255)  # قرمز برای اشیاء ممنوع
                label_color = (0, 0, 255)
            elif det.class_name in self.IGNORED_CLASSES:
                color = (255, 165, 0)  # نارنجی برای اشیاء نادیده گرفته شده
                label_color = (255, 165, 0)
            else:
                color = (255, 0, 0)  # آبی برای اشیاء دیگر
                label_color = (255, 0, 0)

            # رسم bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

            # رسم label با پس‌زمینه
            label = f"{det.class_name} {det.confidence:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1-label_h-10), (x1+label_w+10, y1), color, -1)
            cv2.putText(img, label, (x1+5, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Validation status panel (at top)
        status_color = (0, 255, 0) if validation_result.is_valid else (0, 0, 255)
        status_text = "VALID" if validation_result.is_valid else "INVALID"

        # محاسبه ارتفاع پنل بر اساس تعداد نقض‌ها
        base_panel_height = 180
        violation_height = len(validation_result.violations) * 30 if validation_result.violations else 0
        panel_height = base_panel_height + violation_height

        # رسم پنل اطلاعات با پس‌زمینه نیمه‌شفاف
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (min(w-10, 900), panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        cv2.rectangle(img, (10, 10), (min(w-10, 900), panel_height), status_color, 3)

        y_offset = 50
        # نمایش وضعیت اصلی
        cv2.putText(img, f"Validation: {status_text}",
                   (25, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.3, status_color, 3)

        y_offset += 45
        # نمایش ناحیه فعال
        area_text = "Scale Area (Yellow Lines)" if validation_result.used_area == 'scale_area' else "Prohibited Zone (Large Vehicle)"
        area_color = (0, 255, 255) if validation_result.used_area == 'scale_area' else (0, 0, 255)
        cv2.putText(img, f"Active Area: {area_text}",
                   (25, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, area_color, 2)

        y_offset += 35
        # نمایش وضعیت خطوط زرد
        yellow_status = 'Visible' if validation_result.yellow_lines_visible else 'Masked'
        yellow_color = (0, 255, 255) if validation_result.yellow_lines_visible else (128, 128, 128)
        cv2.putText(img, f"Yellow Lines: {yellow_status} ({validation_result.yellow_line_visibility_percent:.1f}%)",
                   (25, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.65, yellow_color, 2)

        y_offset += 35
        # نمایش آمار تشخیص‌ها
        total_detections = validation_result.vehicles_count + validation_result.forbidden_objects_count
        info_text = f"Detections: Vehicles={validation_result.vehicles_count} | Forbidden={validation_result.forbidden_objects_count} | Total={total_detections}"
        cv2.putText(img, info_text,
                   (25, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # نمایش نقض‌ها (با ترجمه به انگلیسی)
        if validation_result.violations:
            y_offset += 40
            cv2.putText(img, "Violations:",
                       (25, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
            y_offset += 30

            for violation in validation_result.violations:
                # ترجمه پیام‌های نقض به انگلیسی
                violation_en = violation
                if "هیچ وسیله نقلیه" in violation:
                    violation_en = "No vehicle detected in active area"
                elif "بیش از یک وسیله" in violation:
                    violation_en = f"Multiple vehicles detected ({validation_result.vehicles_count} vehicles)"
                elif "اشیاء ممنوع" in violation:
                    violation_en = "Forbidden objects detected (bicycle/motorcycle)"
                elif "اشیاء اضافی" in violation:
                    violation_en = "Additional objects detected in area"

                cv2.putText(img, f"  - {violation_en}",
                           (35, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 2)
                y_offset += 30

        # لجند (راهنما) در پایین تصویر
        legend_y = h - 130
        overlay = img.copy()
        cv2.rectangle(overlay, (10, legend_y-10), (min(w-10, 600), h-10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        cv2.rectangle(img, (10, legend_y-10), (min(w-10, 600), h-10), (100, 100, 100), 2)

        # نمایش راهنمای نواحی
        cv2.putText(img, "Legend:", (20, legend_y+10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        legend_y += 25
        if validation_result.used_area == 'scale_area':
            cv2.rectangle(img, (20, legend_y), (45, legend_y+18), (0, 255, 255), -1)
            cv2.putText(img, "Scale Area (Active - Yellow Lines Visible)", (55, legend_y+14),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            legend_y += 25
            cv2.rectangle(img, (20, legend_y), (45, legend_y+18), (0, 0, 180), -1)
            cv2.putText(img, "Prohibited Zone (Inactive)", (55, legend_y+14),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.rectangle(img, (20, legend_y), (45, legend_y+18), (0, 0, 255), -1)
            cv2.putText(img, "Prohibited Zone (Active - Lines Masked)", (55, legend_y+14),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            legend_y += 25
            cv2.rectangle(img, (20, legend_y), (45, legend_y+18), (128, 128, 128), -1)
            cv2.putText(img, "Scale Area (Inactive - Covered)", (55, legend_y+14),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # راهنمای رنگ‌های bounding box
        legend_y += 25
        cv2.rectangle(img, (20, legend_y), (45, legend_y+18), (0, 255, 0), -1)
        cv2.putText(img, "Allowed Vehicle (truck/car/boat)", (55, legend_y+14),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        legend_y += 25
        cv2.rectangle(img, (20, legend_y), (45, legend_y+18), (0, 0, 255), -1)
        cv2.putText(img, "Forbidden (bicycle/motorcycle)", (55, legend_y+14),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Save output image
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Only create if there's a directory component
            os.makedirs(output_dir, exist_ok=True)

        # Write the image
        success = cv2.imwrite(output_path, img)
        if not success:
            raise IOError(f"Failed to save image to {output_path}")
