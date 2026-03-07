#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Weighbridge Validator - Direct 1920x1080 Processing
No cropping, no padding - straightforward implementation
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class Detection:
    """Detection result"""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)


@dataclass
class ValidationResult:
    """Validation result"""
    is_valid: bool
    used_area: str  # 'scale_area' or 'prohibited_zone'
    yellow_lines_visible: bool
    yellow_line_visibility_percent: float
    vehicles_count: int
    detections: List[Detection]
    violations: List[str]
    message: str


class SimpleValidator:
    """Simple validator for weighbridge monitoring"""

    # Allowed vehicle classes
    ALLOWED_VEHICLE_CLASSES = {'car', 'truck', 'bus'}

    # Forbidden classes (bicycles, motorcycles)
    FORBIDDEN_CLASSES = {'bicycle', 'motorbike', 'motorcycle'}

    def __init__(self, model_path: str = 'weights/best.pt', confidence_threshold: float = 0.25):
        """
        Initialize validator

        Args:
            model_path: Path to YOLO model
            confidence_threshold: Detection confidence threshold
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def detect_objects(self, image_path: str) -> List[Detection]:
        """
        Detect objects in image

        Args:
            image_path: Path to image (must be 1920x1080)

        Returns:
            List of detections
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        # Ensure 1920x1080
        h, w = img.shape[:2]
        if w != 1920 or h != 1080:
            img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_LINEAR)

        # Run YOLO
        results = self.model(img, conf=self.confidence_threshold, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                class_name = result.names[cls_id]

                bbox = (int(x1), int(y1), int(x2), int(y2))

                detections.append(Detection(
                    class_name=class_name,
                    confidence=conf,
                    bbox=bbox
                ))

        return detections

    def detect_yellow_lines(self, image_path: str, scale_polygon: List[List[int]]) -> Tuple[bool, float]:
        """
        Detect yellow lines visibility

        Args:
            image_path: Path to image
            scale_polygon: Scale area polygon points

        Returns:
            (is_visible, visibility_percent)
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return False, 0.0

        # Ensure 1920x1080
        h, w = img.shape[:2]
        if w != 1920 or h != 1080:
            img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_LINEAR)

        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Yellow color range
        lower_yellow = np.array([20, 50, 50])
        upper_yellow = np.array([40, 255, 255])

        # Yellow mask
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Create scale area mask
        scale_mask = np.zeros((1080, 1920), dtype=np.uint8)
        pts = np.array(scale_polygon, dtype=np.int32)
        cv2.fillPoly(scale_mask, [pts], 255)

        # Calculate yellow pixels in scale area
        yellow_in_scale = cv2.bitwise_and(mask_yellow, scale_mask)

        scale_area_pixels = np.sum(scale_mask > 0)
        yellow_pixels = np.sum(yellow_in_scale > 0)

        if scale_area_pixels == 0:
            return False, 0.0

        visibility_percent = (yellow_pixels / scale_area_pixels) * 100

        # Threshold: 1% (lowered to detect faint yellow lines)
        is_visible = bool(visibility_percent > 1.0)

        return is_visible, float(visibility_percent)

    def is_point_in_polygon(self, point: Tuple[int, int], polygon: List[List[int]]) -> bool:
        """Check if point is inside polygon"""
        pts = np.array(polygon, dtype=np.int32)
        result = cv2.pointPolygonTest(pts, point, measureDist=False)
        return result >= 0

    def get_bbox_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Get center point of bbox"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        return (center_x, center_y)

    def is_bbox_overlapping_polygon_boundary(self, bbox: Tuple[int, int, int, int],
                                              polygon: List[List[int]],
                                              tolerance: int = 40) -> bool:
        """
        Check if bbox edges overlap the polygon boundary
        Checks 4 corners - flags if ANY corner is outside OR very close to edge

        Args:
            bbox: (x1, y1, x2, y2)
            polygon: List of points
            tolerance: Distance tolerance in pixels (default 40)

        Returns:
            True if any corner is outside polygon or 2+ corners very close to edge
        """
        x1, y1, x2, y2 = bbox
        pts = np.array(polygon, dtype=np.int32)

        # Check only 4 corners
        check_points = [
            (x1, y1),  # Top-left corner
            (x2, y1),  # Top-right corner
            (x2, y2),  # Bottom-right corner
            (x1, y2),  # Bottom-left corner
        ]

        corners_outside = 0
        corners_far_outside = 0

        for point in check_points:
            dist = cv2.pointPolygonTest(pts, point, measureDist=True)

            # If corner is outside polygon (negative distance)
            if dist < 0:
                corners_outside += 1
                # If corner is significantly far outside (> tolerance pixels outside)
                if abs(dist) > tolerance:
                    corners_far_outside += 1

        # Flag as overlapping if: 2+ corners far outside (>tolerance px)
        # This allows small bbox inaccuracies but catches real collisions
        return corners_far_outside >= 2

    def calculate_polygon_area(self, polygon: List[List[int]]) -> float:
        """
        Calculate the area of a polygon in square pixels

        Args:
            polygon: List of polygon points

        Returns:
            Area in square pixels
        """
        pts = np.array(polygon, dtype=np.int32)
        return float(cv2.contourArea(pts))

    def calculate_bbox_area(self, bbox: Tuple[int, int, int, int]) -> float:
        """
        Calculate bounding box area

        Args:
            bbox: (x1, y1, x2, y2)

        Returns:
            Area in square pixels
        """
        x1, y1, x2, y2 = bbox
        return float((x2 - x1) * (y2 - y1))

    def create_restricted_margin(self, polygon: List[List[int]], margin_width: int = 50) -> List[List[int]]:
        """
        Create a vertical rectangular margin in the middle of the restricted area

        Args:
            polygon: Original polygon points
            margin_width: Width of the vertical margin strip (default 50 pixels)

        Returns:
            Vertical rectangular margin polygon points (4 corners)
        """
        pts = np.array(polygon, dtype=np.int32)

        # Calculate bounding box and centroid
        x_coords = pts[:, 0]
        y_coords = pts[:, 1]

        min_y = np.min(y_coords)
        max_y = np.max(y_coords)

        # Calculate center x
        M = cv2.moments(pts)
        if M["m00"] == 0:
            center_x = int(np.mean(x_coords))
        else:
            center_x = int(M["m10"] / M["m00"])

        # Create vertical rectangle: center_x ± margin_width/2
        half_width = margin_width // 2

        margin_rect = [
            [center_x - half_width, min_y],  # Top-left
            [center_x + half_width, min_y],  # Top-right
            [center_x + half_width, max_y],  # Bottom-right
            [center_x - half_width, max_y]   # Bottom-left
        ]

        return margin_rect

    def validate(self, image_path: str, config: Dict) -> ValidationResult:
        """
        Validate image

        Validation Rules:
        1. Only consider detections whose CENTER is inside the active area
        2. No objects → Valid
        3. Exactly 1 vehicle → Check if center overlaps with boundary
        4. More than 1 vehicle → Invalid
        5. Any forbidden objects → Invalid

        Args:
            image_path: Path to image
            config: Configuration dictionary

        Returns:
            ValidationResult
        """
        # Extract polygons
        scale_area_points = config['areas']['scale_area']['points']
        prohibited_zone_points = config['areas']['prohibited_zone']['points']

        # Detect yellow lines
        yellow_lines_visible, visibility_percent = self.detect_yellow_lines(
            image_path, scale_area_points
        )

        # Check if any detected vehicle is large (≥92% of prohibited zone)
        # If so, use prohibited zone logic regardless of yellow line visibility
        is_large_vehicle_present = False
        all_detections_temp = self.detect_objects(image_path)
        for det in all_detections_temp:
            if det.class_name in self.ALLOWED_VEHICLE_CLASSES:
                bbox_area = self.calculate_bbox_area(det.bbox)
                prohibited_area = self.calculate_polygon_area(prohibited_zone_points)
                if (bbox_area / prohibited_area) * 100 >= 92.0:
                    is_large_vehicle_present = True
                    break

        # Determine active area based on large vehicle presence or yellow line visibility
        if is_large_vehicle_present:
            # Large vehicle forces prohibited zone logic
            active_area = 'prohibited_zone'
            active_polygon = prohibited_zone_points
            area_name = "red area (prohibited zone)"
        elif yellow_lines_visible:
            active_area = 'scale_area'
            active_polygon = scale_area_points
            area_name = "yellow area (scale area)"
        else:
            active_area = 'prohibited_zone'
            active_polygon = prohibited_zone_points
            area_name = "red area (prohibited zone)"

        # Detect all objects
        all_detections = self.detect_objects(image_path)

        # ✨ KEY FIX: Only consider detections whose CENTER is inside active polygon
        detections_in_area = []
        for det in all_detections:
            center = self.get_bbox_center(det.bbox)
            if self.is_point_in_polygon(center, active_polygon):
                detections_in_area.append(det)

        # Separate by type
        vehicles = [det for det in detections_in_area if det.class_name in self.ALLOWED_VEHICLE_CLASSES]
        forbidden = [det for det in detections_in_area if det.class_name in self.FORBIDDEN_CLASSES]
        other_objects = [det for det in detections_in_area
                        if det.class_name not in self.ALLOWED_VEHICLE_CLASSES
                        and det.class_name not in self.FORBIDDEN_CLASSES]

        # Validation logic
        is_valid = True
        violations = []

        # Rule 1: No objects → Valid (skip)
        if len(detections_in_area) == 0:
            return ValidationResult(
                is_valid=True,
                used_area=active_area,
                yellow_lines_visible=yellow_lines_visible,
                yellow_line_visibility_percent=visibility_percent,
                vehicles_count=0,
                detections=[],
                violations=[],
                message=f"VALID - No objects detected in {area_name}"
            )

        # Rule 2: Must have exactly 1 vehicle
        if len(vehicles) == 0:
            is_valid = False
            violations.append(f"No vehicle detected in {area_name}")
        elif len(vehicles) > 1:
            is_valid = False
            violations.append(f"Multiple vehicles detected in {area_name} ({len(vehicles)} vehicles)")

        # Rule 3: No forbidden objects
        if len(forbidden) > 0:
            is_valid = False
            forbidden_names = ', '.join([f.class_name for f in forbidden])
            violations.append(f"Forbidden objects detected: {forbidden_names}")

        # Rule 4: No other objects (unknown classes)
        if len(other_objects) > 0:
            is_valid = False
            other_names = ', '.join([o.class_name for o in other_objects])
            violations.append(f"Additional objects detected: {other_names}")

        # Rule 5: Check vehicle position and overlaps (if exactly 1 vehicle)
        if len(vehicles) == 1:
            vehicle = vehicles[0]

            # Special logic for prohibited zone (red area)
            if active_area == 'prohibited_zone':
                # Calculate areas
                restricted_area = self.calculate_polygon_area(active_polygon)
                vehicle_bbox_area = self.calculate_bbox_area(vehicle.bbox)
                bbox_percentage = (vehicle_bbox_area / restricted_area) * 100

                # Large vehicle: bbox >= 92% of restricted area
                if bbox_percentage >= 92.0:
                    # Check if center is within margin (vertical rectangle)
                    margin_polygon = self.create_restricted_margin(active_polygon, margin_width=200)
                    vehicle_center = self.get_bbox_center(vehicle.bbox)

                    if not self.is_point_in_polygon(vehicle_center, margin_polygon):
                        is_valid = False
                        violations.append(f"Large vehicle center outside margin - overlaps with red boundary lines")
                else:
                    # Small vehicle: only check overlaps with red boundary
                    if self.is_bbox_overlapping_polygon_boundary(vehicle.bbox, active_polygon, tolerance=40):
                        is_valid = False
                        violations.append(f"Small vehicle overlaps with red boundary lines")

            # Logic for scale area (yellow area) - same as before
            else:
                if self.is_bbox_overlapping_polygon_boundary(vehicle.bbox, active_polygon, tolerance=60):
                    is_valid = False
                    violations.append(f"Vehicle overlaps with {area_name} boundary lines")

        # Generate message
        if is_valid:
            message = f"VALID - Vehicle properly positioned in {area_name}"
        else:
            message = f"INVALID - {area_name}: " + "; ".join(violations)

        return ValidationResult(
            is_valid=is_valid,
            used_area=active_area,
            yellow_lines_visible=yellow_lines_visible,
            yellow_line_visibility_percent=visibility_percent,
            vehicles_count=len(vehicles),
            detections=detections_in_area,
            violations=violations,
            message=message
        )

    def draw_results(self, image_path: str, config: Dict, result: ValidationResult, output_path: str) -> None:
        """
        Draw validation results on image

        Args:
            image_path: Input image path
            config: Configuration
            result: Validation result
            output_path: Output image path
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        # Ensure 1920x1080
        h, w = img.shape[:2]
        if w != 1920 or h != 1080:
            img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_LINEAR)

        # Extract polygons
        scale_pts = np.array(config['areas']['scale_area']['points'], dtype=np.int32)
        prohibited_pts = np.array(config['areas']['prohibited_zone']['points'], dtype=np.int32)

        # Draw prohibited zone (red)
        if result.used_area == 'prohibited_zone':
            # Active - bold
            overlay = img.copy()
            cv2.fillPoly(overlay, [prohibited_pts], (0, 0, 255))
            cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
            cv2.polylines(img, [prohibited_pts], True, (0, 0, 255), 4)
        else:
            # Inactive - light
            cv2.polylines(img, [prohibited_pts], True, (0, 0, 180), 2)

        # Draw scale area (yellow)
        if result.used_area == 'scale_area':
            # Active - bold
            overlay = img.copy()
            cv2.fillPoly(overlay, [scale_pts], (0, 255, 255))
            cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
            cv2.polylines(img, [scale_pts], True, (0, 255, 255), 4)
        else:
            # Inactive - gray
            cv2.polylines(img, [scale_pts], True, (128, 128, 128), 2)

        # Check if we need to draw margin for large vehicle in prohibited zone
        draw_margin = False
        large_vehicle_center = None

        if result.used_area == 'prohibited_zone' and result.vehicles_count == 1:
            vehicles = [det for det in result.detections if det.class_name in self.ALLOWED_VEHICLE_CLASSES]
            if vehicles:
                vehicle = vehicles[0]
                restricted_area = self.calculate_polygon_area(config['areas']['prohibited_zone']['points'])
                vehicle_bbox_area = self.calculate_bbox_area(vehicle.bbox)
                bbox_percentage = (vehicle_bbox_area / restricted_area) * 100

                if bbox_percentage >= 92.0:
                    draw_margin = True
                    large_vehicle_center = self.get_bbox_center(vehicle.bbox)
                    # Draw margin polygon (cyan vertical rectangle)
                    margin_polygon = self.create_restricted_margin(
                        config['areas']['prohibited_zone']['points'],
                        margin_width=200
                    )
                    margin_pts = np.array(margin_polygon, dtype=np.int32)
                    cv2.polylines(img, [margin_pts], True, (255, 255, 0), 3)  # Cyan

        # Draw detections
        for det in result.detections:
            x1, y1, x2, y2 = det.bbox

            # Color based on type
            if det.class_name in self.ALLOWED_VEHICLE_CLASSES:
                color = (0, 255, 0)  # Green for vehicles
            elif det.class_name in self.FORBIDDEN_CLASSES:
                color = (0, 0, 255)  # Red for forbidden
            else:
                color = (255, 165, 0)  # Orange for others

            # Draw bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

            # Draw center point - ONLY for large vehicles in prohibited zone
            is_large_vehicle = (draw_margin and det.class_name in self.ALLOWED_VEHICLE_CLASSES)

            if is_large_vehicle:
                center = self.get_bbox_center(det.bbox)
                # Draw larger, more visible center point for large vehicles
                cv2.circle(img, center, 12, (255, 255, 0), -1)  # Cyan filled circle
                cv2.circle(img, center, 15, (255, 255, 255), 3)  # White border

            # Draw label
            label = f"{det.class_name} {det.confidence:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1-label_h-10), (x1+label_w+10, y1), color, -1)
            cv2.putText(img, label, (x1+5, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Status panel
        status_color = (0, 255, 0) if result.is_valid else (0, 0, 255)
        status_text = "VALID" if result.is_valid else "INVALID"

        # Panel height based on violations
        base_panel_height = 180
        violation_height = len(result.violations) * 30 if result.violations else 0
        panel_height = base_panel_height + violation_height

        # Draw panel
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (900, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        cv2.rectangle(img, (10, 10), (900, panel_height), status_color, 3)

        y_offset = 50
        # Status
        cv2.putText(img, f"Validation: {status_text}",
                   (25, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.3, status_color, 3)

        y_offset += 45
        # Active area
        area_text = "Scale Area (Yellow Lines)" if result.used_area == 'scale_area' else "Prohibited Zone"
        area_color = (0, 255, 255) if result.used_area == 'scale_area' else (0, 0, 255)
        cv2.putText(img, f"Active Area: {area_text}",
                   (25, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, area_color, 2)

        y_offset += 35
        # Yellow lines status
        yellow_status = 'Visible' if result.yellow_lines_visible else 'Masked'
        yellow_color = (0, 255, 255) if result.yellow_lines_visible else (128, 128, 128)
        cv2.putText(img, f"Yellow Lines: {yellow_status} ({result.yellow_line_visibility_percent:.1f}%)",
                   (25, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.65, yellow_color, 2)

        y_offset += 35
        # Detection stats
        cv2.putText(img, f"Vehicles: {result.vehicles_count} | Total Detections: {len(result.detections)}",
                   (25, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Violations
        if result.violations:
            y_offset += 40
            cv2.putText(img, "Violations:",
                       (25, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
            y_offset += 30

            for violation in result.violations:
                cv2.putText(img, f"  - {violation}",
                           (35, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 2)
                y_offset += 30

        # Legend
        legend_y = 1080 - 130
        overlay = img.copy()
        cv2.rectangle(overlay, (10, legend_y-10), (600, 1070), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        cv2.rectangle(img, (10, legend_y-10), (600, 1070), (100, 100, 100), 2)

        cv2.putText(img, "Legend:", (20, legend_y+10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        legend_y += 25
        if result.used_area == 'scale_area':
            cv2.rectangle(img, (20, legend_y), (45, legend_y+18), (0, 255, 255), -1)
            cv2.putText(img, "Active: Scale Area (Yellow)", (55, legend_y+15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            cv2.rectangle(img, (20, legend_y), (45, legend_y+18), (0, 0, 255), -1)
            cv2.putText(img, "Active: Prohibited Zone (Red)", (55, legend_y+15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        legend_y += 30
        cv2.rectangle(img, (20, legend_y), (45, legend_y+18), (0, 255, 0), -1)
        cv2.putText(img, "Vehicle (Allowed)", (55, legend_y+15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        legend_y += 30
        cv2.rectangle(img, (20, legend_y), (45, legend_y+18), (0, 0, 255), -1)
        cv2.putText(img, "Forbidden (Bicycle/Motorcycle)", (55, legend_y+15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Save
        cv2.imwrite(output_path, img)
