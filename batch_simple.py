#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Processing with Simple Validator
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, 'src')
from simple_validator import SimpleValidator

def main():
    parser = argparse.ArgumentParser(description='Batch process weighbridge images')
    parser.add_argument('-w', '--weighbridge', type=int, required=True, choices=[1, 2, 3],
                       help='Weighbridge ID (1, 2, or 3)')
    parser.add_argument('-f', '--folder', type=str, required=True,
                       help='Input folder with images')
    parser.add_argument('-o', '--output', type=str, default='outputs',
                       help='Output folder')
    parser.add_argument('-c', '--confidence', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('-m', '--model', type=str, default='weights/best.pt',
                       help='Model path (default: weights/best.pt)')

    args = parser.parse_args()

    print("=" * 80)
    print("Batch Processing - Weighbridge Images (Simple Validator)")
    print("=" * 80)
    print(f"Input Folder: {args.folder}")
    print(f"Weighbridge: {args.weighbridge}")
    print(f"Confidence Threshold: {args.confidence}")
    print(f"Model: {args.model}")
    print("=" * 80)

    # Load config
    config_path = f'configs/wb{args.weighbridge}_single_cam_roi.json'
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        print(f"Please create the config file using weighbridge_config_simple.html")
        return 1

    print(f"\n📁 Loading config: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Find images
    print(f"\n🔍 Finding images in {args.folder}...")
    image_extensions = {'.jpg', '.jpeg', '.png'}
    images = []
    for ext in image_extensions:
        images.extend(Path(args.folder).glob(f'*{ext}'))
        images.extend(Path(args.folder).glob(f'*{ext.upper()}'))

    images = sorted(images)
    print(f"✅ Found {len(images)} images\n")

    if len(images) == 0:
        print("❌ No images found!")
        return 1

    # Create output folder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output) / f'wb{args.weighbridge}_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output folder: {output_dir}\n")

    # Load validator
    print(f"🤖 Loading model: {args.model}")
    validator = SimpleValidator(model_path=args.model, confidence_threshold=args.confidence)

    # Process images
    print("\n" + "=" * 80)
    print("Processing images...")
    print("=" * 80 + "\n")

    results = []
    valid_count = 0
    invalid_count = 0

    for i, image_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {image_path.name}")
        print("-" * 80)

        try:
            # Validate
            result = validator.validate(str(image_path), config)

            # Draw results
            output_path = output_dir / f"{image_path.stem}_annotated{image_path.suffix}"
            validator.draw_results(str(image_path), config, result, str(output_path))

            # Count
            if result.is_valid:
                valid_count += 1
                status_symbol = "✅"
            else:
                invalid_count += 1
                status_symbol = "❌"

            print(f"{status_symbol} Status: {'VALID' if result.is_valid else 'INVALID'}")
            print(f"🟡 Yellow Lines: {'Visible' if result.yellow_lines_visible else 'Masked'} ({result.yellow_line_visibility_percent:.1f}%)")
            print(f"📍 Active Area: {result.used_area}")
            print(f"🚗 Detections: Vehicles={result.vehicles_count}, Total={len(result.detections)}")

            if result.violations:
                print(f"⚠️  Violations:")
                for v in result.violations:
                    print(f"   - {v}")

            print(f"💾 Saved: {output_path.name}")

            # Store result
            results.append({
                'image': image_path.name,
                'is_valid': result.is_valid,
                'used_area': result.used_area,
                'yellow_lines_visible': result.yellow_lines_visible,
                'yellow_line_visibility_percent': result.yellow_line_visibility_percent,
                'vehicles_count': result.vehicles_count,
                'total_detections': len(result.detections),
                'violations': result.violations,
                'message': result.message
            })

        except Exception as e:
            print(f"❌ Error processing {image_path.name}: {e}")
            invalid_count += 1
            results.append({
                'image': image_path.name,
                'is_valid': False,
                'error': str(e)
            })

        print()

    # Save JSON report
    report_path = output_dir / 'batch_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            'weighbridge_id': args.weighbridge,
            'timestamp': timestamp,
            'total_images': len(images),
            'valid_count': valid_count,
            'invalid_count': invalid_count,
            'success_rate': (valid_count / len(images) * 100) if len(images) > 0 else 0,
            'results': results
        }, f, indent=2, ensure_ascii=False)

    # Print summary
    print("=" * 80)
    print("Final Summary")
    print("=" * 80)
    print(f"\n📊 Results:")
    print(f"   Total Images: {len(images)}")
    print(f"   Processed: {valid_count + invalid_count}")
    print(f"   ✅ Valid: {valid_count}")
    print(f"   ❌ Invalid: {invalid_count}")
    print(f"   📈 Success Rate: {(valid_count / len(images) * 100):.1f}%")
    print(f"\n📄 Report: {report_path}")
    print(f"📁 Output Images: {output_dir}")
    print("\n" + "=" * 80)
    print("✅ Batch processing completed successfully")
    print("=" * 80 + "\n")

    return 0

if __name__ == '__main__':
    sys.exit(main())
