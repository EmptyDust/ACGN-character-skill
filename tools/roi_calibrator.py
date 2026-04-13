"""
ROI Calibration Tool

Interactive tool for defining and validating ROI (Region of Interest) regions
for dialogue box and name box extraction.

This tool helps users:
1. Load sample frames from video
2. Visually define ROI regions
3. Validate ROI across multiple frames
4. Save calibrated configuration

Usage:
    # Create new configuration
    python roi_calibrator.py --video video.mp4 --output config.yaml

    # Validate existing configuration
    python roi_calibrator.py --config config.yaml --video video.mp4 --validate

    # Extract ROI crops for manual inspection
    python roi_calibrator.py --config config.yaml --video video.mp4 --extract-crops
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional
import sys

# Note: This tool provides CLI-based ROI definition
# For GUI-based calibration, opencv-python with cv2.selectROI() can be used

try:
    from .roi_config import ROIConfig, create_default_config
except ImportError:
    # Allow running as script
    sys.path.insert(0, str(Path(__file__).parent))
    from roi_config import ROIConfig, create_default_config


class ROICalibrator:
    """ROI calibration tool."""

    def __init__(self, video_path: Optional[Path] = None):
        """Initialize calibrator."""
        self.video_path = video_path
        self.config = None

    def create_interactive_config(self, work_id: str, width: int, height: int, frame_path: Optional[Path] = None) -> ROIConfig:
        """
        Create ROI configuration interactively.

        If frame_path is provided and opencv is available, uses GUI-based ROI selection.
        Otherwise falls back to CLI-based coordinate input.
        """
        print(f"\n=== ROI Configuration for {work_id} ===")
        print(f"Video resolution: {width}x{height}")
        print()

        config_dict = create_default_config(work_id, width, height)

        # Try GUI-based calibration if frame is provided
        if frame_path:
            try:
                import cv2
                import numpy as np

                frame = cv2.imread(str(frame_path))
                if frame is None:
                    print(f"Warning: Could not load frame {frame_path}")
                    print("Falling back to CLI-based input")
                else:
                    print(f"Loaded frame: {frame_path}")
                    print(f"Frame size: {frame.shape[1]}x{frame.shape[0]}")
                    print()

                    # Select dialogue box ROI
                    print("Select DIALOGUE BOX region (bottom dialogue text area)")
                    print("  - Click and drag to select the region")
                    print("  - Press ENTER to confirm")
                    print("  - Press C to cancel and use defaults")

                    dialogue_roi = cv2.selectROI("Select Dialogue Box ROI", frame, fromCenter=False, showCrosshair=True)
                    cv2.destroyAllWindows()

                    if dialogue_roi[2] > 0 and dialogue_roi[3] > 0:
                        config_dict['dialogue_box']['roi'] = {
                            'x': int(dialogue_roi[0]),
                            'y': int(dialogue_roi[1]),
                            'width': int(dialogue_roi[2]),
                            'height': int(dialogue_roi[3]),
                            'notes': f'GUI-calibrated from {frame_path.name}'
                        }
                        print(f"Dialogue box ROI: x={dialogue_roi[0]}, y={dialogue_roi[1]}, w={dialogue_roi[2]}, h={dialogue_roi[3]}")
                    else:
                        print("No selection made, using defaults")

                    print()

                    # Select name box ROI
                    print("Select NAME BOX region (character name above dialogue)")
                    print("  - Click and drag to select the region")
                    print("  - Press ENTER to confirm")
                    print("  - Press C to cancel and use defaults")

                    name_roi = cv2.selectROI("Select Name Box ROI", frame, fromCenter=False, showCrosshair=True)
                    cv2.destroyAllWindows()

                    if name_roi[2] > 0 and name_roi[3] > 0:
                        config_dict['name_box']['roi'] = {
                            'x': int(name_roi[0]),
                            'y': int(name_roi[1]),
                            'width': int(name_roi[2]),
                            'height': int(name_roi[3]),
                            'notes': f'GUI-calibrated from {frame_path.name}'
                        }
                        print(f"Name box ROI: x={name_roi[0]}, y={name_roi[1]}, w={name_roi[2]}, h={name_roi[3]}")
                    else:
                        print("No selection made, using defaults")

                    # Mark as calibrated
                    config_dict['validation']['calibrated'] = True
                    config_dict['validation']['calibration_date'] = None  # Will be set on save
                    config_dict['validation']['calibration_frames'] = [str(frame_path)]
                    config_dict['validation']['notes'] = 'GUI-calibrated with cv2.selectROI()'

                    return ROIConfig(config_dict)

            except ImportError:
                print("opencv-python not installed, falling back to CLI-based input")
                print("Install with: pip install opencv-python")
                print()

        # CLI-based fallback
        return self._create_cli_config(work_id, width, height, config_dict)

    def _create_cli_config(self, work_id: str, width: int, height: int, config_dict: dict) -> ROIConfig:
        """Create configuration via CLI prompts."""
        # Dialogue box configuration
        print("Dialogue Box ROI:")
        print("  Default: x=90, y=785, width=1540, height=170")
        print("  (Bottom-center region where dialogue text appears)")

        if self._confirm("Use default dialogue box ROI?"):
            pass  # Keep default
        else:
            config_dict['dialogue_box']['roi'] = self._input_roi("dialogue box")

        # Name box configuration
        print("\nName Box ROI:")
        print("  Default: x=95, y=708, width=210, height=78")
        print("  (Above dialogue box, shows character name)")

        if self._confirm("Use default name box ROI?"):
            pass  # Keep default
        else:
            config_dict['name_box']['roi'] = self._input_roi("name box")

        return ROIConfig(config_dict)

    def _confirm(self, message: str) -> bool:
        """Ask user for yes/no confirmation."""
        while True:
            response = input(f"{message} [y/n]: ").strip().lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("Please enter 'y' or 'n'")

    def _input_roi(self, box_name: str) -> dict:
        """Input ROI coordinates interactively."""
        print(f"\nEnter {box_name} ROI coordinates:")
        x = int(input("  x (left edge): "))
        y = int(input("  y (top edge): "))
        width = int(input("  width: "))
        height = int(input("  height: "))

        return {
            'x': x,
            'y': y,
            'width': width,
            'height': height,
            'notes': f'Manually calibrated for {box_name}'
        }

    def validate_config(self, config: ROIConfig, frame_paths: List[Path]) -> bool:
        """
        Validate ROI configuration across multiple frames.

        Returns True if validation passes, False otherwise.
        """
        print(f"\n=== Validating ROI Configuration ===")
        print(f"Config: {config.config['work_id']}")
        print(f"Resolution: {config.get_resolution()}")
        print(f"Dialogue box ROI: {config.get_dialogue_box_roi()}")
        print(f"Name box ROI: {config.get_name_box_roi()}")
        print()

        if not frame_paths:
            print("Warning: No frames provided for validation")
            return False

        print(f"Validating against {len(frame_paths)} frames...")

        # Check if frames exist
        missing_frames = [f for f in frame_paths if not f.exists()]
        if missing_frames:
            print(f"Error: {len(missing_frames)} frames not found")
            for f in missing_frames[:5]:
                print(f"  - {f}")
            return False

        # Try visual validation with opencv
        try:
            import cv2

            dialogue_roi = config.get_dialogue_box_roi()
            name_roi = config.get_name_box_roi()

            issues = []

            for i, frame_path in enumerate(frame_paths):
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    issues.append(f"Frame {i}: Could not read {frame_path}")
                    continue

                frame_h, frame_w = frame.shape[:2]

                # Check dialogue box bounds
                dx, dy, dw, dh = dialogue_roi
                if dx < 0 or dy < 0 or dx + dw > frame_w or dy + dh > frame_h:
                    issues.append(f"Frame {i}: Dialogue box ROI out of bounds ({frame_w}x{frame_h})")

                # Check name box bounds
                nx, ny, nw, nh = name_roi
                if nx < 0 or ny < 0 or nx + nw > frame_w or ny + nh > frame_h:
                    issues.append(f"Frame {i}: Name box ROI out of bounds ({frame_w}x{frame_h})")

                # Check if ROIs are too small
                if dw < 100 or dh < 20:
                    issues.append(f"Frame {i}: Dialogue box ROI too small ({dw}x{dh})")
                if nw < 50 or nh < 20:
                    issues.append(f"Frame {i}: Name box ROI too small ({nw}x{nh})")

            if issues:
                print("\nValidation issues found:")
                for issue in issues:
                    print(f"  - {issue}")
                return False

            print("[OK] Configuration structure is valid")
            print("[OK] All validation frames exist")
            print("[OK] ROI bounds are valid across all frames")
            print("[OK] ROI dimensions are reasonable")

            return True

        except ImportError:
            # Fallback to basic validation without opencv
            print("[OK] Configuration structure is valid")
            print("[OK] All validation frames exist")
            print()
            print("Note: Visual validation requires opencv-python")
            print("      Install with: pip install opencv-python")
            print("      Then re-run validation for full ROI quality checks")

            return True

    def extract_roi_crops(self, config: ROIConfig, frame_paths: List[Path], output_dir: Path):
        """
        Extract ROI crops from frames for manual inspection.

        Requires opencv-python to be installed.
        """
        try:
            import cv2
        except ImportError:
            print("Error: opencv-python not installed")
            print("Install with: pip install opencv-python")
            return False

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dialogue_roi = config.get_dialogue_box_roi()
        name_roi = config.get_name_box_roi()

        print(f"\n=== Extracting ROI Crops ===")
        print(f"Output directory: {output_dir}")
        print(f"Processing {len(frame_paths)} frames...")

        for i, frame_path in enumerate(frame_paths):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                print(f"Warning: Could not read {frame_path}")
                continue

            # Extract dialogue box crop
            x, y, w, h = dialogue_roi
            dialogue_crop = frame[y:y+h, x:x+w]
            dialogue_output = output_dir / f"dialogue_{i:03d}.jpg"
            cv2.imwrite(str(dialogue_output), dialogue_crop)

            # Extract name box crop
            x, y, w, h = name_roi
            name_crop = frame[y:y+h, x:x+w]
            name_output = output_dir / f"name_{i:03d}.jpg"
            cv2.imwrite(str(name_output), name_crop)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1} frames...")

        print(f"\n[OK] Extracted crops saved to: {output_dir}")
        print(f"  - dialogue_*.jpg: Dialogue box crops")
        print(f"  - name_*.jpg: Name box crops")
        print(f"\nManually inspect these crops to verify ROI accuracy")

        return True


def main():
    parser = argparse.ArgumentParser(description="ROI Calibration Tool")

    # Input/output
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--config", type=str, help="Path to existing config file")
    parser.add_argument("--output", "-o", type=str, help="Output config file path")

    # Operations
    parser.add_argument("--create", action="store_true",
                       help="Create new configuration interactively")
    parser.add_argument("--validate", action="store_true",
                       help="Validate existing configuration")
    parser.add_argument("--extract-crops", action="store_true",
                       help="Extract ROI crops for manual inspection")

    # Parameters
    parser.add_argument("--work-id", type=str, default="default",
                       help="Work identifier for new configuration")
    parser.add_argument("--width", type=int, default=1920,
                       help="Video width (default: 1920)")
    parser.add_argument("--height", type=int, default=1080,
                       help="Video height (default: 1080)")
    parser.add_argument("--frame", type=str,
                       help="Sample frame for GUI-based ROI selection (used with --create)")
    parser.add_argument("--frames", type=str, nargs='+',
                       help="Frame paths for validation")
    parser.add_argument("--crops-output", type=str, default="roi_crops",
                       help="Output directory for extracted crops")

    args = parser.parse_args()

    calibrator = ROICalibrator(Path(args.video) if args.video else None)

    # Create new configuration
    if args.create:
        frame_path = Path(args.frame) if args.frame else None
        config = calibrator.create_interactive_config(args.work_id, args.width, args.height, frame_path)

        if args.output:
            output_path = Path(args.output)
            config.save(output_path, format='yaml' if output_path.suffix in ['.yaml', '.yml'] else 'json')
            print(f"\n[OK] Configuration saved to: {output_path}")
        else:
            print("\nConfiguration created but not saved (use --output to save)")

    # Validate existing configuration
    elif args.validate:
        if not args.config:
            print("Error: --config required for validation")
            return 1

        config = ROIConfig.from_file(Path(args.config))
        frame_paths = [Path(f) for f in args.frames] if args.frames else []

        if calibrator.validate_config(config, frame_paths):
            print("\n[OK] Validation passed")
        else:
            print("\n[X] Validation failed")
            return 1

    # Extract ROI crops
    elif args.extract_crops:
        if not args.config:
            print("Error: --config required for crop extraction")
            return 1

        if not args.frames:
            print("Error: --frames required for crop extraction")
            return 1

        config = ROIConfig.from_file(Path(args.config))
        frame_paths = [Path(f) for f in args.frames]
        output_dir = Path(args.crops_output)

        if not calibrator.extract_roi_crops(config, frame_paths, output_dir):
            return 1

    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Create new configuration with GUI-based ROI selection (recommended)")
        print("  python roi_calibrator.py --create --work-id yuexia_ep01 --frame sample.jpg --output config.yaml")
        print()
        print("  # Create new configuration with CLI-based input")
        print("  python roi_calibrator.py --create --work-id yuexia_ep01 --output config.yaml")
        print()
        print("  # Validate configuration with sample frames")
        print("  python roi_calibrator.py --validate --config config.yaml --frames frame1.jpg frame2.jpg")
        print()
        print("  # Extract ROI crops for manual inspection")
        print("  python roi_calibrator.py --extract-crops --config config.yaml --frames frame*.jpg --crops-output crops/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
