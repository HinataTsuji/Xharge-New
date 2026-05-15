"""Example script to call the backend APIs with a local test image."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description="Run example inference against local solar backend")
    parser.add_argument("--image", required=True, type=Path, help="Path to rooftop image")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Backend base URL")
    parser.add_argument("--meters-per-pixel", default=0.1, type=float)
    args = parser.parse_args()

    image_bytes = args.image.read_bytes()
    with Image.open(args.image) as im:
        image_width, image_height = im.size

    with httpx.Client(timeout=60) as client:
        response = client.post(
            f"{args.base_url}/analyze-roof",
            files={"image": (args.image.name, image_bytes, "image/png")},
            data={"meters_per_pixel": str(args.meters_per_pixel), "backend": "auto"},
        )
        response.raise_for_status()
        analysis = response.json()
        print("Analyze roof response:")
        print(json.dumps(analysis, indent=2))

        roof_polygon = analysis["data"]["roof_polygon"]
        obstacles = analysis["data"]["obstacles"]
        estimate_payload = {
            "image_width": image_width,
            "image_height": image_height,
            "roof_polygon": roof_polygon,
            "obstacles": obstacles,
            "meters_per_pixel": args.meters_per_pixel,
            "panel_config": {
                "panel_width_m": 1.134,
                "panel_height_m": 2.278,
                "row_spacing_m": 0.1,
                "col_spacing_m": 0.1,
                "panel_power_kw": 0.62,
            },
            "annual_yield_factor_kwh_per_kw": 1450.0,
        }
        estimate = client.post(f"{args.base_url}/estimate-panels", json=estimate_payload)
        estimate.raise_for_status()
        print("Estimate panels response:")
        print(json.dumps(estimate.json(), indent=2))


if __name__ == "__main__":
    main()
