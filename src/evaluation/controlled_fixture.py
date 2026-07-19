import argparse
import csv
import os

import cv2
import numpy as np


def render_controlled_face(eyes_open=True, mouth_open=False, brightness=210):
    """Render a deterministic fixture for functional state-separation tests."""
    image = np.full((240, 240), brightness, dtype=np.uint8)
    skin = max(80, brightness - 45)
    dark = max(5, brightness - 185)
    cv2.ellipse(image, (120, 120), (96, 112), 0, 0, 360, skin, -1)

    for center_x in (70, 170):
        if eyes_open:
            cv2.ellipse(image, (center_x, 91), (31, 25), 0, 0, 360, dark, 2)
            cv2.circle(image, (center_x, 91), 14, dark, -1)
            cv2.circle(
                image,
                (center_x - 3, 87),
                3,
                min(255, brightness + 15),
                -1,
            )
        else:
            cv2.line(
                image,
                (center_x - 30, 92),
                (center_x + 30, 92),
                dark,
                4,
            )

    if mouth_open:
        cv2.ellipse(image, (120, 174), (37, 28), 0, 0, 360, dark, -1)
        cv2.ellipse(
            image,
            (120, 174),
            (38, 29),
            0,
            0,
            360,
            min(100, dark + 45),
            2,
        )
    else:
        cv2.line(image, (82, 174), (158, 174), dark, 4)
    return image


def write_controlled_fixture(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for brightness in (165, 210, 235):
        for eyes_open, mouth_open in (
            (True, False),
            (True, True),
            (False, False),
            (False, True),
        ):
            sample = "b{brightness}_{eye}_{mouth}".format(
                brightness=brightness,
                eye="open" if eyes_open else "closed",
                mouth="yawn" if mouth_open else "normal",
            )
            filename = f"{sample}.png"
            image = render_controlled_face(
                eyes_open=eyes_open,
                mouth_open=mouth_open,
                brightness=brightness,
            )
            image_path = os.path.join(output_dir, filename)
            if not cv2.imwrite(image_path, image):
                raise IOError(f"Cannot write fixture image: {image_path}")
            rows.append(
                {
                    "sample": sample,
                    "image": filename,
                    "x1": 0,
                    "y1": 0,
                    "x2": image.shape[1],
                    "y2": image.shape[0],
                    "eye_closed": int(not eyes_open),
                    "yawn": int(mouth_open),
                }
            )

    manifest_path = os.path.join(output_dir, "roi_manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate deterministic eye-closure/yawn ROI fixtures."
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(write_controlled_fixture(args.output))


if __name__ == "__main__":
    main()
