import argparse
import csv
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm


LIPS_IDX = sorted(
    {
        0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95,
        146, 178, 181, 185, 191, 267, 269, 270, 291, 292, 308, 310, 311,
        312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415,
    }
)


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def smooth_box(previous, current, alpha):
    if previous is None:
        return current
    return tuple(int(alpha * old + (1 - alpha) * new) for old, new in zip(previous, current))


def fallback_box(width, height):
    size = int(min(width, height) * 0.46)
    cx = int(width * 0.50)
    cy = int(height * 0.64)
    half = max(8, size // 2)
    return (
        clamp(cx - half, 0, width - 1),
        clamp(cy - half, 0, height - 1),
        clamp(cx + half, 0, width - 1),
        clamp(cy + half, 0, height - 1),
    )


def crop_video(source, target, out_size, pad_ratio, smooth_alpha, max_no_face, min_det, min_track):
    target.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_size, out_size))
    face_mesh_api = mp.solutions.face_mesh
    previous_box = None
    no_face_count = 0
    frames = 0
    fallback_frames = 0

    with face_mesh_api.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=min_det,
        min_tracking_confidence=min_track,
    ) as face_mesh:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames += 1
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            if result.multi_face_landmarks:
                no_face_count = 0
                landmarks = result.multi_face_landmarks[0].landmark
                xs = np.array([landmarks[idx].x * width for idx in LIPS_IDX], dtype=np.float32)
                ys = np.array([landmarks[idx].y * height for idx in LIPS_IDX], dtype=np.float32)
                x1, x2 = float(xs.min()), float(xs.max())
                y1, y2 = float(ys.min()), float(ys.max())
                box_w = max(1.0, x2 - x1)
                box_h = max(1.0, y2 - y1)
                pad_x = box_w * pad_ratio
                pad_y = box_h * pad_ratio
                current_box = (
                    clamp(int(x1 - pad_x), 0, width - 1),
                    clamp(int(y1 - pad_y), 0, height - 1),
                    clamp(int(x2 + pad_x), 0, width - 1),
                    clamp(int(y2 + pad_y), 0, height - 1),
                )
                previous_box = smooth_box(previous_box, current_box, smooth_alpha)
            else:
                no_face_count += 1
                fallback_frames += 1
                if previous_box is None or no_face_count > max_no_face:
                    previous_box = fallback_box(width, height)

            x1, y1, x2, y2 = previous_box
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                fallback_frames += 1
                crop = np.zeros((out_size, out_size, 3), dtype=np.uint8)
            else:
                crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
            writer.write(crop)

    cap.release()
    writer.release()
    return {"frames": frames, "fallback_frames": fallback_frames}


def load_rows(metadata_csv, split):
    rows = []
    with Path(metadata_csv).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if split and row["split"] != split:
                continue
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--metadata", default=str(repo_root / "prepared" / "vlips" / "manifest" / "metadata.csv"))
    parser.add_argument("--video-dir", default=str(repo_root / "prepared" / "vlips" / "video"))
    parser.add_argument("--split", choices=["train", "valid", "test"], default=None)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--nshard", type=int, default=1)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--out-size", type=int, default=96)
    parser.add_argument("--pad-ratio", type=float, default=0.45)
    parser.add_argument("--smooth-alpha", type=float, default=0.80)
    parser.add_argument("--max-no-face", type=int, default=8)
    parser.add_argument("--min-det", type=float, default=0.5)
    parser.add_argument("--min-track", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = load_rows(args.metadata, args.split)
    if args.nshard > 1:
        rows = rows[args.rank :: args.nshard]
    if args.max_items:
        rows = rows[: args.max_items]

    video_dir = Path(args.video_dir)
    failures = []
    stats = []
    for row in tqdm(rows, desc=f"crop rank {args.rank}/{args.nshard}"):
        source = Path(row["source_video"])
        target = video_dir / f"{row['fid']}.mp4"
        if target.exists() and not args.overwrite:
            continue
        try:
            info = crop_video(
                source,
                target,
                args.out_size,
                args.pad_ratio,
                args.smooth_alpha,
                args.max_no_face,
                args.min_det,
                args.min_track,
            )
            stats.append({"fid": row["fid"], **info})
        except Exception as exc:
            failures.append({"fid": row["fid"], "source": str(source), "error": str(exc)})

    log_dir = video_dir.parent / "crop_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.split or 'all'}.{args.rank}-of-{args.nshard}"
    (log_dir / f"failures.{suffix}.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    with (log_dir / f"stats.{suffix}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fid", "frames", "fallback_frames"])
        writer.writeheader()
        writer.writerows(stats)
    print(json.dumps({"processed": len(stats), "failures": len(failures), "skipped_or_existing": len(rows) - len(stats) - len(failures)}, indent=2))


if __name__ == "__main__":
    main()
