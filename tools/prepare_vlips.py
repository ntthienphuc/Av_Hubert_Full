import argparse
import csv
import json
import math
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path

import sentencepiece as spm
from tqdm import tqdm


SPECIAL_TOKENS = {
    "<s>": 0,
    "<pad>": 1,
    "</s>": 2,
    "<unk>": 3,
}


def normalize_text(text):
    text = unicodedata.normalize("NFC", text)
    text = text.strip()
    if text.lower().startswith("text:"):
        text = text.split(":", 1)[1].strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def read_label_and_end(txt_path):
    last_end = 0.0
    with txt_path.open("r", encoding="utf-8-sig") as handle:
        label = None
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if label is None:
                label = normalize_text(line)
                continue
            parts = line.split()
            if len(parts) >= 3 and parts[0].lower() != "word":
                try:
                    last_end = max(last_end, float(parts[2]))
                except ValueError:
                    pass
        if label:
            return label, last_end
    raise ValueError(f"empty label file: {txt_path}")


def split_entry_to_source_path(dataset_root, entry):
    entry = entry.strip().replace("\\", "/")
    if not entry:
        return None
    return dataset_root / entry


def source_to_fid(source_path, source_root):
    rel = source_path.relative_to(source_root).with_suffix("")
    return rel.as_posix()


def estimate_video_info(label_end, fps):
    duration = max(0.04, label_end + 0.20)
    frames = int(math.ceil(duration * fps))
    if frames <= 0:
        return None
    return {
        "frames": frames,
        "fps": fps,
        "width": 0,
        "height": 0,
        "duration": duration,
    }


def train_sentencepiece(labels, out_dir, vocab_size):
    out_dir.mkdir(parents=True, exist_ok=True)
    text_path = out_dir / "spm_train_text.txt"
    prefix = out_dir / f"spm_unigram{vocab_size}"
    with text_path.open("w", encoding="utf-8", newline="\n") as handle:
        for label in labels:
            handle.write(label + "\n")

    args = [
        f"--input={text_path.as_posix()}",
        f"--model_prefix={prefix.as_posix()}",
        "--model_type=unigram",
        f"--vocab_size={vocab_size}",
        "--character_coverage=1.0",
        f"--unk_id={SPECIAL_TOKENS['<unk>']}",
        f"--bos_id={SPECIAL_TOKENS['<s>']}",
        f"--eos_id={SPECIAL_TOKENS['</s>']}",
        f"--pad_id={SPECIAL_TOKENS['<pad>']}",
    ]
    spm.SentencePieceTrainer.Train(" ".join(args))

    processor = spm.SentencePieceProcessor()
    processor.Load(prefix.as_posix() + ".model")
    with (prefix.with_suffix(".txt")).open("w", encoding="utf-8", newline="\n") as handle:
        for idx in range(processor.GetPieceSize()):
            piece = processor.IdToPiece(idx)
            if piece not in SPECIAL_TOKENS:
                handle.write(f"{piece} 1\n")
    return prefix.with_suffix(".model"), prefix.with_suffix(".txt")


def write_manifest_split(out_dir, split, rows, video_dir, audio_placeholder):
    tsv_path = out_dir / f"{split}.tsv"
    wrd_path = out_dir / f"{split}.wrd"
    with tsv_path.open("w", encoding="utf-8", newline="\n") as tsv:
        tsv.write("/\n")
        for row in rows:
            video_path = video_dir / f"{row['fid']}.mp4"
            tsv.write(
                "\t".join(
                    [
                        row["fid"],
                        str(video_path.resolve()),
                        str(audio_placeholder.resolve()),
                        str(row["frames"]),
                        str(row["frames"]),
                    ]
                )
                + "\n"
            )
    with wrd_path.open("w", encoding="utf-8", newline="\n") as wrd:
        for row in rows:
            wrd.write(row["label"] + "\n")


def write_silent_wav(path):
    import wave

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16000)


def main():
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--out-root", default=str(repo_root / "prepared" / "vlips"))
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--skip-existing-spm", action="store_true")
    parser.add_argument("--fps", type=float, default=29.0)
    args = parser.parse_args()

    if not args.dataset_root:
        raise SystemExit("Missing --dataset-root. Point this to the VLips LRS2-format dataset root.")
    dataset_root = Path(args.dataset_root)
    split_root = dataset_root / "split"
    source_root = dataset_root / "data_output" / "Vlips_v1_video_non_audio"
    out_root = Path(args.out_root)
    manifest_dir = out_root / "manifest"
    video_dir = out_root / "video"
    audio_placeholder = out_root / "audio" / "silent.wav"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    write_silent_wav(audio_placeholder)

    all_rows = []
    split_rows = {}
    errors = []
    seen_fids = set()
    for split_name, split_file in [("train", "train.txt"), ("valid", "val.txt"), ("test", "test.txt")]:
        rows = []
        entries = (split_root / split_file).read_text(encoding="utf-8").splitlines()
        if args.max_items:
            entries = entries[: args.max_items]
        for entry in tqdm(entries, desc=f"scan {split_name}"):
            source_path = split_entry_to_source_path(dataset_root, entry)
            if source_path is None:
                continue
            if not source_path.is_file():
                errors.append({"split": split_name, "path": str(source_path), "error": "missing_video"})
                continue
            if not source_path.is_relative_to(source_root):
                errors.append({"split": split_name, "path": str(source_path), "error": "outside_source_root"})
                continue
            fid = source_to_fid(source_path, source_root)
            if fid in seen_fids:
                errors.append({"split": split_name, "fid": fid, "error": "duplicate_fid"})
                continue
            txt_path = source_path.with_suffix(".txt")
            if not txt_path.is_file():
                errors.append({"split": split_name, "fid": fid, "error": "missing_label"})
                continue
            try:
                label, label_end = read_label_and_end(txt_path)
            except Exception as exc:
                errors.append({"split": split_name, "fid": fid, "error": f"bad_label:{exc}"})
                continue
            info = estimate_video_info(label_end, args.fps)
            if info is None:
                errors.append({"split": split_name, "fid": fid, "error": "bad_video"})
                continue
            row = {
                "split": split_name,
                "fid": fid,
                "source_video": str(source_path),
                "source_label": str(txt_path),
                "label": label,
                **info,
            }
            rows.append(row)
            all_rows.append(row)
            seen_fids.add(fid)
        split_rows[split_name] = rows

    with (manifest_dir / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["split", "fid", "source_video", "source_label", "label", "frames", "fps", "width", "height", "duration"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    with (manifest_dir / "file.list").open("w", encoding="utf-8", newline="\n") as handle:
        for row in all_rows:
            handle.write(row["fid"] + "\n")
    with (manifest_dir / "label.list").open("w", encoding="utf-8", newline="\n") as handle:
        for row in all_rows:
            handle.write(row["label"] + "\n")

    with (manifest_dir / "errors.json").open("w", encoding="utf-8") as handle:
        json.dump(errors, handle, ensure_ascii=False, indent=2)

    spm_model = manifest_dir / f"spm_unigram{args.vocab_size}.model"
    spm_dict = manifest_dir / f"spm_unigram{args.vocab_size}.txt"
    if not args.skip_existing_spm or not spm_model.exists() or not spm_dict.exists():
        spm_model, spm_dict = train_sentencepiece([row["label"] for row in split_rows["train"]], manifest_dir, args.vocab_size)

    for split_name, rows in split_rows.items():
        write_manifest_split(manifest_dir, split_name, rows, video_dir, audio_placeholder)
    with (manifest_dir / "dict.wrd.txt").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(spm_dict.read_text(encoding="utf-8"))

    summary = {
        "dataset_root": str(dataset_root),
        "out_root": str(out_root),
        "source_root": str(source_root),
        "manifest_dir": str(manifest_dir),
        "video_dir": str(video_dir),
        "audio_placeholder": str(audio_placeholder),
        "spm_model": str(spm_model),
        "splits": {name: len(rows) for name, rows in split_rows.items()},
        "errors": len(errors),
        "labels_unique": len(Counter(row["label"] for row in all_rows)),
    }
    (manifest_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
