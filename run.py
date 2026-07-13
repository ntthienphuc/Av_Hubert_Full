#!/usr/bin/env python3
"""Single command-line entry point for the Vietnamese AV-HuBERT workflow."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_PRETRAINED_URL = "https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/clean-pretrain/large_vox_iter5.pt"


def run(command: list[str], cwd: Path = ROOT) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def require(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"Missing {description}: {path}")
    return path


def fairseq_train() -> str:
    executable = shutil.which("fairseq-hydra-train")
    if not executable:
        raise SystemExit("fairseq-hydra-train is unavailable. Run: python run.py setup")
    return executable


def download(url: str, output: Path) -> None:
    from urllib.request import urlretrieve

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        print(f"Already exists: {output}")
        return
    print(f"Downloading {url} -> {output}")
    urlretrieve(url, output)


def cmd_setup(args: argparse.Namespace) -> None:
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    torch_args = [sys.executable, "-m", "pip", "install", "torch==1.13.1", "torchvision==0.14.1"]
    if args.cuda == "cu117":
        torch_args += ["--extra-index-url", "https://download.pytorch.org/whl/cu117"]
    run(torch_args)
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    run([sys.executable, "-m", "pip", "install", "-e", "./fairseq"])
    run([sys.executable, "verify_env.py"])


def cmd_download(args: argparse.Namespace) -> None:
    download(args.url, args.output.expanduser().resolve())


def cmd_prepare(args: argparse.Namespace) -> None:
    require(args.dataset_root, "VLips dataset root")
    run([
        sys.executable, "tools/prepare_vlips.py",
        "--dataset-root", str(args.dataset_root), "--out-root", str(args.output),
        "--vocab-size", str(args.vocab_size), "--fps", str(args.fps),
    ] + (["--max-items", str(args.max_items)] if args.max_items else []))


def cmd_crop(args: argparse.Namespace) -> None:
    metadata = require(args.metadata, "metadata.csv; run prepare first")
    common = [
        sys.executable, "tools/crop_mouth.py", "--metadata", str(metadata),
        "--video-dir", str(args.video_dir), "--out-size", str(args.out_size),
    ]
    if args.overwrite:
        common.append("--overwrite")
    if args.workers == 1:
        run(common)
        return
    processes = []
    for rank in range(args.workers):
        command = common + ["--rank", str(rank), "--nshard", str(args.workers)]
        print("+", subprocess.list2cmdline(command), flush=True)
        processes.append(subprocess.Popen(command, cwd=ROOT))
    codes = [process.wait() for process in processes]
    if any(codes):
        raise SystemExit(f"Crop workers failed with exit codes: {codes}")


def cmd_finetune(args: argparse.Namespace) -> None:
    data = require(args.data, "prepared manifest directory")
    checkpoint = require(args.checkpoint, "pretrained checkpoint")
    tokenizer = require(args.tokenizer or data / "spm_unigram1000.model", "SentencePiece model")
    for name in ("train.tsv", "train.wrd", "valid.tsv", "valid.wrd", "dict.wrd.txt"):
        require(data / name, name)
    args.output.mkdir(parents=True, exist_ok=True)
    decay = max(1, args.max_update - args.warmup_steps)
    overrides = [
        f"task.data={data.resolve()}", f"task.label_dir={data.resolve()}",
        f"task.tokenizer_bpe_model={tokenizer}", f"model.w2v_path={checkpoint}",
        f"hydra.run.dir={args.output.resolve()}", f"common.user_dir={(ROOT / 'avhubert').resolve()}",
        f"common.fp16={str(args.fp16).lower()}", f"optimization.max_update={args.max_update}",
        f"optimization.lr=[{args.lr}]", f"optimization.clip_norm={args.clip_norm}",
        f"dataset.max_tokens={args.max_tokens}", f"dataset.num_workers={args.workers}",
        f"model.freeze_finetune_updates={args.freeze_updates}",
        f"lr_scheduler.warmup_steps={args.warmup_steps}", f"lr_scheduler.decay_steps={decay}",
        f"checkpoint.save_interval_updates={args.save_every}",
    ] + args.override
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", HYDRA_FULL_ERROR="1")
    command = [fairseq_train(), "--config-dir", str(ROOT / "avhubert/conf/finetune"), "--config-name", "base_vlips_1gpu"] + overrides
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT / "avhubert", env=env, check=True)


def cmd_pretrain(args: argparse.Namespace) -> None:
    data = require(args.data, "pretraining manifest directory")
    labels = require(args.labels, "pseudo-label directory containing .km files")
    args.output.mkdir(parents=True, exist_ok=True)
    overrides = [
        f"task.data={data}", f"task.label_dir={labels}", f"hydra.run.dir={args.output.resolve()}",
        f"common.user_dir={(ROOT / 'avhubert').resolve()}", f"model.label_rate={args.label_rate}",
    ] + args.override
    run([fairseq_train(), "--config-dir", str(args.config_dir.resolve()), "--config-name", args.config_name] + overrides, ROOT / "avhubert")


def cmd_infer(args: argparse.Namespace) -> None:
    data = require(args.data, "manifest directory")
    checkpoint = require(args.checkpoint, "fine-tuned checkpoint")
    require(data / f"{args.subset}.tsv", f"{args.subset}.tsv")
    require(data / f"{args.subset}.wrd", f"{args.subset}.wrd")
    args.output.mkdir(parents=True, exist_ok=True)
    run([
        sys.executable, "avhubert/infer_s2s.py", "--config-dir", "avhubert/conf", "--config-name", "s2s_decode",
        f"common.user_dir={(ROOT / 'avhubert').resolve()}", f"common.fp16={str(args.fp16).lower()}",
        f"common_eval.path={checkpoint}", f"common_eval.results_path={args.output.resolve()}",
        f"dataset.gen_subset={args.subset}", f"dataset.max_tokens={args.max_tokens}",
        f"generation.beam={args.beam}", f"override.data={data}", f"override.label_dir={data}",
        "override.modalities=[video]",
    ])


def cmd_all(args: argparse.Namespace) -> None:
    prepared = args.work_dir / "prepared/vlips"
    checkpoint = args.pretrained
    if not checkpoint.exists():
        download(DEFAULT_PRETRAINED_URL, checkpoint)
    cmd_prepare(argparse.Namespace(dataset_root=args.dataset_root, output=prepared, vocab_size=1000, fps=args.fps, max_items=0))
    cmd_crop(argparse.Namespace(metadata=prepared / "manifest/metadata.csv", video_dir=prepared / "video", out_size=96, workers=args.crop_workers, overwrite=False))
    cmd_finetune(argparse.Namespace(
        data=prepared / "manifest", checkpoint=checkpoint, tokenizer=None,
        output=args.work_dir / "experiments/vlips_finetune", max_update=args.max_update,
        max_tokens=args.max_tokens, workers=args.data_workers, lr=args.lr, warmup_steps=args.warmup_steps,
        freeze_updates=args.freeze_updates, save_every=2000, clip_norm=1.0, fp16=args.fp16, override=[]))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AV-HuBERT Vietnamese lip-reading workflow")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("setup", help="install the reproducible Python environment"); s.add_argument("--cuda", choices=["cpu", "cu117"], default="cu117"); s.set_defaults(func=cmd_setup)
    s = sub.add_parser("download", help="download the Meta pretrained checkpoint"); s.add_argument("--url", default=DEFAULT_PRETRAINED_URL); s.add_argument("--output", type=Path, default=Path("checkpoints/pretrained/large_vox_iter5.pt")); s.set_defaults(func=cmd_download)
    s = sub.add_parser("prepare", help="build manifests and Vietnamese SentencePiece files"); s.add_argument("--dataset-root", type=Path, required=True); s.add_argument("--output", type=Path, default=Path("work/prepared/vlips")); s.add_argument("--vocab-size", type=int, default=1000); s.add_argument("--fps", type=float, default=29.0); s.add_argument("--max-items", type=int, default=0); s.set_defaults(func=cmd_prepare)
    s = sub.add_parser("crop", help="crop 96x96 mouth videos with MediaPipe"); s.add_argument("--metadata", type=Path, default=Path("work/prepared/vlips/manifest/metadata.csv")); s.add_argument("--video-dir", type=Path, default=Path("work/prepared/vlips/video")); s.add_argument("--workers", type=int, default=4); s.add_argument("--out-size", type=int, default=96); s.add_argument("--overwrite", action="store_true"); s.set_defaults(func=cmd_crop)
    s = sub.add_parser("finetune", help="fine-tune AV-HuBERT for Vietnamese VSR"); s.add_argument("--data", type=Path, default=Path("work/prepared/vlips/manifest")); s.add_argument("--checkpoint", type=Path, required=True); s.add_argument("--tokenizer", type=Path); s.add_argument("--output", type=Path, default=Path("work/experiments/vlips_finetune")); s.add_argument("--max-update", type=int, default=50000); s.add_argument("--max-tokens", type=int, default=500); s.add_argument("--workers", type=int, default=4); s.add_argument("--lr", type=float, default=0.0002); s.add_argument("--warmup-steps", type=int, default=5000); s.add_argument("--freeze-updates", type=int, default=4000); s.add_argument("--save-every", type=int, default=2000); s.add_argument("--clip-norm", type=float, default=1.0); s.add_argument("--fp16", action="store_true"); s.add_argument("--override", action="append", default=[]); s.set_defaults(func=cmd_finetune)
    s = sub.add_parser("pretrain", help="pretrain from manifests and frame-aligned .km pseudo-labels"); s.add_argument("--data", type=Path, required=True); s.add_argument("--labels", type=Path, required=True); s.add_argument("--config-dir", type=Path, required=True); s.add_argument("--config-name", required=True); s.add_argument("--output", type=Path, default=Path("work/experiments/pretrain")); s.add_argument("--label-rate", type=int, default=100); s.add_argument("--override", action="append", default=[]); s.set_defaults(func=cmd_pretrain)
    s = sub.add_parser("infer", help="decode valid/test and compute WER"); s.add_argument("--data", type=Path, default=Path("work/prepared/vlips/manifest")); s.add_argument("--checkpoint", type=Path, required=True); s.add_argument("--subset", choices=["valid", "test"], default="test"); s.add_argument("--output", type=Path, default=Path("work/experiments/decode_test")); s.add_argument("--beam", type=int, default=10); s.add_argument("--max-tokens", type=int, default=900); s.add_argument("--fp16", action="store_true"); s.set_defaults(func=cmd_infer)
    s = sub.add_parser("all", help="download, prepare, crop, and fine-tune end to end"); s.add_argument("--dataset-root", type=Path, required=True); s.add_argument("--work-dir", type=Path, default=Path("work")); s.add_argument("--pretrained", type=Path, default=Path("checkpoints/pretrained/large_vox_iter5.pt")); s.add_argument("--crop-workers", type=int, default=4); s.add_argument("--data-workers", type=int, default=4); s.add_argument("--fps", type=float, default=29.0); s.add_argument("--max-update", type=int, default=50000); s.add_argument("--max-tokens", type=int, default=500); s.add_argument("--lr", type=float, default=0.0002); s.add_argument("--warmup-steps", type=int, default=5000); s.add_argument("--freeze-updates", type=int, default=4000); s.add_argument("--fp16", action="store_true"); s.set_defaults(func=cmd_all)
    return p


if __name__ == "__main__":
    ns = parser().parse_args()
    ns.func(ns)
