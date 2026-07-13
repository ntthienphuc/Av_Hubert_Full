# AV-HuBERT Full — Vietnamese Lip Reading

Clean, reproducible AV-HuBERT workflow for video-only Vietnamese speech recognition (VSR): dataset preparation, MediaPipe mouth cropping, AV-HuBERT pretraining, Vietnamese fine-tuning, and beam-search inference.

This repository bundles the required AV-HuBERT and Fairseq source. All user-facing operations go through one cross-platform entry point:

```bash
python run.py <command> [options]
```

## Available models

| Model | Purpose | Link |
|---|---|---|
| `large_vox_iter5.pt` | Official Meta AV-HuBERT initialization checkpoint | [Download from Meta](https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/clean-pretrain/large_vox_iter5.pt) |
| Vietnamese fine-tuned model | Completed Vietnamese VLips model used for the reported evaluation | [Download from Google Drive](https://drive.google.com/file/d/1MJy-H60cCKOXHRD6lC-6diI9-zyhCQbn/view?usp=drive_link) |

Model weights are deliberately not committed to Git. Put downloaded files under `checkpoints/`; that directory is ignored by Git.

## Reported Vietnamese run

The packaged experiment records produced the following measurements. These are historical results, not metrics recomputed automatically during repository setup.

| Evaluation | Result |
|---|---:|
| Best validation accuracy | 29.574% |
| Best validation perplexity | 22.91 |
| Test WER (`checkpoint_best.pt`) | 38.4492% |
| Best update | 48,856 / 50,000 |

Lower WER is better. The reported test WER is the standard test-set result decoded from `checkpoint_best.pt`.

```text
WER (%) — lower is better
Test — checkpoint_best.pt   38.45 |██████████████████████████████████████
```

The stable recipe used video-only input, 96×96 MediaPipe mouth crops, a Vietnamese unigram SentencePiece vocabulary of 1,000 pieces, learning rate `2e-4`, 5,000 warm-up steps, 4,000 frozen fine-tuning updates, gradient clipping `1.0`, and 50,000 total updates.

## Requirements

- Python 3.9 (recommended; the bundled Fairseq revision is legacy code)
- NVIDIA GPU for practical training
- A CUDA-compatible PyTorch environment (the setup command defaults to CUDA 11.7 wheels)
- FFmpeg available on `PATH`
- Windows, Linux, or WSL
- Enough disk space for raw videos, cropped videos, checkpoints, and logs

Create an isolated environment first.

Windows PowerShell:

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
python run.py setup --cuda cu117
```

Linux/WSL:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python run.py setup --cuda cu117
```

For a CPU-only environment check:

```bash
python run.py setup --cuda cpu
```

CPU training is technically possible but not practical. Verify an existing environment with `python verify_env.py`.

## Dataset layout

The prepared VLips adapter expects this structure:

```text
VLips/
├── split/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
└── data_output/
    └── Vlips_v1_video_non_audio/
        ├── path/to/sample.mp4
        └── path/to/sample.txt
```

Each split file contains paths relative to the dataset root. Each `.txt` file contains the normalized transcript on the first non-empty line. Optional following rows may contain `word start end` timings.

## Run each step

### 1. Download the official initialization checkpoint

```bash
python run.py download
```

Default output:

```text
checkpoints/pretrained/large_vox_iter5.pt
```

### 2. Prepare manifests and Vietnamese tokenizer

```bash
python run.py prepare --dataset-root D:/data/VLips
```

This creates `train/valid/test.tsv`, `.wrd` transcripts, `dict.wrd.txt`, a 1,000-piece Vietnamese SentencePiece model, metadata, and an error report under `work/prepared/vlips/manifest`.

Use `--max-items 20` for a small preparation test.

### 3. Crop mouth regions

```bash
python run.py crop --workers 8
```

MediaPipe Face Mesh tracks lip landmarks, applies temporal box smoothing, and writes 96×96 MP4 crops while preserving relative sample IDs. Failures and fallback-frame statistics are stored in `work/prepared/vlips/crop_logs`.

### 4. Fine-tune for Vietnamese lip reading

```bash
python run.py finetune \
  --checkpoint checkpoints/pretrained/large_vox_iter5.pt \
  --max-update 50000
```

On PowerShell, use a backtick for multiline commands or put the command on one line. The best checkpoint is expected at:

```text
work/experiments/vlips_finetune/checkpoints/checkpoint_best.pt
```

Mixed precision is opt-in with `--fp16`. The stable Windows run used FP32 because it was more reliable with this legacy stack.

### 5. Inference and WER

```bash
python run.py infer \
  --checkpoint work/experiments/vlips_finetune/checkpoints/checkpoint_best.pt \
  --subset test \
  --beam 10
```

Use `--subset valid` for validation decoding. Hypotheses, references, and WER outputs are written below `work/experiments/decode_test` by the upstream decoder.

## One command from raw VLips to fine-tuned model

Environment setup remains a separate one-time step. After setup:

```bash
python run.py all --dataset-root D:/data/VLips --crop-workers 8
```

The `all` command downloads the Meta checkpoint when absent, prepares manifests/tokenizer, crops every split, and starts Vietnamese fine-tuning. It is restart-friendly: the downloader and cropper skip existing outputs.

Inference is intentionally separate so you can choose the best/final checkpoint and the evaluation subset explicitly.

## Pretraining from scratch

AV-HuBERT pretraining is different from supervised Vietnamese fine-tuning. It requires frame-aligned `.km` pseudo-labels generated through the upstream clustering pipeline. Once manifests, labels, and an appropriate Hydra config exist:

```bash
python run.py pretrain \
  --data D:/data/avhubert_manifest \
  --labels D:/data/avhubert_labels \
  --config-dir avhubert/conf/pretrain \
  --config-name large_vox_iter5
```

`large_vox_iter5.yaml` is included, but its required labels/features are not. Config choice depends on the pretraining stage and modalities you choose. See [`avhubert/clustering`](avhubert/clustering) and [`avhubert/preparation`](avhubert/preparation) for upstream feature clustering and corpus preparation. For most Vietnamese VSR use cases, start from Meta's pretrained checkpoint and run `finetune`.

## CLI reference

```bash
python run.py --help
python run.py setup --help
python run.py prepare --help
python run.py crop --help
python run.py pretrain --help
python run.py finetune --help
python run.py infer --help
python run.py all --help
```

Hydra settings not exposed as named fine-tuning options can be appended repeatedly:

```bash
python run.py finetune \
  --checkpoint checkpoints/pretrained/large_vox_iter5.pt \
  --override dataset.update_freq='[2]' \
  --override checkpoint.patience=8
```

## Repository layout

```text
.
├── run.py                  # only user-facing workflow entry point
├── verify_env.py           # environment/import verification
├── requirements.txt        # pinned non-PyTorch dependencies
├── tools/
│   ├── prepare_vlips.py    # VLips manifests + SentencePiece
│   └── crop_mouth.py       # MediaPipe mouth ROI extraction
├── avhubert/               # AV-HuBERT model, tasks, configs, decoder
└── fairseq/                # bundled compatible Fairseq revision
```

## Reproducibility notes

- Raw datasets, prepared videos, caches, logs, and model weights are excluded from Git.
- Preparation writes absolute video paths into Fairseq manifests. Re-run `prepare` after moving the dataset/work directory to another machine.
- The preparation adapter estimates frame counts from transcript timing data. Check `summary.json` and `errors.json` before a long run.
- Mouth-crop fallback frames are reported. Inspect high-fallback samples before training.
- Exact results depend on GPU, PyTorch/CUDA versions, dataset revision, and random seed.
- The upstream AV-HuBERT code and license are retained. Review `LICENSE` before redistribution or commercial use.

## Credits

Based on Meta Research's [AV-HuBERT](https://github.com/facebookresearch/av_hubert) and its compatible Fairseq code. If you use this repository in research, cite the original AV-HuBERT and robust audio-visual speech recognition papers listed by the upstream project.
