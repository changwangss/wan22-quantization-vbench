# Wan2.2 quantization quality benchmarks

Reproduce selected VBench quality measurements for Wan2.2-A14B **BF16, MXFP8, and NVFP4**, using the same inputs and generation settings within each task. Inference uses the official [vLLM-Omni source at 7465c0f52bcb46a55f62c1b8613846cda30d0083](https://github.com/vllm-project/vllm-omni/tree/7465c0f52bcb46a55f62c1b8613846cda30d0083).

These are **selected, one-sample protocols**, not full standard VBench runs or official aggregate VBench scores. This repository contains input metadata, code, and compact results—not model weights, reference images, generated videos, or evaluator checkpoints. Follow the [upstream data terms](data/README.md) and each model's license.

| Task | Inputs per model | Output | Evaluation |
|---|---:|---|---|
| [Text-to-video](docs/t2v.md) | 165 prompts | 1280 × 720, 81 frames, 16 fps | Five metrics, each on all 165 videos together |
| [Image-to-video](docs/i2v.md) | 355 prompt/image pairs | 832 × 480, 81 frames, 16 fps | Five metrics, filtered by official dimension tags |

## Start here

1. Read [environment setup and its verification limits](docs/environment.md). The measured system was Linux with NVIDIA B200 GPUs; it used a Python overlay, not a container. **A clean installation has not been tested.**
2. Obtain the appropriate checkpoint locally. [Model IDs and quantization provenance](docs/quantization.md) link all six Hugging Face models. T2V MXFP8 used AutoRound **200 iterations**, as confirmed by the model owner. I2V MXFP8's card records RTN/zero iterations, but its original export history is not independently proven. Both recorded NVFP4 export recipes use RTN/zero iterations with OpenS2V calibration.
3. Use a separate run root for every model, precision, and task. Run commands from this repository's root in the configured environment.

For example, download a checkpoint with the Hugging Face CLI (substitute another
ID from the model table for a different task or precision):

```bash
hf download intel-ai/Wan2.2-I2V-A14B-Diffusers-NVFP4-AutoRound \
  --local-dir models/i2v-nvfp4
```

For repeatable future downloads, add `--revision FULL_HUB_COMMIT` and retain that
revision with your run. The historical evaluation did not capture immutable Hub
weight revisions; current remote `main` is not guaranteed byte-identical. The runner
records configuration/index hashes and weight file sizes, **not full weight-byte
hashes**: keep downloaded weights unchanged while generating or resuming a run.

### T2V: one GPU

```bash
export OMNI_SOURCE=/path/to/vllm-omni
export MODEL=/path/to/local/Wan2.2-T2V-A14B-Diffusers
python scripts/benchmark.py prepare --task t2v --root runs/t2v-bf16
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark.py generate \
  --task t2v --root runs/t2v-bf16 --model "$MODEL" --precision bf16 \
  --omni-source "$OMNI_SOURCE" --shard-index 0 --num-shards 1
python scripts/benchmark.py verify --task t2v --root runs/t2v-bf16
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark.py evaluate \
  --task t2v --root runs/t2v-bf16
```

### I2V: download images, then launch workers

The destination is external data storage; the downloader obtains the official cropped-image archive and extracts it. Pass the resulting `crop/16-9` directory, not the archive or its parent.

```bash
bash scripts/download_i2v.sh /path/to/external/vbench-i2v-data
export OMNI_SOURCE=/path/to/vllm-omni
export MODEL=/path/to/local/Wan2.2-I2V-A14B-Diffusers
bash scripts/launch_workers.sh i2v "$MODEL" bf16 runs/i2v-bf16 \
  "$OMNI_SOURCE" 0,1,2,5,7 /path/to/external/vbench-i2v-data/crop/16-9
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark.py evaluate \
  --task i2v --root runs/i2v-bf16
```

The launcher takes `task model precision root omni-source gpu-list [i2v-images]`: it prepares the run, starts indexed workers with logs, waits for all workers, and verifies outputs. Each worker owns a complete model on one visible GPU; this is independent input sharding, not tensor parallelism. Select GPUs with adequate free memory. For MXFP8 or NVFP4, change the checkpoint, `--precision`, and run root together.

## Run contract

- `prepare` snapshots only the selected inputs: exactly **165 T2V jobs** or **355 I2V jobs**. I2V also requires `--i2v-images /path/to/crop/16-9`. The snapshot is immutable; preparation does not bind a model or initialize an inference engine.
- `generate` binds and persists the generation configuration. A root belongs to **one model only**. Resume requires matching fingerprints and valid `ffprobe` checks, not just existing filenames.
- `--shard-index 0 --num-shards 1` is the single-worker case. For multiple workers, use distinct indices `0 … N-1` with the same `--num-shards N`, root, model, and configuration.
- `--limit N` is for smoke generation only; it does not shrink the official selected-job snapshot. Incomplete runs cannot be evaluated as full runs. Keep smoke roots separate.
- `verify` checks completion against the fixed input set and recorded configuration. Run it before scoring.
- `evaluate` uses all five task metrics by default; `--metrics` selects a subset. Subset scores are not a full five-metric result. Choose the evaluation GPU through `CUDA_VISIBLE_DEVICES`, not an `--gpu` argument.
- `environment --task t2v --root runs/t2v-bf16 --omni-source "$OMNI_SOURCE"` records/checks runtime evidence. Inspect source paths as well as versions; package metadata alone can identify the wrong Omni checkout.

## Observed results

Higher is better for these reported metrics. Values below are rounded to six decimals; [T2V JSON](results/t2v.json) and [I2V JSON](results/i2v.json) retain the original precision and protocol. These measurements do not establish speed, memory savings, or statistical significance.

### T2V · all 165 videos per metric

| Metric | BF16 | MXFP8 | NVFP4 |
|---|---:|---:|---:|
| Imaging quality | 0.721889 | 0.722244 | 0.719336 |
| Aesthetic quality | 0.622004 | 0.621066 | 0.609699 |
| Motion smoothness | 0.980194 | 0.980503 | 0.980903 |
| Overall consistency | 0.237381 | 0.238093 | 0.238401 |
| Subject consistency | 0.951550 | 0.952117 | 0.947457 |

### I2V · official per-metric filtering

| Metric | Videos | BF16 | MXFP8 | NVFP4 |
|---|---:|---:|---:|---:|
| I2V subject | 246 | 0.977584 | 0.975640 | 0.961735 |
| Subject consistency | 246 | 0.952603 | 0.951910 | 0.936307 |
| Background consistency | 109 | 0.969536 | 0.968830 | 0.961778 |
| Motion smoothness | 246 | 0.984598 | 0.985272 | 0.983227 |
| I2V background | 109 | 0.991074 | 0.989965 | 0.982622 |

## Documentation

- [Environment, upstream installation references, and sanitized package inventory](docs/environment.md)
- [T2V settings, metrics, and sharding](docs/t2v.md)
- [I2V images, settings, metrics, and historical five-GPU schedule](docs/i2v.md)
- [Quantization recipes and provenance limits](docs/quantization.md)
- [Data attribution and snapshot checksums](data/README.md)
- [Result provenance](results/README.md)
- [Official-example showcases (separate from benchmarks)](docs/showcases.md)
- [Evaluator checkpoints and download troubleshooting](docs/evaluator-checkpoints.md)

## Validation and licensing

```bash
python -S -m unittest discover -s tests -v
```

The CPU suite checks frozen input counts, sampling parameters, all recorded scores,
evaluation filtering, video/prompt mapping, cache invalidation, and resume guards.
It does not load GPU models or substitute for end-to-end generation. New portable
scripts are extracted from the measured workflow; see [validation status](docs/validation.md)
for what has actually been exercised, separately from the historical full results.

Repository code is Apache-2.0 licensed; third-party data/model/source terms remain
separate. See [LICENSE](LICENSE), [NOTICE](NOTICE), and [data attribution](data/README.md).

**中文快速指南：** 先阅读[环境说明](docs/environment.md)，准备本地模型及固定版本的官方 vLLM-Omni 源码；I2V 还需下载官方裁剪图片，并传入 `crop/16-9`。每个模型使用独立的运行目录，按 `prepare → generate → verify → evaluate` 执行，也可用多 GPU 启动脚本自动分片。`--limit` 仅用于冒烟测试，不能把未完成的视频集当作完整评测。这里是选定输入、每个输入一个视频的质量对比，不是完整标准 VBench 总分；当前未验证全新环境安装。