# NEMO INFERENCE

This repository is an extension of [NVIDIA/NeMo](https://github.com/NVIDIA/NeMo).

As far as I know and tested, it only works on **NeMo ver. 1.18.0**.

`transcribe_speech.py` was edited to include the predicted confidence scores in the output manifest.
`custom_transcribe_utils.py` is an edited version of the original `transcribe_utils.py` to include the computed confidence scores.

## Download Model(s)
1. Download the model you want from [NVIDIA Catalog](https://catalog.ngc.nvidia.com/models). Recommended for testing: [STT En Conformer-CTC Small](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_conformer_ctc_small)

2. Create a directory `models` in the main directory and save the model there.

## Pulling the prebuilt image (development)
```
docker pull dleongsh/nemo-asr:1.18.0-inference
```

## Building the docker image (development)
```bash
cd nemo-inference
docker build -f build/asr.dev.Dockerfile -t dleongsh/nemo-asr:1.18.0-inference .
```

## Building the docker image (deployment)
```bash
cd nemo-inference
docker build -f build/asr.deploy.Dockerfile -t dleongsh/nemo-asr:1.18.0-inference .
```

## Run inference (development)

1. In `build/docker-compose.yaml`, mount your required local folders into the container.
```yaml
- /mnt/d/datasets:/datasets
- /mnt/d/pretrained_models:/pretrained_models 
```

2. Spin up the docker container.
```bash
cd build && docker-compose run --rm local bash
```

3. Run inference inside the docker container.
```bash
python3 transcribe_speech.py \
    model_path=models/stt_en_conformer_ctc_small.nemo \
    dataset_manifest=/datasets/test_manifest.json \
    compute_confidence=true
```
