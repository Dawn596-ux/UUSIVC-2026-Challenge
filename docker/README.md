# UUSIVC 2026 Docker Usage Guide

This guide explains how to build and run the inference for UUSIVC 2026 in a Docker environment with GPU support.

## 1. [Reference Examples](https://github.com/uusic2025/challenge/tree/main/docker#3-tag-your-image)

- Baseline source project: `UUSIVC-2026-Challenge`
- Loadable baseline Docker image: `uusivc2026-baseline-docker.tar`

[HERE](https://huggingface.co/datasets/WyattH/UUSIVC2026/tree/main)

## 2. Install Docker

First, install Docker Desktop (available for Windows, macOS, and Linux):

- Download: <https://docs.docker.com/get-docker/>

After installation, verify Docker is available:

```sh
docker --version
```

If it shows a version number, Docker is installed correctly.

## 3. Build the Docker Image

Assuming the project code and `Dockerfile` are in the same directory:

```sh
cd /path/to/project
docker build -f Dockerfile -t [image_name] .
```

Parameters:

- `-f Dockerfile` — specify the Dockerfile to use.
- `-t [image_name]` — assign an image name and optional tag.
- `.` — use the current project directory as the Docker build context.

Example:

```sh
docker build -f Dockerfile -t uusivc2026-submission:latest .
```

Docker image names should use lowercase letters.

## 4. Check the Docker Container

The submitted image must automatically start inference without an interactive terminal. We use the following container interface:

```sh
docker run --gpus all --rm \
  -v [/path/to/input]:/input:ro \
  -v [/path/to/output]:/output \
  [image_name]
```

Parameters:

- `--gpus all` — make the available NVIDIA GPUs visible inside the container.
- `--rm` — remove the container automatically after it stops.
- `/input:ro` — input package in read-only mode.
- `/output` — output directory.
- `[image_name]` — the submitted image name and tag.

The container contract is:

| Container path | Access | Purpose |
| --- | --- | --- |
| `/input` | Read-only | Test metadata and input files |
| `/output` | Read-write | Generated submission results |


Example:

```sh
docker run --gpus all --rm -v /path/to/data/Val/:/input/:ro -v /path/to/sample_result_submission/:/output repository:tag
```

## 5. Submit the Docker Image

After the image has been built and tested locally, export it as a Docker image archive:

```sh
docker save -o [file_name].tar [image_name]
```

Example:

```sh
docker save -o [teamname-uusivc2026-docker].tar repository:tag
```

Verify that the archive can be loaded:

```sh
docker load -i [teamname-uusivc2026-docker].tar
```

Large `docker save` and `docker load` operations may show no progress for several minutes. Wait for the command to finish before interrupting it.

Before submission, confirm that:

- The image contains all required code, dependencies, and weights.
- The image runs with `--gpus all`.
