# UUSIVC 2026 测试阶段推理镜像
# 基础镜像对齐服务器 torch 2.6.0 + CUDA 12.4
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

LABEL maintainer="Dawn596-ux" \
      description="UUSIVC 2026 testing-phase inference image"

# 依赖：albumentations 先装（会连带升 numpy/scipy 到 2.x），再分步锁回（阶段 B 验证顺序）
RUN pip install --no-cache-dir albumentations==2.0.8 && \
    pip install --no-cache-dir "numpy<2" && \
    pip install --no-cache-dir "scipy<1.14" && \
    pip install --no-cache-dir pyyaml tqdm pillow matplotlib scikit-learn scikit-image

WORKDIR /app

# 代码
COPY predict.py /app/predict.py
COPY configs/ /app/configs/
COPY models/ /app/models/
COPY datasets/ /app/datasets/
COPY utils/ /app/utils/

# 权重：双 checkpoint（主 stage2_cls_aug2 rank1 用于 task1/2/5；辅 stage2_cls rank2 用于 task3/4）
COPY weights/best_stage2_cls_aug2_rank1.pth /app/checkpoint.pth
COPY weights/best_stage2_cls_rank2.pth /app/secondary_checkpoint.pth
COPY weights/swin_tiny_patch4_window7_224.pth /app/pretrained_ckpt/swin_tiny_patch4_window7_224.pth

# 入口脚本
COPY docker/run.sh /app/run.sh
RUN chmod +x /app/run.sh

# 非 root 用户（官方要求容器以非 root 运行）
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# 默认路径，可被环境变量覆盖
ENV INPUT_DIR=/app/input \
    OUTPUT_DIR=/app/output \
    DEVICE=cuda

ENTRYPOINT ["/app/run.sh"]
