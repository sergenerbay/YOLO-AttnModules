
# 🧠 Attention-YOLO

This project focuses on enhancing the YOLO architecture (Ultralytics-supported versions such as YOLOv5, YOLOv8) by integrating various attention mechanisms to improve detection performance.

Traditional object detection architectures may benefit from embedded attention modules that help the network focus on more informative spatial and channel-wise features. This repository experiments with several popular attention modules within the YOLOv8 model.

---

## ✅ Attention Modules Checklist

- ✅ **SKAttention** (implemented and tested)  
**  Selective Kernel Networks (SKNet) dynamically adjust receptive fields using multi-scale inputs, enabling effective multi-scale object capture with low complexity. [Paper](https://arxiv.org/abs/1903.06586)**


- ✅ **CBAM** (implemented and tested)
- ✅ **PSA** (implemented and tested)
- ✅ **SimAM** (implemented and tested)
- ⏳ **GAM** (planned)
- ⏳ **SE** (planned)

---

## ⚙️ Example Configuration Block with SKAttention

```yaml
- [-1, 1, SKAttention, [1024, [3, 5, 7], 16]]
```

- `1024` — Number of input/output channels  
- `[3, 5, 7]` — Multi-scale kernel sizes for spatial attention  
- `16` — Channel reduction ratio in the attention block  

---

## ⚙️ Example Configuration Block with PSA

```yaml
- [-1, 1, PSAPlug, [1024,4]]
```

- `1024` — Number of input/output channels  
- `4` — Channel reduction ratio in the attention block

---

## ⚙️ Example Configuration Block with CBAM

```yaml
- [-1, 1, CBAM, [256,3]]
```

- `256` — Number of input/output channels  
- `3` — Size of the convolutional kernel for spatial attention

---

## 🧪 Sample Training Script

```python
from ultralytics import YOLO

model = YOLO("yolov8s-sk.yaml")
model.train(
    data="your-dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=32,
    name="yolov8s-sk"
)
```

---

## 📌 Future Plans

- Benchmark each attention module on the same dataset.
- Analyze the contribution of each attention block during training.
- Add performance comparison table (mAP, precision, recall) across modules.
