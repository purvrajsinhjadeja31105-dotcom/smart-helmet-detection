# Model weights

| File | Model | Trained on | Test mAP50 |
|---|---|---|---|
| `helmet/helmet_best.pt` | Model A: heads as `helmet` / `no_helmet` (YOLO26-m, 960 px) | 5,593 images, fine-tuned from COCO | 0.891 |
| `plate/plate_crop_best.pt` | Model B: `number_plate` on motorcycle crops (YOLO26-m, 640 px) | 4,708 images, fine-tuned from COCO | 0.923 |

The COCO-pretrained detector for motorcycles and persons (`coco/yolo26m.pt`) is **not** stored here: Ultralytics downloads it automatically on the first run.

ONNX exports can be recreated with:

```bash
yolo export model=models/helmet/helmet_best.pt format=onnx
```

To retrain, see [`phase1/README.md`](../phase1/README.md).
