# 本地模型配置

图片识别由两部分组成：字段检测器定位票面字段，OCR 读取文字。本地推理使用 ONNX Runtime CPU 后端；训练阶段使用 PyTorch。

## 目录结构

增值税字段模型目录：

```text
models/vat/
├── detector.onnx
└── model.json
```

`detector.onnx` 使用项目训练及导出脚本生成，架构为 `ssdlite320_mobilenet_v3_large`。`model.json` 记录字段顺序、输入尺寸和阈值；它必须与检测器一起使用，不能仅把任意 ONNX 文件改成相同文件名。

OCR 目录需要以下兼容 RapidOCR 的文件：

```text
models/ocr/
├── PP-OCRv6_det_small.onnx
├── ch_ppocr_mobile_v2.0_cls_mobile.onnx
└── PP-OCRv6_rec_small.onnx
```

安装 `cninvoice[vision]` 后可以加载这些模型；模型文件需要单独准备。

## 按票种加载

```python
from cninvoice import ReaderConfig

reader = ReaderConfig(
    recognizer="specialists",
    specialist_model_dirs={
        "vat_invoice": "./models/vat",
        # 取得对应模型后可增加以下票种：
        # "rail_einvoice": "./models/rail",
        # "ride_itinerary": "./models/ride",
        # "air_itinerary": "./models/air",
    },
    ocr_model_dir="./models/ocr",
).create_reader()
```

OCR 标题及版式规则先选择票种，再调用对应检测器。未配置的票种不会借用其他检测器。`ReaderConfig(document_profile="vat_invoice", ...)` 可在已知票种时显式指定；冲突的票面标题仍会产生复核提示。

## 训练和导出

安装开发及训练依赖后，用以下命令查看数据、检测器和导出脚本的参数：

```bash
python -m pip install -e ".[train,vision,benchmark]"
python training/build_specialists.py --help
python training/train_detector.py --help
python training/export_detector.py --help
```

训练脚本接收本地数据清单和初始权重，导出脚本从训练目录中的 `best.pt` 与 `model.json` 生成 `detector.onnx`，并检查 ONNX 与 PyTorch 输出的一致性。

## 打包便携模型

准备好字段模型目录、OCR 文件、模型说明及上游许可后：

```bash
python tools/build_model_bundle.py \
  --model-dir ./models/vat \
  --ocr-dir ./models/ocr \
  --ocr-license-dir ./models/ocr-licenses \
  --model-card ./models/MODEL_CARD.md \
  --output ./model-release
```

`ocr-license-dir` 包含 `LICENSE` 和 `MODEL_LICENSES.md`。工具将运行文件、模型说明、许可和 SHA-256 清单打包为 ZIP；不包含训练数据和检查点。当前打包工具用于增值税专用模型。

使用者解压后加载包含 `bundle.json` 的目录：

```python
from cninvoice import ReaderConfig

reader = ReaderConfig.from_bundle("./models/vat-bundle").create_reader()
invoice = reader.read("invoice.png")
```
