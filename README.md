# CN-Invoice

中国发票与报销票据解析库：输入 PDF、OFD、XML 或图片，返回统一的 Python `Invoice` 对象。

优先解析文件中的结构化数据和文字，必要时接入本地视觉模型与 OCR。适合用于报销系统、发票归档和票据信息提取。

- 支持增值税发票、数电发票以及铁路、航空等报销票据的数据结构。
- 金额使用 `Decimal`，缺失字段返回 `None`，保留发票号码的前导零。
- 提供字段来源、金额一致性检查和待复核标记。
- 支持按票种配置本地专用模型，也可选用云端识别或 AI 复核。

## 安装

Python 3.10+。

从 GitHub 安装：

```bash
python -m pip install "git+https://github.com/YuqiPlus/CN-Invoice.git"
```

PyPI 发行版的安装命令：

```bash
python -m pip install cninvoice
```

本地图片识别需要额外安装 `vision` 依赖：

```bash
python -m pip install "cninvoice[vision]"
# 或从 GitHub 安装：
python -m pip install "cninvoice[vision] @ git+https://github.com/YuqiPlus/CN-Invoice.git"
```

## 快速开始

```python
from cninvoice import read_invoice

invoice = read_invoice("invoice.pdf")

print(invoice.kind)            # 票据类型
print(invoice.invoice_number)  # 发票号码
print(invoice.amount)          # 不含税金额
print(invoice.tax)             # 税额
print(invoice.total)           # 价税合计
print(invoice.needs_review)    # 是否需要人工核对

# 转为 JSON，金额序列化为字符串
print(invoice.model_dump_json(indent=2))
```

`read_invoice()` 接受文件路径、`bytes` 或二进制文件对象。PDF、OFD、XML 优先直接解析；扫描件和图片需要配置识图后端。

```python
from pathlib import Path
from cninvoice import read_invoice

invoice = read_invoice(Path("invoice.xml"))
invoice = read_invoice(Path("invoice.ofd").read_bytes())

with open("invoice.pdf", "rb") as file:
    invoice = read_invoice(file)
```

## 返回字段

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `kind` | `InvoiceKind` | 增值税发票、铁路客票、航空行程单等票据类型 |
| `invoice_number` | `str \| None` | 发票号码，保留前导零 |
| `invoice_code` | `str \| None` | 发票代码 |
| `issue_date` | `date \| None` | 开票日期 |
| `amount` | `Decimal \| None` | 不含税合计金额 |
| `tax` | `Decimal \| None` | 合计税额 |
| `total` | `Decimal \| None` | 价税合计或票据总金额 |
| `buyer` / `seller` | `Party` | 购销方名称、税号、地址及银行信息 |
| `items` | `list[LineItem]` | 明细项目 |
| `rail` / `air` | `RailJourney \| AirJourney \| None` | 铁路或航空行程信息 |
| `sources` | `dict[str, Evidence]` | 字段来源及可用的票面证据 |
| `issues` | `list[Issue]` | 缺项、冲突和校验提示 |
| `needs_review` | `bool` | 是否需要人工核对 |

字段是否有值取决于票面内容、文件格式和识别后端。未识别出的金额保持 `None`；票面明确为零才返回零。`amount`、`tax`、`total` 全部读取到时会检查相加关系，但不会通过计算补齐缺项。

## 本地图片识别

本地识别使用 ONNX 字段检测器与 OCR，支持 CPU 推理，不需要部署大语言模型。按票据类型选择一个专用检测器，增值税模型重点提取发票号码、不含税金额、税额和价税合计。

模型文件与 Python 包分开管理。**仓库目前提供推理及训练代码，尚无公开的预训练模型下载。** 使用图片识别前，需要准备兼容的模型包或自行训练并导出模型。

### 加载模型包

将模型包解压，传入包含 `bundle.json` 的目录：

```python
from cninvoice import ReaderConfig

reader = ReaderConfig.from_bundle("./models/vat-bundle").create_reader()
invoice = reader.read("invoice.png")

print(invoice.total)
print(invoice.needs_review)
```

`from_bundle()` 校验模型文件的大小与 SHA-256，使用包内的检测器及 OCR 文件。

### 直接指定模型目录

```python
from cninvoice import ReaderConfig

reader = ReaderConfig(
    recognizer="specialists",
    specialist_model_dirs={"vat_invoice": "./models/vat"},
    ocr_model_dir="./models/ocr",
).create_reader()

invoice = reader.read("invoice.jpg")
```

检测器目录包含 `detector.onnx` 和 `model.json`。票种配置项包括 `vat_invoice`、`rail_einvoice`、`ride_itinerary`、`air_itinerary`；每项需要对应的模型。OCR 模型文件名及模型包构建方式见 [模型配置](https://github.com/YuqiPlus/CN-Invoice/blob/main/docs/MODELS.md)。

视觉识别当前用于单张图片或单页文件。OCR／模型结果会标记为待复核，模糊、遮挡、红字及不同版式可能导致字段缺失。

## 命令行

```bash
cninvoice invoice.pdf
cninvoice invoice.xml
cninvoice invoice.png --vision-model ./models/vat --ocr-models ./models/ocr
```

输出为 JSON。更多选项见 `cninvoice --help`。

## 开发与训练

```bash
git clone https://github.com/YuqiPlus/CN-Invoice.git
cd CN-Invoice
python -m pip install -e ".[dev,benchmark]"
python -m pytest
```

训练依赖：

```bash
python -m pip install -e ".[train,vision,benchmark]"
```

`training/` 包含样本准备、可选教师模型标注、标注校验、票种拆分、检测器训练和 ONNX 导出工具。各脚本通过 `--help` 查看参数。数据处理工具的私有输出放在 `.private/` 下。

欢迎通过 [Issues](https://github.com/YuqiPlus/CN-Invoice/issues) 提交问题和通过 Pull Request 贡献改进。发行流程见 [发布指南](https://github.com/YuqiPlus/CN-Invoice/blob/main/docs/PUBLISHING.md)。

## 许可证

代码采用 [MIT License](https://github.com/YuqiPlus/CN-Invoice/blob/main/LICENSE)。使用第三方模型或数据时，请保留其各自的许可证与来源说明。
