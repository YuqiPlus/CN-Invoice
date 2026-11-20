# cninvoice

将中国发票文件读取为 Python `Invoice` 对象。优先读取 PDF、OFD、XML 中的文字和结构化数据；图片或读取不完整的文件可使用本地视觉模型与 OCR。

适用于自动提取后由用户核对的报销流程。金额使用 `Decimal`，识别不到返回 `None`，不把空值当作零，不通过加减补出缺失金额。

## 安装

需要 Python 3.10 或更新版本。当前版本为 `0.1.0a11`，尚未上传 PyPI。现在可从源码安装：

```bash
git clone https://github.com/YuqiPlus/CN-Invoice.git
cd CN-Invoice
python -m pip install .
# 需要本地图片识别依赖时：python -m pip install ".[vision]"
```

发布到 PyPI 后，可以使用：

```bash
python -m pip install --pre cninvoice
```

基础安装支持直接读取文件；识别图片还需安装本地推理依赖，并另行取得模型文件：

```bash
python -m pip install --pre "cninvoice[vision]"
```

代码包不包含模型、真实发票样本或云端密钥。不会自动联网下载模型或上传发票。

## 直接读取 PDF、OFD 或 XML

```python
from cninvoice import read_invoice

invoice = read_invoice("invoice.pdf")
print(invoice.invoice_number)
print(invoice.amount)         # 不含税合计，Decimal 或 None
print(invoice.tax)            # 税额，Decimal 或 None
print(invoice.total)          # 价税合计，Decimal 或 None
print(invoice.needs_review)
print(invoice.model_dump_json(indent=2))
```

支持文件路径、字节串和二进制文件对象。仅使用基础接口时，图片不会自动识别；结果会带上缺少识图后端的提示。

## 使用本地模型识别图片

当前仓库不提供预训练模型下载；图片识别需要自行准备兼容模型。取得模型包后，将其解压，找到包含 `bundle.json` 的目录，然后：

```python
from cninvoice import ReaderConfig

reader = ReaderConfig.from_bundle("./cninvoice-vat-models-0.1.0a11").create_reader()
invoice = reader.read("invoice.png")
print(invoice.total)
print(invoice.needs_review)
```

模型包加载时校验文件大小与 SHA-256。运行使用 CPU，不需要 GPU、Torch 或大语言模型；完整模型包包含字段检测器及所需 OCR 文件。`ReaderConfig.from_bundle()` 不会下载或解压文件。也可通过 `specialist_model_dirs` 和 `ocr_model_dir` 手动指定本地模型目录。

增值税模型当前重点读取价税合计、不含税金额、税额和发票号码。模型包包含哪些票种，就启用哪些专用模型；未提供对应模型的票种会提示缺少能力，不自动请求云端服务。

## 结果与限制

- `amount`、`tax`、`total` 均允许为空。票面明确印着 `0.00` 才能得到零元；星号和空白不自动变为零。
- 三个金额均读取到时检查相加关系；缺项、冲突及校验失败会记录在 `invoice.issues`。
- OCR／模型结果保持待复核状态。字段证据可在 `invoice.sources` 中查看。
- 视觉后端目前要求一张图片或单页文件；多张不同发票应拆分处理。
- 现有实验不能保证跨行业、红字发票和严重模糊图片的全部字段都可读出。`Invoice` 对象的其他字段也可能为空。
- AI 复核及云服务是显式可选配置，不是本地识别的必需依赖。

## 开发与训练

```bash
python -m pip install -e ".[dev,benchmark]"
python -m pytest
```

训练另需 `python -m pip install -e ".[train,vision,benchmark]"`。`training/` 提供样本收集、可选云端标注、标注校验、按票种拆分、检测器训练及 ONNX 导出工具；各脚本使用 `--help` 查看参数。训练数据和云端配置由使用者自行准备，不随仓库提供。数据处理工具要求将私有输出放在 `.private/` 下。

`tools/build_model_bundle.py` 用于打包本地推理文件，`tools/prepare_release.py` 用于构建并检查 Python 发行包。详细发布步骤见 [docs/PUBLISHING.md](https://github.com/YuqiPlus/CN-Invoice/blob/main/docs/PUBLISHING.md)。

## 许可证

cninvoice 的 Python 代码采用 MIT 许可证。独立模型、上游 OCR 模型和训练数据适用各自的发布说明；代码的 MIT 许可证不会自动替代这些文件的许可。
