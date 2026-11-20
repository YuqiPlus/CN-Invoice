# PyPI 发布流程

代码采用 MIT 许可证，包名暂定 `cninvoice`，当前预发布版本为 `0.1.0a11`。2026-11-20 查询 PyPI JSON 接口未发现同名公开项目；这不代表包名已被保留，首次上传时仍以 PyPI 的结果为准。

## 用户如何使用

包发布到 PyPI 后：

```bash
python -m pip install --pre "cninvoice[vision]"
```

另行下载并解压模型包，再用：

```python
from cninvoice import ReaderConfig

reader = ReaderConfig.from_bundle("./cninvoice-vat-models-0.1.0a11").create_reader()
invoice = reader.read("invoice.png")
print(invoice.total)
```

代码包与模型包独立：基础代码可直接读取 PDF／OFD／XML；图片识别还需本地模型。`from_bundle()` 校验模型文件，不自动联网，也不上传发票。

## 本地准备发行文件

```bash
python -m pip install build "twine>=7,<8"
python -m pip install -e ".[dev]"
python -m pytest
python tools/prepare_release.py
```

准备工具只构建和检查，不会上传。输出到 `release/0.1.0a11/`：

- `cninvoice-0.1.0a11-py3-none-any.whl`
- `cninvoice-0.1.0a11.tar.gz`
- `SHA256SUMS` 和 `release.json`

工具会校验 MIT 元数据、许可证文件、PyPI 项目说明的渲染，以及发行包中未包含私有目录、模型权重或 YAML 配置。模型应另做真实读取测试。已有版本目录不会被覆盖；修改代码或发行内容后应递增版本号，重新构建。

## 上传前配置发布身份

需要自己的 PyPI 账号，完成邮箱验证及账号要求的身份验证。源码仓库为 [YuqiPlus/CN-Invoice](https://github.com/YuqiPlus/CN-Invoice)。PyPI 发布身份需要在发布者自己的环境单独配置，仓库不保存凭据；当前版本尚未上传 PyPI。

若使用 GitHub 等受支持平台发布，优先按 [PyPI Trusted Publishing 文档](https://docs.pypi.org/publishing/publishing-to-pypi/) 配置发布身份。首次建项目可使用 Pending Publisher，但该配置本身不会保留包名。没有 CI 时也可以在本机用 PyPI API token 上传；token 不写进项目文件，不放进命令参数或聊天记录。

## 先传 TestPyPI，再传 PyPI

TestPyPI 和 PyPI 的账号、token 分别配置。只上传当前版本的两个具体文件，不要使用包含所有历史版本的 `dist/*`。

```bash
python -m twine upload --repository testpypi --username __token__ \
  release/0.1.0a11/cninvoice-0.1.0a11-py3-none-any.whl \
  release/0.1.0a11/cninvoice-0.1.0a11.tar.gz
```

在密码提示中输入 TestPyPI token。检查 TestPyPI 项目说明及文件后，正式上传同一份已验证文件：

```bash
python -m twine upload --repository pypi --username __token__ \
  release/0.1.0a11/cninvoice-0.1.0a11-py3-none-any.whl \
  release/0.1.0a11/cninvoice-0.1.0a11.tar.gz
```

正式发布使用 PyPI token。发布后在另一个环境运行 `python -m pip install --pre "cninvoice[vision]"`，再分别验证结构化文件和模型图片读取。PyPI 不允许用新文件覆盖同一个发行文件名；后续修改使用新版本号。

## 模型文件的分发

Python 代码的 MIT 授权不自动覆盖模型文件。模型包的 `MODEL_CARD.md` 应分别说明自训练字段模型与上游 OCR 文件的来源、用途、限制和许可；上游 OCR 的许可证及来源说明随包保留。当前字段模型的公开许可和模型下载地址尚未确定，不把准备好的本地模型归档当作已经公开发布。

可用下列工具从明确指定的本地文件构建独立模型包：

```bash
python tools/build_model_bundle.py \
  --model-dir /path/to/vat-model \
  --ocr-dir /path/to/ocr-models \
  --ocr-license-dir /path/to/ocr-license-files \
  --model-card /path/to/MODEL_CARD.md \
  --output /path/to/model-release
```

仅打包运行需要的五个 ONNX／模型元数据文件、模型说明和上游许可，不打包训练图、教师回复、密钥或训练检查点。模型归档可在确定公开说明后放到项目 Release 或模型托管站点，再把真实下载地址补充到项目说明。构建工具不会自动上传。

参考：[Python 官方打包教程](https://packaging.python.org/en/latest/tutorials/packaging-projects/)、[Twine 使用说明](https://twine.readthedocs.io/en/stable/)。
