# 发布指南

项目通过 GitHub Release 发布到 PyPI。发布工作流会检查版本、运行测试、构建 wheel 和源码包，再使用 PyPI Trusted Publishing 上传。

## 首次配置

1. 登录 PyPI，打开 [账户 Publishing 页面](https://pypi.org/manage/account/publishing/)。
2. 在 **Add a new pending publisher → GitHub** 中填写：

   | 项目 | 填写内容 |
   | --- | --- |
   | PyPI Project Name | `cninvoice` |
   | Owner | `YuqiPlus` |
   | Repository name | `CN-Invoice` |
   | Workflow name | `publish.yml` |
   | Environment name | `pypi` |

3. 点击 **Add**。如果项目已经存在，在项目的 **Manage → Publishing** 页面添加同样的发布配置。
4. 在 GitHub 仓库的 **Settings → Environments** 中创建名为 `pypi` 的环境。

该方式不需要生成或保存 PyPI API token。首次上传成功后，PyPI 会自动创建项目；配置 pending publisher 本身不会占用包名。

## 发布一个版本

1. 确认 `pyproject.toml` 的 `project.version` 与 `src/cninvoice/__init__.py` 的 `__version__` 一致。
2. 将修改提交并推送到 GitHub。已经上传过 PyPI 的版本需要换用新版本号。
3. 打开 [创建 Release 页面](https://github.com/YuqiPlus/CN-Invoice/releases/new)。
4. 创建与版本对应的标签，例如版本 `0.1.0` 对应 `v0.1.0`；目标选择包含本次修改的提交。
5. 填写更新说明。`0.1.0` 是正式版，不勾选 **Set as a pre-release**，点击 **Publish release**。
6. 在仓库 **Actions → Publish to PyPI** 中查看测试、构建及发布结果。

需要预先验证时，可在 **Actions → Publish to PyPI → Run workflow** 手动运行测试和构建；手动运行不会上传 PyPI。

Release 标签必须严格等于 `v` 加包版本号。工作流只在 Release 正式发布时触发，普通推送和保存草稿不会上传 PyPI。

成功后，在新环境中验证：

```bash
python -m pip install "cninvoice==0.1.0"
python -c "import cninvoice; print(cninvoice.__version__)"
cninvoice --help
```

PyPI 不允许覆盖已上传的同名发行文件。后续发布应修改版本号并创建新标签。

## 本机手动发布

也可使用 Twine 从本机发布：

```bash
python -m pip install -e ".[dev,benchmark]"
python -m pytest
python tools/prepare_release.py
```

构建工具会检查包内文件、许可证元数据和项目说明。输出目录为 `release/<version>/`，已有目录不会覆盖。以 `0.1.0` 为例：

```bash
python -m twine upload --repository pypi --username __token__ \
  release/0.1.0/cninvoice-0.1.0-py3-none-any.whl \
  release/0.1.0/cninvoice-0.1.0.tar.gz
```

在密码提示中输入 PyPI API token。首次创建项目时使用具备创建权限的 token，项目创建后可改用项目级 token。不要把 token 放入仓库或命令参数。

如需先试发 TestPyPI，将 `--repository pypi` 改成 `--repository testpypi`，并使用 TestPyPI 的账号及 token。

## 模型文件

Python 发行包只包含库代码与文档。独立模型包可以作为 GitHub Release 附件或托管于模型仓库；其格式与构建命令见 [模型配置](https://github.com/YuqiPlus/CN-Invoice/blob/main/docs/MODELS.md)。

参考：[PyPI 首次可信发布](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)、[Python 官方 GitHub 发布指南](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)。
