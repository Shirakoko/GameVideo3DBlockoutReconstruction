# GameWM 阅读笔记网页

在线阅读：https://shirakoko.github.io/GameVideo3DBlockoutReconstruction/gamewm/

本目录保存用户提供的 HTML 阅读笔记及其引用的 8 张图片。正文与原有样式保持不变；发布版使用相对图片路径、简短资源文件名和本地字体回退，并增加站点导航和中文语言标记。原始桌面文件未修改。

## 自动部署

本仓库已经启用 GitHub Pages，发布源为 `main` 分支的 `/docs`。提交并推送本目录后，GitHub 的现有 `pages build and deployment` 自动构建、发布网页，无需另加 GitHub Actions 工作流或修改发布源。

站点首页 `docs/index.html` 与仓库 README 均提供阅读笔记入口。此页面只引用 `assets/` 内的图片，不引用实验视频、场景模型、验证记录或任何本机绝对路径。

## 更新

编辑 `index.html` 并将新图片放入 `assets/`，保持相对路径。提交前运行：

```powershell
python -m unittest discover -s tests -p test_gamewm_pages.py -v
```
