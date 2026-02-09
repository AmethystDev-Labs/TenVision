# 🐧 TenVision

这是一个基于 **OpenCV** 的实验性项目，专门用于识别并计算腾讯云风格的“图形点选”验证码位置。

## ✨ 核心特性

* **OpenCV 驱动**：纯计算机视觉逻辑，不依赖大型深度学习模型，主打一个轻量。
* **形状识别**：针对特定几何特征进行特征点提取与匹配。
* **GitHub Action 闭环**：利用 GitHub 免费算力，实现“提交-识别-回推”的自动化体验。

---

## 🎮 怎么玩？(互动体验)

不需要在本地配置 Python 环境，你直接在 GitHub 就能体验这个算法：

1. **提交 Issue**：点击 [New Issue](https://github.com/AmethystDev-Labs/TenVision/issues/new)。
2. **上传图片**：Issue标题随意，内容可以是一张或多张腾讯云点选验证码
3. **自动回帖**：算法处理完成后，机器人会自动在你的 Issue 下回复一张**标记了坐标点**的图片。

> [腾讯云图形点选验证码](https://cloud.tencent.com/product/captcha)
> <img width="2160" height="1226" alt="image" src="https://github.com/user-attachments/assets/417cfbde-bdc2-42b9-939b-48d93ecd7d5e" />

| 正确截图 | 错误截图 |
| --- | --- |
| <img width="451" height="451" alt="image" src="https://github.com/user-attachments/assets/61a84683-37e4-429e-b262-85c537c83a0c" /> | <img width="451" height="399" alt="image" src="https://github.com/user-attachments/assets/e28668db-af6e-4af7-bf91-f10ded0897bc" /> |




---

## 🛠️ 技术栈

* **Language:** Python 3.12
* **Library:** OpenCV-Python, Numpy
* **Automation:** GitHub Actions 

## 🏗️ 本地开发

如果你想在本地跑跑看：

```bash
# 克隆项目
git clone https://github.com/AmethystDev-Labs/TenVision.git

# 安装依赖
pip install opencv-python numpy

# 运行 Demo
python main.py <输入的图片> <输出的图片位置>

```

---

## ⚖️ 本项目使用MIT许可证

