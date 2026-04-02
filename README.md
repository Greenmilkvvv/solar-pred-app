# 基于 `Gradio` 的 Web 应用搭建

## 项目结构

### 本体

- `requirements.txt`: 项目依赖
- `app.py`: 主程序，用于启动 `Gradio` 应用

### 其他

- `app_test.py`: 用于测试的 `Gradio` 应用
- `NN.py`: 神经网络模型和训练工具
- `best_generator.pth`: 训练好的模型权重
- `simple_scaler_1.pkl`: 缩放器 (scaler), 来源于模型训练时
- `examples.pkl`: 测试用例
  
## 使用方法

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动应用

```bash
python app.py
```

