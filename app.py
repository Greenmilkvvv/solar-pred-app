# %%
import gradio as gr
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import joblib

# ========== 导入模型类 ==========
from NN import *  # 根据实际路径调整


# %%
# ========== 配置参数 ==========
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOOKBACK = 24  # 输入窗口长度（24个15分钟 = 6小时）
STEP_MINUTES = 15
FEATURE_ORDER = ['power', 'hour_sin', 'hour_cos', 
                 'shortwave_radiation (W/m2)', 'direct_radiation (W/m2)', 
                 'diffuse_radiation (W/m2)', 'direct_normal_irradiance (W/m2)']

MODEL_CONFIG = {
    'input_size': 7,           # 特征维度
    'hidden_size': 128,        # 从权重形状推断
    'num_layers': 2,           # 有 l0 和 l1
    'output_size': 1,
    'dropout': 0.2,            # 假设
    'bidirectional': False      
}

# ========== 加载模型 ==========
def load_model_and_scaler(model_path, scaler_path):
    """加载训练好的模型权重和标准化器"""
    
    # 创建基础LSTM模型
    lstm_model = LSTMPredictor(**MODEL_CONFIG)
    
    # 创建 GeneratorWithFeatures 包装器
    model = GeneratorWithFeatures(lstm_model)
    
    # 加载权重
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)
    
    # 加载 scaler
    scaler = joblib.load(scaler_path)
    
    return model, scaler

model, scaler = load_model_and_scaler('best_generator.pth', 'simple_scaler_1.pkl')

# model


# %%
# ========== 特征构造 ==========
def encode_hour(hour):
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    return hour_sin, hour_cos

def construct_input_sequence(history_power, history_radiation, target_hours):
    """构造模型的输入序列"""
    if len(history_power) < LOOKBACK:
        raise ValueError(f"历史数据不足，需要至少 {LOOKBACK} 个时间步")
    
    recent_power = history_power[-LOOKBACK:]
    recent_radiation = history_radiation[-LOOKBACK:]
    
    input_seq = []
    for t in range(LOOKBACK):
        hour = target_hours[t] % 24 if t < len(target_hours) else (target_hours[-1] + t - len(target_hours) + 1) % 24
        hour_sin, hour_cos = encode_hour(hour)
        
        rad = recent_radiation[t] if t < len(recent_radiation) else recent_radiation[-1]
        
        row = [
            recent_power[t],
            hour_sin,
            hour_cos,
            rad.get('shortwave_radiation (W/m2)', 0),
            rad.get('direct_radiation (W/m2)', 0),
            rad.get('diffuse_radiation (W/m2)', 0),
            rad.get('direct_normal_irradiance (W/m2)', 0)
        ]
        input_seq.append(row)
    
    return np.array(input_seq, dtype=np.float32)

def predict_single_step(model, input_seq, scaler):
    """单步预测"""
    input_scaled = scaler.transform(input_seq)
    input_tensor = torch.FloatTensor(input_scaled).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        pred_scaled = model(input_tensor).cpu().numpy()
    
    pred_orig = pred_scaled[0, 0] * scaler.scale_[0] + scaler.min_[0]
    return float(pred_orig)

def predict_multistep(model, initial_history, initial_radiation, 
                      target_hours_future, n_steps, scaler):
    """递归多步预测"""
    predictions = []
    history_power = list(initial_history)
    history_radiation = list(initial_radiation)
    
    for step in range(n_steps):
        input_seq = construct_input_sequence(
            history_power, 
            history_radiation, 
            list(range(len(history_power) - LOOKBACK, len(history_power)))
        )
        
        pred = predict_single_step(model, input_seq, scaler)
        predictions.append(pred)
        history_power.append(pred)
        
        future_rad = history_radiation[-1].copy() if history_radiation else {}
        history_radiation.append(future_rad)
    
    return predictions, history_power

def plot_predictions(history, predictions):
    """绘制预测结果"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    hist_len = len(history)
    hist_indices = np.arange(hist_len) * STEP_MINUTES / 60
    ax.plot(hist_indices, history, 'b-', label='历史数据', linewidth=2)
    
    pred_indices = np.arange(hist_len, hist_len + len(predictions)) * STEP_MINUTES / 60
    ax.plot(pred_indices, predictions, 'r--', label='预测值', linewidth=2, marker='o', markersize=4)
    
    ax.axvline(x=hist_indices[-1], color='gray', linestyle=':', alpha=0.7)
    ax.set_xlabel('时间 (小时)')
    ax.set_ylabel('功率 (kW)')
    ax.set_title('负荷预测结果')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


# %%
# ========== Gradio 接口函数 ==========
def predict_with_ui(history_power_str, history_radiation_str, n_steps, future_hours_str):
    """Gradio 调用的预测函数"""
    try:
        # 解析历史功率
        history_power = [float(x.strip()) for x in history_power_str.split(',') if x.strip()]
        
        # 解析历史辐射数据
        history_radiation = []
        for row in history_radiation_str.strip().split(';'):
            if not row.strip():
                continue
            vals = [float(x.strip()) for x in row.split(',') if x.strip()]
            if len(vals) >= 4:
                history_radiation.append({
                    'shortwave_radiation (W/m2)': vals[0],
                    'direct_radiation (W/m2)': vals[1],
                    'diffuse_radiation (W/m2)': vals[2],
                    'direct_normal_irradiance (W/m2)': vals[3]
                })
        
        # 解析未来小时
        future_hours = []
        if future_hours_str.strip():
            future_hours = [int(x.strip()) for x in future_hours_str.split(',') if x.strip()]
        if len(future_hours) < n_steps:
            last_hour = future_hours[-1] if future_hours else 0
            for i in range(len(future_hours), n_steps):
                future_hours.append((last_hour + i + 1) % 24)
        
        # 加载模型
        model, scaler = load_model_and_scaler('best_generator.pth', 'simple_scaler_1.pkl')
        
        if len(history_power) < LOOKBACK:
            return f"错误：历史数据不足，需要至少 {LOOKBACK} 个时间步", None
        
        predictions, _ = predict_multistep(
            model, history_power, history_radiation, future_hours, n_steps, scaler
        )
        
        result_text = "=== 预测结果 ===\n"
        for i, pred in enumerate(predictions):
            hour = future_hours[i] if i < len(future_hours) else (future_hours[-1] + i + 1) % 24
            result_text += f"第 {i+1} 步 (时刻: {hour:02d}:00): {pred:.2f} kW\n"
        
        fig = plot_predictions(history_power[-LOOKBACK:], predictions)
        
        return result_text, fig
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"预测出错：{str(e)}", None

# ========== 示例数据 ==========
EXAMPLE_DATA = [
    {
        "power": "0.0, 5.2, 12.3, 8.7, 15.2, 22.5, 18.3, 25.6, 32.1, 28.4, 35.2, 42.3, 38.7, 45.2, 52.1, 48.5, 55.3, 62.8, 58.2, 65.5, 72.3, 68.7, 75.2, 82.1",
        "radiation": "500,400,100,600; 520,410,110,620; 540,420,120,640; 560,430,130,660; 580,440,140,680; 600,450,150,700; 620,460,160,720; 640,470,170,740; 660,480,180,760; 680,490,190,780; 700,500,200,800; 720,510,210,820; 740,520,220,840; 760,530,230,860; 780,540,240,880; 800,550,250,900; 820,560,260,920; 840,570,270,940; 860,580,280,960; 880,590,290,980; 900,600,300,1000; 880,590,290,980; 860,580,280,960; 840,570,270,940",
        "n_steps": 12,
        "hours": "10,11,12,13,14,15,16,17,18,19,20,21"
    },
    {
        "power": "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0",
        "radiation": "0,0,0,0; 0,0,0,0; 0,0,0,0; 0,0,0,0; 0,0,0,0; 0,0,0,0; 0,0,0,0; 0,0,0,0; 0,0,0,0; 0,0,0,0; 0,0,0,0; 0,0,0,0; 0,0,0,0; 0,0,0,0; 0,0,0,0; 0,0,0,0; 0,0,0,0; 0,0,0,0; 0,0,0,0; 0,0,0,0; 0,0,0,0; 0,0,0,0; 0,0,0,0; 0,0,0,0",
        "n_steps": 8,
        "hours": "20,21,22,23,0,1,2,3"
    },
    {
        "power": "0.0, 0.0, 0.0, 0.0, 0.0, 5.2, 12.3, 18.5, 25.6, 32.1, 38.7, 45.2, 52.1, 58.2, 65.5, 72.3, 68.7, 62.8, 55.3, 48.5, 42.3, 35.2, 28.4, 22.5",
        "radiation": "0,0,0,0; 0,0,0,0; 0,0,0,0; 0,0,0,0; 0,0,0,0; 50,40,10,60; 150,120,30,180; 250,200,50,300; 350,280,70,420; 450,360,90,540; 550,440,110,660; 650,520,130,780; 750,600,150,900; 850,680,170,1020; 900,720,180,1080; 850,680,170,1020; 750,600,150,900; 650,520,130,780; 550,440,110,660; 450,360,90,540; 350,280,70,420; 250,200,50,300; 150,120,30,180; 50,40,10,60",
        "n_steps": 12,
        "hours": "9,10,11,12,13,14,15,16,17,18,19,20"
    }
]

# ========== 构建 Gradio 界面 ==========
def create_demo():
    with gr.Blocks(title="光储充微电网负荷预测", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # ⚡ 光储充微电网负荷预测系统
        ### 基于 LSTM-GAN 的15分钟功率预测
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                power_input = gr.Textbox(
                    label="历史功率序列 (kW)",
                    placeholder="示例：0.0, 5.2, 12.3, 8.7, ...",
                    lines=3,
                    info=f"需要至少 {LOOKBACK} 个时间步（每步15分钟）"
                )
                
                radiation_input = gr.Textbox(
                    label="历史辐射数据 (W/m²)",
                    placeholder="格式：shortwave,direct,diffuse,direct_normal; ...",
                    lines=5,
                    info="顺序：短波辐射,直接辐射,散射辐射,法向直接辐射"
                )
                
                with gr.Row():
                    n_steps_input = gr.Slider(
                        minimum=1, maximum=48, step=1, value=12,
                        label="预测步数"
                    )
                
                hours_input = gr.Textbox(
                    label="未来时刻的小时列表 (可选)",
                    placeholder="示例：10,11,12,13",
                    info="不填则自动推算，长度应等于预测步数"
                )
                
                predict_btn = gr.Button("🚀 开始预测", variant="primary")
                
                # 示例数据按钮区域
                gr.Markdown("### 📋 示例数据")
                example_btns = []
                for i, example in enumerate(EXAMPLE_DATA):
                    btn = gr.Button(f"示例 {i+1}")
                    btn.click(
                        fn=lambda p=example["power"], r=example["radiation"], 
                           n=example["n_steps"], h=example["hours"]: (p, r, n, h),
                        inputs=[],
                        outputs=[power_input, radiation_input, n_steps_input, hours_input]
                    )
                    example_btns.append(btn)
                
            with gr.Column(scale=2):
                output_text = gr.Textbox(label="预测详情", lines=10)
                output_plot = gr.Plot(label="预测曲线")
        
        predict_btn.click(
            fn=predict_with_ui,
            inputs=[power_input, radiation_input, n_steps_input, hours_input],
            outputs=[output_text, output_plot]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch( 
        share=True, 
        # server_name="0.0.0.0", 
        # server_port=7860
    )