import gradio as gr
import numpy as np
import torch
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ========== 导入模型类 ==========
from NN import LSTMPredictor, GeneratorWithFeatures

# ========== 配置参数 ==========
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOOKBACK = 24                      # 输入窗口长度（24个15分钟 = 6小时）
STEP_MINUTES = 15                  # 每个时间步的分钟数
HOURS_PER_DAY = 24
STEPS_PER_HOUR = 60 // STEP_MINUTES  # 60/15 = 4步/小时

# 特征列顺序（与训练时一致）
FEATURE_ORDER = ['power', 'hour_sin', 'hour_cos', 
                 'shortwave_radiation (W/m2)', 'direct_radiation (W/m2)', 
                 'diffuse_radiation (W/m2)', 'direct_normal_irradiance (W/m2)']

# 模型配置（与训练时一致）
MODEL_CONFIG = {
    'input_size': len(FEATURE_ORDER),
    'hidden_size': 128,
    'num_layers': 2,
    'output_size': 1,
    'dropout': 0.2,
    'bidirectional': False
}

# ========== 辅助函数：时间编码 ==========
def encode_hour(hour):
    """对小时进行周期编码"""
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    return hour_sin, hour_cos

def get_hour_from_step(start_datetime, step_index):
    """
    根据起始时间和步数索引，获取对应的小时
    
    Args:
        start_datetime: 起始时间 (datetime 对象)
        step_index: 步数索引（0表示起始时刻）
    
    Returns:
        hour: 0-23 的小时数
    """
    current_time = start_datetime + timedelta(minutes=step_index * STEP_MINUTES)
    return current_time.hour

# ========== 加载模型 ==========
def load_model_and_scaler(model_path, scaler_path):
    """加载训练好的模型和标准化器"""
    
    # 加载模型权重
    state_dict = torch.load(model_path, map_location=DEVICE)
    
    lstm_model = LSTMPredictor(**MODEL_CONFIG)
    model = GeneratorWithFeatures(lstm_model)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)
    
    # 加载标准化器
    scaler = joblib.load(scaler_path)
    
    print(f"模型加载成功，输入特征维度: {MODEL_CONFIG['input_size']}")
    return model, scaler

# ========== 构造输入序列 ==========
def construct_input_sequence(history_power, history_radiation, start_datetime):
    """
    构造模型的输入序列
    
    Args:
        history_power: 历史功率序列 (list, 长度 = LOOKBACK)
        history_radiation: 历史辐射数据 (list of dict, 每个dict包含4个辐射值)
        start_datetime: 起始时间 (datetime 对象)
    
    Returns:
        input_seq: (LOOKBACK, n_features) 的 numpy 数组
    """
    if len(history_power) != LOOKBACK:
        raise ValueError(f"历史数据需要 {LOOKBACK} 个时间步，当前 {len(history_power)} 个")
    
    input_seq = []
    
    for t in range(LOOKBACK):
        # 获取当前时刻的小时
        hour = get_hour_from_step(start_datetime, t)
        hour_sin, hour_cos = encode_hour(hour)
        
        # 获取辐射数据
        rad = history_radiation[t] if t < len(history_radiation) else history_radiation[-1]
        
        row = [
            history_power[t],
            hour_sin,
            hour_cos,
            rad.get('shortwave_radiation (W/m2)', 0),
            rad.get('direct_radiation (W/m2)', 0),
            rad.get('diffuse_radiation (W/m2)', 0),
            rad.get('direct_normal_irradiance (W/m2)', 0)
        ]
        input_seq.append(row)
    
    return np.array(input_seq, dtype=np.float32)

# ========== 单步预测 ==========
def predict_single_step(model, input_seq, scaler):
    """单步预测，返回原始单位的预测值"""
    # 标准化
    input_scaled = scaler.transform(input_seq)
    input_tensor = torch.FloatTensor(input_scaled).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        pred_scaled = model(input_tensor).cpu().numpy()
    
    # 反标准化（MinMaxScaler）
    pred_orig = pred_scaled[0, 0] * scaler.scale_[0] + scaler.min_[0]
    return float(pred_orig)

# ========== 多步预测（递归）==========
def predict_multistep(model, initial_history_power, initial_history_radiation, 
                      start_datetime, n_steps, scaler):
    """
    递归多步预测
    
    Args:
        initial_history_power: 初始历史功率序列 (长度 = LOOKBACK)
        initial_history_radiation: 初始历史辐射序列 (长度 = LOOKBACK)
        start_datetime: 起始时间（历史数据的第一个点）
        n_steps: 预测步数
        scaler: 标准化器
    
    Returns:
        predictions: 预测值列表
        future_datetimes: 预测时刻的时间列表
    """
    predictions = []
    future_datetimes = []
    
    # 当前历史（用于递归预测）
    current_history_power = list(initial_history_power)
    current_history_radiation = list(initial_history_radiation)
    
    for step in range(n_steps):
        # 预测时刻的时间（历史最后一个点后的第 step+1 个点）
        pred_time = start_datetime + timedelta(minutes=(LOOKBACK + step) * STEP_MINUTES)
        future_datetimes.append(pred_time)
        
        # 构造输入序列（使用当前历史）
        input_seq = construct_input_sequence(
            current_history_power[-LOOKBACK:],
            current_history_radiation[-LOOKBACK:],
            start_datetime
        )
        
        # 预测
        pred = predict_single_step(model, input_seq, scaler)
        predictions.append(pred)
        
        # 更新历史（将预测值加入）
        current_history_power.append(pred)
        
        # 辐射数据：使用最近的真实辐射或估算值
        if len(current_history_radiation) > 0:
            future_rad = current_history_radiation[-1].copy()
        else:
            future_rad = {'shortwave_radiation (W/m2)': 0, 'direct_radiation (W/m2)': 0,
                          'diffuse_radiation (W/m2)': 0, 'direct_normal_irradiance (W/m2)': 0}
        current_history_radiation.append(future_rad)
    
    return predictions, future_datetimes

# ========== 解析用户输入 ==========
def parse_power_sequence(power_str):
    """解析功率序列字符串"""
    return [float(x.strip()) for x in power_str.split(',') if x.strip()]

def parse_radiation_sequence(rad_str):
    """
    解析辐射数据字符串
    格式：每行一个时间步，逗号分隔四个值
    例如：500,400,100,600
         520,410,110,620
    """
    history_radiation = []
    for line in rad_str.strip().split('\n'):
        if not line.strip():
            continue
        vals = [float(x.strip()) for x in line.split(',') if x.strip()]
        if len(vals) >= 4:
            history_radiation.append({
                'shortwave_radiation (W/m2)': vals[0],
                'direct_radiation (W/m2)': vals[1],
                'diffuse_radiation (W/m2)': vals[2],
                'direct_normal_irradiance (W/m2)': vals[3]
            })
    return history_radiation

# ========== 绘图函数 ==========
def plot_predictions(history_power, predictions, start_datetime, future_datetimes):
    """绘制预测结果"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 历史数据时间轴
    hist_times = [start_datetime + timedelta(minutes=i * STEP_MINUTES) 
                  for i in range(len(history_power))]
    hist_hours = [t.hour + t.minute/60 for t in hist_times]
    
    # 预测数据时间轴
    pred_hours = [t.hour + t.minute/60 for t in future_datetimes]
    
    # 绘图
    ax.plot(hist_hours, history_power, 'b-', label='历史数据', linewidth=2, marker='o', markersize=3)
    ax.plot(pred_hours, predictions, 'r--', label='预测值', linewidth=2, marker='s', markersize=4)
    
    # 分界线
    split_hour = hist_hours[-1]
    ax.axvline(x=split_hour, color='gray', linestyle=':', alpha=0.7)
    
    ax.set_xlabel('时间 (小时)')
    ax.set_ylabel('功率 (kW)')
    ax.set_title('光储充微电网负荷预测结果')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 设置x轴刻度
    all_hours = hist_hours + pred_hours
    ax.set_xticks(np.arange(int(min(all_hours)), int(max(all_hours)) + 1, 2))
    
    return fig

# ========== 解析起始时间字符串 ==========
def parse_start_datetime(date_str, hour, minute):
    """
    解析起始时间
    
    Args:
        date_str: 日期字符串，格式 "YYYY-MM-DD"
        hour: 小时 (0-23)
        minute: 分钟 (0, 15, 30, 45)
    """
    return datetime.strptime(date_str, "%Y-%m-%d").replace(hour=hour, minute=minute)

# ========== Gradio 接口函数 ==========
def predict_with_ui(start_date_str, start_hour, start_minute,
                    power_history_str, radiation_history_str, n_steps):
    """
    预测函数
    """
    try:
        # 1. 解析起始时间
        start_datetime = parse_start_datetime(start_date_str, start_hour, start_minute)
        
        # 2. 解析历史数据
        history_power = parse_power_sequence(power_history_str)
        history_radiation = parse_radiation_sequence(radiation_history_str)
        
        # 3. 验证数据长度
        if len(history_power) < LOOKBACK:
            return f"错误：历史功率数据不足！需要至少 {LOOKBACK} 个时间步（{LOOKBACK * STEP_MINUTES} 分钟），当前 {len(history_power)} 个。", None
        if len(history_radiation) < LOOKBACK:
            return f"错误：历史辐射数据不足！需要至少 {LOOKBACK} 个时间步，当前 {len(history_radiation)} 个。", None
        
        # 只取前 LOOKBACK 个
        history_power = history_power[:LOOKBACK]
        history_radiation = history_radiation[:LOOKBACK]
        
        # 4. 加载模型
        model, scaler = load_model_and_scaler('best_generator.pth', 'simple_scaler_1.pkl')
        
        # 5. 多步预测
        predictions, future_datetimes = predict_multistep(
            model, history_power, history_radiation, start_datetime, n_steps, scaler
        )
        
        # 6. 生成结果文本
        result_text = "=== 预测结果 ===\n"
        result_text += f"历史数据时间范围: {start_datetime.strftime('%Y-%m-%d %H:%M')} 至 "
        result_text += f"{(start_datetime + timedelta(minutes=(LOOKBACK-1)*STEP_MINUTES)).strftime('%Y-%m-%d %H:%M')}\n"
        result_text += f"预测步数: {n_steps} 步（每步 {STEP_MINUTES} 分钟）\n"
        result_text += "-" * 40 + "\n"
        
        for i, (pred, dt) in enumerate(zip(predictions, future_datetimes)):
            result_text += f"第 {i+1:2d} 步: {dt.strftime('%m-%d %H:%M')} → {pred:.2f} kW\n"
        
        # 7. 生成图表
        fig = plot_predictions(history_power, predictions, start_datetime, future_datetimes)
        
        return result_text, fig
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"预测出错：{str(e)}", None

# ========== 构建 Gradio 界面 ==========
def create_demo():
    with gr.Blocks(title="光储充微电网负荷预测", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # ⚡ 光储充微电网负荷预测系统
        ### 基于 LSTM-GAN 的 15分钟 功率预测
        
        输入历史数据（6小时，24个时间步），预测未来负荷
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📅 历史数据起始时间")
                with gr.Row():
                    # 使用 Textbox 代替 Date（兼容性更好）
                    start_date = gr.Textbox(
                        label="起始日期", 
                        placeholder="YYYY-MM-DD，例如：2024-01-15",
                        value="2024-01-15"
                    )
                with gr.Row():
                    start_hour = gr.Slider(minimum=0, maximum=23, step=1, value=8, label="起始小时")
                    start_minute = gr.Slider(minimum=0, maximum=45, step=15, value=0, label="起始分钟")
                
                gr.Markdown(f"### 📊 历史数据（需要 {LOOKBACK} 个时间步，每步 {STEP_MINUTES} 分钟）")
                
                power_history = gr.Textbox(
                    label="历史功率序列 (kW)",
                    placeholder=f"输入 {LOOKBACK} 个功率值，逗号分隔\n示例：0.0, 5.2, 12.3, 8.7, ...",
                    lines=3
                )
                
                radiation_history = gr.Textbox(
                    label="历史辐射数据 (W/m²)",
                    placeholder=f"输入 {LOOKBACK} 行，每行格式：shortwave,direct,diffuse,direct_normal\n示例：\n500,400,100,600\n520,410,110,620\n...",
                    lines=6
                )
                
                n_steps_input = gr.Slider(
                    minimum=1, maximum=48, step=1, value=12,
                    label=f"预测步数（每步 {STEP_MINUTES} 分钟，最多12小时）"
                )
                
                predict_btn = gr.Button("🚀 开始预测", variant="primary")
                
                # 示例数据
                gr.Markdown("### 📋 示例数据")
                
                # 示例
                example_power = "0.0, 5.2, 12.3, 18.5, 25.6, 32.1, 38.7, 45.2, 52.1, 58.2, 65.5, 72.3, 68.7, 62.8, 55.3, 48.5, 42.3, 35.2, 28.4, 22.5, 15.2, 8.7, 3.2, 0.0"
                example_radiation = "0,0,0,0\n50,40,10,60\n150,120,30,180\n250,200,50,300\n350,280,70,420\n450,360,90,540\n550,440,110,660\n650,520,130,780\n750,600,150,900\n850,680,170,1020\n900,720,180,1080\n850,680,170,1020\n750,600,150,900\n650,520,130,780\n550,440,110,660\n450,360,90,540\n350,280,70,420\n250,200,50,300\n150,120,30,180\n50,40,10,60\n0,0,0,0\n0,0,0,0\n0,0,0,0\n0,0,0,0"
                
                gr.Examples(
                    examples=[
                        ["2024-01-15", 8, 0, example_power, example_radiation, 12],
                    ],
                    inputs=[start_date, start_hour, start_minute, power_history, radiation_history, n_steps_input],
                    label="点击加载示例（日出后上升场景）"
                )
                
            with gr.Column(scale=2):
                output_text = gr.Textbox(label="预测详情", lines=15)
                output_plot = gr.Plot(label="预测曲线")
        
        predict_btn.click(
            fn=predict_with_ui,
            inputs=[start_date, start_hour, start_minute, power_history, radiation_history, n_steps_input],
            outputs=[output_text, output_plot]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=True)