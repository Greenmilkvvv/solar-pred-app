# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Tuple, Optional
import time


# 可视化
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']  # 英文字体优先, 中文回退到宋体
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式字体, 与Times风格匹配
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号


# %%
class LSTMPredictor(nn.Module):
    """多变量LSTM预测模型, 支持单步或多步输出"""
    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, output_size: int = 1,
                 dropout: float = 0.2, bidirectional: bool = False):
        """
        Args:
            input_size: 输入特征维度
            hidden_size: LSTM隐藏层维度
            num_layers: LSTM层数
            output_size: 输出维度 (预测步数) 
            dropout: Dropout比率 (多层LSTM之间)
            bidirectional: 是否使用双向LSTM
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * direction_factor, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)          # out: (batch, seq_len, hidden_size * direction)
        # 取最后一个时间步的输出
        last_step = out[:, -1, :]       # (batch, hidden_size * direction)
        y_pred = self.fc(last_step)     # (batch, output_size)
        return y_pred


class CNN_LSTM(nn.Module):
    """
    CNN-LSTM混合模型: 先用CNN提取局部特征, 再输入LSTM捕捉时序依赖
    完全兼容原有的LSTMPredictor接口, 可直接替换
    """
    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, output_size: int = 1,
                 dropout: float = 0.2, bidirectional: bool = False,
                 cnn_channels: list = [64, 64], kernel_size: int = 3):
        """
        Args:
            input_size: 输入特征维度
            hidden_size: LSTM隐藏层维度
            num_layers: LSTM层数
            output_size: 输出维度 (预测步数) 
            dropout: Dropout比率
            bidirectional: 是否使用双向LSTM
            cnn_channels: CNN各层输出通道数列表, 如[64, 64]表示两层CNN
            kernel_size: 卷积核大小
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.cnn_channels = cnn_channels
        
        # 构建CNN层
        cnn_layers = []
        in_channels = input_size  # 输入通道数等于特征维度
        
        for out_channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, 
                          padding=kernel_size//2),  # 保持序列长度不变
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # CNN输出维度 = 最后一个CNN层的输出通道数
        cnn_output_size = cnn_channels[-1] if cnn_channels else input_size
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 全连接输出层
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * direction_factor, output_size)
        
    def forward(self, x):
        """
        Args:
            x: 输入张量, 形状 (batch_size, seq_len, input_size)
        
        Returns:
            预测值, 形状 (batch_size, output_size)
        """
        batch_size, seq_len, n_features = x.shape
        
        # CNN期望输入形状: (batch, channels, seq_len)
        # 将特征维度作为通道维度
        x_cnn = x.permute(0, 2, 1)  # (batch, features, seq_len)
        
        # 通过CNN层
        cnn_out = self.cnn(x_cnn)  # (batch, cnn_channels[-1], seq_len)
        
        # 转换回LSTM需要的形状: (batch, seq_len, cnn_channels[-1])
        lstm_input = cnn_out.permute(0, 2, 1)  # (batch, seq_len, channels)
        
        # LSTM层
        lstm_out, _ = self.lstm(lstm_input)  # (batch, seq_len, hidden_size * direction)
        
        # 取最后一个时间步的输出
        last_step = lstm_out[:, -1, :]  # (batch, hidden_size * direction)
        
        # 全连接层输出
        output = self.fc(last_step)  # (batch, output_size)
        
        return output



class CNN_LSTM_Advanced(nn.Module):
    """
    增强版CNN-LSTM: 包含残差连接和注意力机制
    """
    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, output_size: int = 1,
                 dropout: float = 0.2, bidirectional: bool = False,
                 cnn_channels: list = [64, 128], kernel_size: int = 3,
                 use_attention: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # CNN特征提取
        cnn_layers = []
        in_channels = input_size
        
        for i, out_channels in enumerate(cnn_channels):
            conv_block = [
                nn.Conv1d(in_channels, out_channels, kernel_size, 
                          padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            # 添加残差连接 (如果输入输出维度匹配) 
            if in_channels == out_channels:
                conv_block.append(ResidualConnection())
            
            cnn_layers.extend(conv_block)
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 注意力机制
        direction_factor = 2 if bidirectional else 1
        lstm_output_size = hidden_size * direction_factor
        
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(lstm_output_size, lstm_output_size // 2),
                nn.Tanh(),
                nn.Linear(lstm_output_size // 2, 1)
            )
        
        # 输出层
        self.fc = nn.Linear(lstm_output_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, n_features = x.shape
        
        # CNN处理
        x_cnn = x.permute(0, 2, 1)
        cnn_out = self.cnn(x_cnn)
        lstm_input = cnn_out.permute(0, 2, 1)
        
        # LSTM处理
        lstm_out, _ = self.lstm(lstm_input)  # (batch, seq_len, hidden_size * direction)
        
        # 注意力机制
        if self.use_attention:
            # 计算注意力权重
            attention_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
            attention_weights = torch.softmax(attention_weights, dim=1)
            
            # 加权求和
            context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, lstm_output_size)
        else:
            # 取最后一个时间步
            context = lstm_out[:, -1, :]
        
        # 输出层
        context = self.dropout(context)
        output = self.fc(context)
        
        return output


class ResidualConnection(nn.Module):
    """残差连接模块"""
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x + x  # 实际应用中需要调整, 这里作为占位符
    


# %%
def create_sequences(data: np.ndarray, target_col_idx: int,
                     lookback: int, horizon: int = 1,
                     feature_cols: Optional[list] = None):
    """
    将多变量时间序列转换为监督学习样本. 

    Args:
        data: 2D数组, 形状 (n_samples, n_features), 假设按时间顺序排列
        target_col_idx: 目标变量 (功率) 在特征矩阵中的列索引
        lookback: 输入窗口长度
        horizon: 预测步数 (默认为1, 单步预测) 
        feature_cols: 指定使用的特征列索引, 若为None则使用所有列

    Returns:
        X: 输入特征, 形状 (n_samples - lookback - horizon + 1, lookback, n_features)
        y: 目标值, 形状 (n_samples - lookback - horizon + 1, horizon)
    """
    if feature_cols is not None:
        data = data[:, feature_cols]
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:i+lookback, :])                 # 所有特征
        y.append(data[i+lookback:i+lookback+horizon, target_col_idx])
    return np.array(X), np.array(y)


def train_model(model, train_loader, val_loader, criterion, optimizer,
                scheduler=None, num_epochs=100, patience=10, device='cpu',
                model_save_path='best_model.pth'):
    """
    训练模型, 包含早停和学习率调度. 

    Args:
        model: 待训练模型
        train_loader: 训练数据DataLoader
        val_loader: 验证数据DataLoader
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器 (可选) 
        num_epochs: 最大训练轮数
        patience: 早停耐心值
        device: 设备
        model_save_path: 最佳模型保存路径
    """
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)

        # 学习率调度
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # 早停与模型保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')

    print(f'Training finished. Best val loss: {best_val_loss:.4f} at epoch {best_epoch+1}')
    # 加载最佳模型
    model.load_state_dict(torch.load(model_save_path))
    return model


def evaluate_model(model, test_loader, criterion, device='cpu'):
    """在测试集上评估模型, 返回损失和所有预测/真实值"""
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item() * X_batch.size(0)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    test_loss /= len(test_loader.dataset)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    return test_loss, all_preds, all_targets





# %%
# 残差修正
class GRUResidualCorrector(nn.Module):
    """
    用 GRU 处理残差历史, 并与当前特征融合预测残差
    """
    def __init__(self, residual_dim=1, hidden_size=16, num_layers=1,
                 feature_dim=None, output_dim=1, dropout=0.1):
        """
        Args:
            residual_dim: 残差序列的维度 (通常为1) 
            hidden_size: GRU隐藏层大小
            num_layers: GRU层数
            feature_dim: 当前特征维度 (如天气、时间等) 
            output_dim: 输出残差修正值维度
            dropout: Dropout比率
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.feature_dim = feature_dim

        # GRU 处理残差历史
        self.gru = nn.GRU(
            input_size=residual_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 将 GRU 最后输出与当前特征拼接后, 通过全连接输出残差
        self.fc = nn.Linear(hidden_size + (feature_dim if feature_dim else 0), output_dim)

    def forward(self, residual_seq, current_features=None):
        """
        Args:
            residual_seq: (batch, seq_len, residual_dim)  残差历史序列
            current_features: (batch, feature_dim) 或 None
        Returns:
            residual_pred: (batch, output_dim)
        """
        # GRU 编码残差历史
        out, _ = self.gru(residual_seq)          # out: (batch, seq_len, hidden_size)
        last_hidden = out[:, -1, :]               # (batch, hidden_size)

        # 拼接当前特征
        if current_features is not None:
            combined = torch.cat([last_hidden, current_features], dim=1)
        else:
            combined = last_hidden

        # 输出残差预测
        residual_pred = self.fc(combined)
        return residual_pred


def build_residual_samples(main_model, data_loader, residual_lookback, device):
    """
    从主模型和数据加载器构建残差修正器的训练样本
    Returns:
        X_residual: (n_samples, lookback, 1) 残差历史
        X_features: (n_samples, n_features)  当前特征 (每个样本最后一个时间步) 
        y_residual: (n_samples, 1)           目标残差
    """
    main_model.eval()
    all_residuals = []
    all_features = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = main_model(X_batch)
            residual = (y_batch - pred).cpu().numpy()   # (batch, 1)
            # 当前特征: 取每个序列的最后一个时间步
            curr_features = X_batch[:, -1, :].cpu().numpy()  # (batch, n_features)

            all_residuals.append(residual)
            all_features.append(curr_features)

    all_residuals = np.vstack(all_residuals).flatten()   # (total_samples,)
    all_features = np.vstack(all_features)               # (total_samples, n_features)

    # 构建滑动窗口样本
    X_res, y_res, X_feat = [], [], []
    for i in range(residual_lookback, len(all_residuals)):
        X_res.append(all_residuals[i-residual_lookback:i].reshape(-1, 1))  # (lookback, 1)
        y_res.append(all_residuals[i])
        X_feat.append(all_features[i])   # 当前特征

    X_res = np.array(X_res)          # (n_samples, lookback, 1)
    y_res = np.array(y_res).reshape(-1, 1)
    X_feat = np.array(X_feat)        # (n_samples, n_features)

    return X_res, X_feat, y_res


def train_gru_corrector(main_model, train_loader, val_loader,
                        residual_lookback=12, hidden_size=16,
                        epochs=30, lr=0.001, patience=5, device='cpu'):
    """
    训练 GRU 残差修正器
    """
    print("构建训练样本...")
    X_train_res, X_train_feat, y_train = build_residual_samples(
        main_model, train_loader, residual_lookback, device
    )
    X_val_res, X_val_feat, y_val = build_residual_samples(
        main_model, val_loader, residual_lookback, device
    )

    print(f"训练样本数: {len(X_train_res)}, 验证样本数: {len(X_val_res)}")
    print(f"残差历史形状: {X_train_res.shape}, 特征形状: {X_train_feat.shape}")

    # 转换为 tensor
    X_train_res = torch.FloatTensor(X_train_res).to(device)
    X_train_feat = torch.FloatTensor(X_train_feat).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_val_res = torch.FloatTensor(X_val_res).to(device)
    X_val_feat = torch.FloatTensor(X_val_feat).to(device)
    y_val = torch.FloatTensor(y_val).to(device)

    # 创建 DataLoader (注意时间序列不打乱) 
    train_dataset = TensorDataset(X_train_res, X_train_feat, y_train)
    val_dataset = TensorDataset(X_val_res, X_val_feat, y_val)
    train_loader_gru = DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_loader_gru = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 初始化模型
    feature_dim = X_train_feat.shape[1]
    model = GRUResidualCorrector(
        residual_dim=1,
        hidden_size=hidden_size,
        num_layers=1,
        feature_dim=feature_dim,
        output_dim=1,
        dropout=0.2
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0

    print("\n开始训练 GRU 残差修正器...")
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0.0
        for res_seq, feat, target in train_loader_gru:
            optimizer.zero_grad()
            pred = model(res_seq, feat)
            loss = criterion(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * res_seq.size(0)
        train_loss /= len(train_dataset)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for res_seq, feat, target in val_loader_gru:
                pred = model(res_seq, feat)
                loss = criterion(pred, target)
                val_loss += loss.item() * res_seq.size(0)
        val_loss /= len(val_dataset)

        print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_gru_corrector.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停于 epoch {epoch+1}")
                break

    # 加载最佳模型
    model.load_state_dict(torch.load("best_gru_corrector.pth"))
    print(f"\n训练完成, 最佳验证损失: {best_val_loss:.6f}")

    return model


def evaluate_gru_corrector(main_model, gru_model, test_loader,
                           residual_lookback, scaler, target_idx, device):
    """
    评估两阶段模型 (主模型 + GRU残差修正) 
    """
    # 构建测试样本
    X_test_res, X_test_feat, y_test = build_residual_samples(
        main_model, test_loader, residual_lookback, device
    )
    X_test_res = torch.FloatTensor(X_test_res).to(device)
    X_test_feat = torch.FloatTensor(X_test_feat).to(device)

    gru_model.eval()
    with torch.no_grad():
        pred_res = gru_model(X_test_res, X_test_feat).cpu().numpy()

    # 主模型的预测值 (从构建样本时对应的主模型预测得到) 
    # 重新跑一次测试集获取主模型预测
    main_model.eval()
    main_preds = []
    targets = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            pred = main_model(X_batch).cpu().numpy()
            main_preds.append(pred)
            targets.append(y_batch.numpy())
    main_preds = np.vstack(main_preds).flatten()
    targets = np.vstack(targets).flatten()

    # 注意: 残差预测对应的是从 residual_lookback 开始的样本, 需要对齐索引
    # build_residual_samples 返回的样本从索引 residual_lookback 开始
    aligned_main_preds = main_preds[residual_lookback:]
    aligned_targets = targets[residual_lookback:]

    # 反标准化
    power_mean = scaler.mean_[target_idx]
    power_std = scaler.scale_[target_idx]

    main_preds_orig = aligned_main_preds * power_std + power_mean
    targets_orig = aligned_targets * power_std + power_mean
    final_preds_orig = (aligned_main_preds + pred_res.flatten()) * power_std + power_mean

    # 计算指标
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae_main = mean_absolute_error(targets_orig, main_preds_orig)
    mae_final = mean_absolute_error(targets_orig, final_preds_orig)
    rmse_main = np.sqrt(mean_squared_error(targets_orig, main_preds_orig))
    rmse_final = np.sqrt(mean_squared_error(targets_orig, final_preds_orig))

    print("\n========== 测试集结果 ==========")
    print(f"{'指标':<15} {'主LSTM':<15} {'+GRU残差':<15} {'改进':<10}")
    print("-" * 55)
    print(f"{'MAE (kW)':<15} {mae_main:<15.4f} {mae_final:<15.4f} "
          f"{(mae_main-mae_final)/mae_main*100:>+6.2f}%")
    print(f"{'RMSE (kW)':<15} {rmse_main:<15.4f} {rmse_final:<15.4f} "
          f"{(rmse_main-rmse_final)/rmse_main*100:>+6.2f}%")

    return {
        'mae_main': mae_main, 'mae_final': mae_final,
        'rmse_main': rmse_main, 'rmse_final': rmse_final
    }


# %% 
# Adversarial LSTM

## 1 基于FGSM的对抗训练 (防御)

def fgsm_attack(model, X, y, epsilon, criterion, device):
    """
    使用FGSM方法生成对抗样本
    Args:
        model: 当前正在训练的LSTM模型
        X: 原始输入数据 (batch, seq_len, n_features)
        y: 真实标签
        epsilon: 扰动大小
        criterion: 损失函数
    Returns:
        X_adv: 生成的对抗样本
    """
    model.eval() # 切换到评估模式, 因为我们不想更新模型参数
    X.requires_grad = True # 开启输入梯度

    # 1. 前向传播
    output = model(X)
    loss = criterion(output, y)

    # 2. 反向传播, 计算关于输入 X 的梯度
    model.zero_grad()
    loss.backward()

    # 3. 获取梯度符号, 并生成扰动
    # X.grad 的形状和 X 一样
    data_grad = X.grad.data
    # 生成扰动: epsilon * sign(grad)
    perturbed_data = X + epsilon * data_grad.sign()

    # 防止数值溢出, 可以进行裁剪 (如果输入是归一化到[0,1]或[-1,1]) 
    # perturbed_data = torch.clamp(perturbed_data, 0, 1)

    model.train() # 切回训练模式
    return perturbed_data.detach() # 返回对抗样本, 并断开计算图


import copy

def train_model_adversarial(model, train_loader, val_loader, criterion, optimizer,
                            scheduler=None, num_epochs=100, patience=10, device='cpu',
                            model_save_path='best_model.pth', epsilon=0.1):
    """
    对抗性训练LSTM模型 (修复版) 
    """
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        # --- 训练阶段 ---
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # 不切换模式, 而是用上下文管理器控制梯度
            # 确保X_batch需要梯度
            X_batch.requires_grad_(True)
            
            # 1. 前向传播 (仍处于train模式) 
            output = model(X_batch)
            loss_adv = criterion(output, y_batch)
            
            # 2. 清空之前的梯度
            model.zero_grad()
            
            # 3. 反向传播计算关于输入的梯度
            loss_adv.backward(retain_graph=False)  # 计算梯度
            
            # 4. 获取梯度并生成对抗样本
            data_grad = X_batch.grad.data.clone()  # 用clone避免后续修改
            X_adv = X_batch + epsilon * data_grad.sign()
            
            # 5. 用对抗样本重新训练 (分离计算图) 
            X_adv = X_adv.detach().requires_grad_(False)
            output_adv = model(X_adv)
            loss = criterion(output_adv, y_batch)
            
            # 6. 清空梯度并优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 7. 清理X_batch的梯度, 为下一个batch做准备
            X_batch.grad = None
            X_batch.requires_grad_(False)
            
            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)

        # --- 学习率调度与早停 ---
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs} | Train Loss (adv): {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')

    print(f'Adversarial Training finished. Best val loss: {best_val_loss:.4f} at epoch {best_epoch+1}')
    model.load_state_dict(torch.load(model_save_path))
    return model



# %%
# GAN
class Discriminator(nn.Module):
    """
    判别器: 判断 (特征, 功率值) 对是真实还是生成的
    """
    def __init__(self, feature_dim, hidden_dims=[64, 32], output_dim=1, dropout=0.2):
        """
        Args:
            feature_dim: 输入特征维度 (LSTM每个时间步的特征数) 
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度 (二分类, 通常为1) 
        """
        super().__init__()
        
        # 输入维度 = 特征维度 + 功率值维度 (1) 
        input_dim = feature_dim + 1
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2),  # GAN中常用LeakyReLU
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # 输出层 (sigmoid会在损失函数中处理) 
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, features, power_value):
        """
        Args:
            features: 输入特征 (batch, feature_dim) - 最后一个时间步的特征
            power_value: 功率值 (batch, 1) - 真实值或生成值
        Returns:
            output: 判别结果 (batch, 1), 未经过sigmoid (BCEWithLogitsLoss会处理) 
        """
        # 拼接特征和功率值
        x = torch.cat([features, power_value], dim=1)
        return self.model(x)


class GeneratorWithFeatures(nn.Module):
    """
    生成器: LSTM模型, 包装成GAN需要的接口
    注意: 这个类只是包装, 实际使用已有的LSTM模型
    """
    def __init__(self, lstm_model):
        super().__init__()
        self.lstm_model = lstm_model
        
    def forward(self, X_seq):
        """
        X_seq: 完整的输入序列 (batch, seq_len, n_features)
        返回: 预测值 (batch, 1)
        """
        return self.lstm_model(X_seq)
    
    def get_last_features(self, X_seq):
        """
        获取最后一个时间步的特征, 用于判别器
        """
        return X_seq[:, -1, :]  # (batch, n_features)
    

class GANTrainer:
    """
    GAN训练器: 管理生成器(LSTM)和判别器的对抗训练
    """
    def __init__(self, generator, discriminator, device='cpu'):
        """
        Args:
            generator: 你的LSTM模型 (生成器) 
            discriminator: 判别器网络
        """
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        
        # 默认优化器 (可以在训练时传入自定义的) 
        self.g_optimizer = None
        self.d_optimizer = None
        
        # 损失函数 (二分类交叉熵, 已包含sigmoid) 
        self.criterion = nn.BCEWithLogitsLoss()
        
        # 训练历史
        self.history = {
            'g_loss': [], 'd_loss': [], 'd_acc_real': [], 'd_acc_fake': []
        }
        
    def set_optimizers(self, g_optimizer, d_optimizer):
        """设置生成器和判别器的优化器"""
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        
    def train_epoch(self, data_loader, lambda_gan=0.1, lambda_mse=1.0, 
                    d_steps=1, g_steps=1):
        """
        训练一个epoch
        
        Args:
            data_loader: 训练数据DataLoader, 返回 (X_seq, y_true)
            lambda_gan: GAN损失的权重
            lambda_mse: MSE损失的权重 (保持预测精度) 
            d_steps: 每轮训练判别器的步数
            g_steps: 每轮训练生成器的步数
        """
        self.generator.train()
        self.discriminator.train()
        
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_d_acc_real = 0.0
        epoch_d_acc_fake = 0.0
        n_batches = 0
        
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            batch_size = X_batch.size(0)
            
            # 获取最后一个时间步的特征
            last_features = X_batch[:, -1, :]  # (batch, n_features)
            
            # ========== 训练判别器 ==========
            for _ in range(d_steps):
                # 真实样本: 特征 + 真实功率值
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)
                
                # 生成器预测 (生成假样本) 
                with torch.no_grad():  # 训练判别器时不更新生成器
                    fake_power = self.generator(X_batch)
                
                # 判别器判断真实样本
                real_output = self.discriminator(last_features, y_batch)
                d_loss_real = self.criterion(real_output, real_labels)
                
                # 判别器判断生成样本
                fake_output = self.discriminator(last_features, fake_power)
                d_loss_fake = self.criterion(fake_output, fake_labels)
                
                # 总判别器损失
                d_loss = d_loss_real + d_loss_fake
                
                # 计算准确率 (用于监控) 
                d_acc_real = (real_output > 0).float().mean().item()
                d_acc_fake = (fake_output < 0).float().mean().item()
                
                # 更新判别器
                if self.d_optimizer is not None:
                    self.d_optimizer.zero_grad()
                    d_loss.backward()
                    self.d_optimizer.step()
            
            # ========== 训练生成器 ==========
            for _ in range(g_steps):
                # 生成器预测
                fake_power = self.generator(X_batch)
                
                # MSE损失 (保持预测精度) 
                mse_loss = nn.MSELoss()(fake_power, y_batch)
                
                # GAN损失 (欺骗判别器) 
                fake_output = self.discriminator(last_features, fake_power)
                g_gan_loss = self.criterion(fake_output, real_labels)  # 希望判别器判断为真
                
                # 总生成器损失
                g_loss = lambda_mse * mse_loss + lambda_gan * g_gan_loss
                
                # 更新生成器
                if self.g_optimizer is not None:
                    self.g_optimizer.zero_grad()
                    g_loss.backward()
                    self.g_optimizer.step()
            
            # 记录统计
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_d_acc_real += d_acc_real
            epoch_d_acc_fake += d_acc_fake
            n_batches += 1
        
        # 平均
        self.history['g_loss'].append(epoch_g_loss / n_batches)
        self.history['d_loss'].append(epoch_d_loss / n_batches)
        self.history['d_acc_real'].append(epoch_d_acc_real / n_batches)
        self.history['d_acc_fake'].append(epoch_d_acc_fake / n_batches)
        
        return {
            'g_loss': self.history['g_loss'][-1],
            'd_loss': self.history['d_loss'][-1],
            'd_acc_real': self.history['d_acc_real'][-1],
            'd_acc_fake': self.history['d_acc_fake'][-1]
        }
    
    def train(self, train_loader, val_loader, epochs=50,
              lambda_gan=0.1, lambda_mse=1.0,
              d_steps=1, g_steps=1, patience=5):
        """
        完整训练循环
        
        Args:
            train_loader: 训练数据
            val_loader: 验证数据
            epochs: 训练轮数
            lambda_gan: GAN损失权重
            lambda_mse: MSE损失权重
            d_steps: 每轮判别器训练步数
            g_steps: 每轮生成器训练步数
            patience: 早停耐心值
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练一个epoch
            train_metrics = self.train_epoch(
                train_loader, lambda_gan, lambda_mse, d_steps, g_steps
            )
            
            # 验证
            val_loss = self.validate(val_loader)
            
            # 打印
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"G Loss: {train_metrics['g_loss']:.4f} | "
                      f"D Loss: {train_metrics['d_loss']:.4f} | "
                      f"D Acc Real: {train_metrics['d_acc_real']:.2f} | "
                      f"D Acc Fake: {train_metrics['d_acc_fake']:.2f} | "
                      f"Val MSE: {val_loss:.4f}")
            
            # 早停 (基于验证集的MSE) 
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存生成器 (LSTM模型) 
                torch.save(self.generator.state_dict(), 'best_generator.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # 加载最佳生成器
        self.generator.load_state_dict(torch.load('best_generator.pth'))
        return self.generator
    
    def validate(self, val_loader):
        """验证生成器的MSE损失"""
        self.generator.eval()
        total_loss = 0.0
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                pred = self.generator(X_batch)
                loss = criterion(pred, y_batch)
                total_loss += loss.item() * X_batch.size(0)
        
        return total_loss / len(val_loader.dataset)
    
    def plot_history(self):
        """绘制训练历史"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # 损失曲线
        axes[0].plot(self.history['g_loss'], label='Generator Loss')
        axes[0].plot(self.history['d_loss'], label='Discriminator Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('GAN Losses')
        axes[0].legend()
        axes[0].grid(True)
        
        # 判别器准确率
        axes[1].plot(self.history['d_acc_real'], label='Real Accuracy')
        axes[1].plot(self.history['d_acc_fake'], label='Fake Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Discriminator Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        # 平衡度 (理想情况下两条线应该在0.5附近) 
        axes[2].plot(np.array(self.history['d_acc_real']) - 0.5, label='Real - 0.5')
        axes[2].plot(np.array(self.history['d_acc_fake']) - 0.5, label='Fake - 0.5')
        axes[2].axhline(y=0, color='r', linestyle='--')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Deviation from 0.5')
        axes[2].set_title('Discriminator Balance')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        # plt.savefig('gan_training_history.png')
        plt.show()