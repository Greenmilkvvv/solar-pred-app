"""
旨在给出 app 的用例
"""

# %%
import numpy as np
import pandas as pd

import pickle


# %%
DATA_PATH = '../Data/aligned_2026_01_02.csv'
TEST_SIZE = 0.15
VARIABLE = [ 
    'datetime',
    'power', 
    'shortwave_radiation (W/m2)', 'direct_radiation (W/m2)', 'diffuse_radiation (W/m2)', 'direct_normal_irradiance (W/m2)'
]
LENGTH = 250


# %%
def prepare_example(DATA_PATH, TEST_SIZE, VARIABLE, LENGTH):
    # 获取数据
    df = pd.read_csv(DATA_PATH, parse_dates=['datetime'])[VARIABLE]
    df.sort_values('datetime', inplace=True)          # 确保时间顺序

    # 获取测试集数据
    n = len(df)
    val_end = int(n * (1 - TEST_SIZE)) # 选取最后的 15% 作为测试集
    df = df.iloc[val_end:, :] # 选取训练集数据
    df = df.iloc[:LENGTH, ] # 只打算要 LENGTH 条数据


    # 1. 获取开始的时间点
    start_datetime = df['datetime'].iloc[0]
    # 2. 获取 power 的历史数据
    power_data = df['power'].values.astype(np.float32)
    # 3. 提取气象数据的特征矩阵
    feature_cols = VARIABLE[2:] # 4 个光照特征
    radiation_data = df[feature_cols].values.astype(np.float32)

    return start_datetime, power_data, radiation_data


# %%
if __name__ == '__main__':
    start_datetime, power_data, radiation_data = prepare_example(DATA_PATH, TEST_SIZE, VARIABLE, LENGTH)

    print(f"选择了测试集的 {LENGTH} 条数据")
    print(f"开始时间点为: {start_datetime}")

    # 1. 日期
    date = start_datetime.strftime('%Y-%m-%d')
    hour = start_datetime.hour
    minute = start_datetime.minute

    # 2. power
    power_data = power_data.tolist()
    power_str = ", ".join([str(i) for i in power_data])

    # 3. radiation
    radiation_data = radiation_data.tolist() # 2 维列表
    # 浮点数元素转为 str
    radiation_data = [','.join([str(i) for i in lst]) for lst in radiation_data]
    radiation_str = '\n'.join(radiation_data) # 1 维列表
    

    # in all
    examples = [ 
        date, hour, minute, 
        power_str, 
        radiation_str
    ]

    # 转为 pkl
    with open('examples.pkl', 'wb') as f:
        pickle.dump(examples, f)
    print("保存成功")
    
