import numpy as np
import pandas as pd

# 定义每个区间的长度
x_intervals_lengths = 0.6
y_interval_length = 150

# 在每个区间上均匀生成100个点
num_x_points = 100
x_points = np.linspace(0,x_intervals_lengths,num_x_points)

# 在 y 轴上均匀生成3000个点
num_y_points = 3000
y_points = np.linspace(0, y_interval_length, num_y_points)

# 归一化到0-1范围内
x_points_normalized = x_points / max(x_points)
y_points_normalized = y_points / max(y_points)
#打印完整信息
#np.set_printoptions(threshold=np.inf)
    #选取随机索引
for k in range(16):
    #选取的随机点索引
    random_x_indices = np.random.choice(len(x_points_normalized[:90]), size=(128, 128))
    random_y_indices = np.random.choice(len(y_points_normalized)-1, size=(128, 128))+1

    # 选取的随机点坐标
    random_x_points = x_points_normalized[random_x_indices]
    random_y_points = y_points_normalized[random_y_indices]
    indices_x, indices_y = np.indices((128, 128))

    df1 = pd.read_csv("output.csv")
    output_values = df1.values[random_y_indices, random_x_indices]
    expanded_output_values = output_values.reshape(1, 128, 128)

    output = expanded_output_values
    print(expanded_output_values)



    stacked_array = np.stack((random_x_points,random_y_points),axis=0)
    inputs = stacked_array



    np.savez(f'testdata{k+4}.npz', inputs=inputs, output=output)
    print("NPZ 文件已生成。")
# 打印每个选定点的坐标