import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from traintest.prediction import predict_and_save
from traintest.errors import calculate_errors
from network.PINNsnetwork import PINN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
xmin = 0
xmax = 0.006
tmin = 0
tmax = 150
hcfl = 120
lambda_eff = 0.0756  # 有效导热系数
rho0 = 1384
cp0 = 1420
keff = 0.0756
# 读取第一行的数据
file_path = 'output.csv'
data = pd.read_csv(file_path)
first_row_data = data.iloc[0, :]
first_row_np = first_row_data.to_numpy()
first_row_tensor = torch.tensor(first_row_np, dtype=torch.float32).unsqueeze(1).to(device)

half_column_data = data.iloc[1501, :]
self_column_np = half_column_data.to_numpy()
self_column_tensor = torch.tensor(self_column_np, dtype=torch.float32).unsqueeze(1).to(device)
# 归一化输入数据
def normalize_input(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)

# 反归一化输出数据
def denormalize_output(y, y_min, y_max):
    return y * (y_max - y_min) + y_min

# 读取数据
file_path = 'output.csv'
data = pd.read_csv(file_path).values

# 初始化网络
model = PINN().to(device)

# 初始条件
def initial_condition(x):
    T0 = 300
    msv = 0.1718
    keff = 0.13
    return torch.tensor(
        np.hstack((T0 * np.ones((x.shape[0], 1)), msv * np.ones((x.shape[0], 1)))),
        dtype=torch.float32).to(device)

# 边界条件
def boundary_condition_second_type(model, x):
    T = model(x)[:, 0:1]
    T_x = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T), create_graph=True)[0][:, 0:1]
    bc_loss = torch.abs(lambda_eff * T_x + 773.15 * hcfl - hcfl * 300).mean()
    return bc_loss

# 定义损失函数
def loss_function(model, x_ic, x_bc, x_interior, x_rightbc):
    # 初始条件
    ic_y = initial_condition(x_ic)
    ic_loss = torch.abs(model(x_ic)[:, 0] - ic_y[:, 0]).mean()
    ic_loss += torch.abs(model(x_ic)[:, 1] - ic_y[:, 1]).mean()
    # 边界条件
    bc_loss = boundary_condition_second_type(model, x_bc)

    left_T = model(x_rightbc)[:, 0:1]
    left_bc_loss = torch.abs(left_T[0, :] - first_row_tensor).mean()

    T_msv = model(x_interior)
    T = T_msv[:, 0:1]
    msv = T_msv[:, 1:2]

    T_x = torch.autograd.grad(T, x_interior, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    T_xx = torch.autograd.grad(T_x[:, 0], x_interior, grad_outputs=torch.ones_like(T_x[:, 0]), create_graph=True)[0][:, 0:1]
    T_t = torch.autograd.grad(T, x_interior, grad_outputs=torch.ones_like(T), create_graph=True)[0][:, 1:2]
    rho0_cp0 = rho0 * cp0
    hvap = 2.792 * 10**6 - 160 * T - 3.43 * T**2
    habs = 1.145465717959184e+06
    f = rho0_cp0 * T_t - lambda_eff * T_xx + (hvap + habs) * msv


    return torch.abs(f).mean() / 30000, bc_loss / 1000, ic_loss / 100, left_bc_loss / 3000

# 生成数据点
def generate_data_points():
    # 初始条件点
    x_ic = torch.tensor(np.linspace(xmin, xmax, 1000), dtype=torch.float32).unsqueeze(1).to(device)
    x_ic = torch.cat([x_ic, torch.zeros_like(x_ic)], dim=1)  # 时间 t = 0

    x_bc = torch.full((100, 1), 0, dtype=torch.float32, requires_grad=True).to(device)
    t_bc = normalize_input(torch.tensor(np.linspace(tmin, tmax, 100), dtype=torch.float32).unsqueeze(1).to(device), tmin, tmax)
    x_bc = torch.cat([x_bc, t_bc], dim=1)
    #内部PDE点
    x_interior = torch.tensor(np.random.rand(20000, 2), dtype=torch.float32, requires_grad=True).to(device)
    x_interior[:, 0] = normalize_input(x_interior[:, 0], xmin, xmax)  # 归一化 x 坐标
    x_interior[:, 1] = normalize_input(x_interior[:, 1], tmin, tmax)  # 归一化 t 坐标
    # 左边界点
    t_leftbc = normalize_input(torch.linspace(tmin, tmax, 3000, dtype=torch.float32, requires_grad=True).unsqueeze(1).to(device), tmin, tmax)
    x_leftbc = torch.tensor(np.zeros(3000), dtype=torch.float32, requires_grad=True).unsqueeze(1).to(device)
    x_leftbc = torch.cat([x_leftbc, t_leftbc], dim=1)


    return x_ic, x_bc, x_interior, x_leftbc

# 训练网络
def train(model, epochs=15000, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_history = []

    x_ic, x_bc, x_interior, x_leftbc = generate_data_points()

    with open('training_loss.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Loss'])

        for epoch in range(epochs):
            optimizer.zero_grad()
            loss1, loss2, loss3, loss4 = loss_function(model, x_ic, x_bc, x_interior, x_leftbc)
            loss1.backward(retain_graph=True)
            loss2.backward(retain_graph=True)
            loss3.backward(retain_graph=True)
            loss4.backward(retain_graph=True)
            loss = loss1 + loss2 + loss3 + loss4
            optimizer.step()

            if epoch % 1000 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')
                loss_history.append(loss.item())
                writer.writerow([epoch, loss.item()])

    plt.plot(loss_history)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

# 运行训练
train(model)

# 生成预测数据点并保存预测结果
predict_and_save(model)

# 计算并保存误差
calculate_errors('prediction1.csv', 'output.csv')
