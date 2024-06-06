import numpy as np
import torch
import csv
xmin = 0
xmax = 0.0006
tmin = 0
tmax = 150
T_min = 300
T_max = 702

device  = torch.device("cuda"if torch.cuda.is_available() else "cpu")
def denormalize_output(y, y_min, y_max):
    return y * (y_max - y_min) + y_min

def predict_and_save(model,filename = 'prediction1.csv'):
    x_test = np.linspace(xmin,xmax,100)
    t_test = np.linspace(tmin,tmax,3001)

    x_test_grid ,t_test_grid = np.meshgrid(x_test,t_test)
    t_test_flat = t_test_grid.flatten()
    x_test_flat = x_test_grid.flatten()
    x_t_points = torch.tensor(np.vstack((x_test_flat,t_test_flat)).T,dtype = torch.float32).to(device)
    predictions = model(x_t_points).detach().cpu().numpy()
    T_pred = predictions[:,0].reshape(len(t_test),len(x_test))

    # 添加硬约束，T不能超过700
    T_pred = np.minimum(T_pred, 702)

    with open(filename,mode = 'w',newline='') as file:
        writer = csv.writer(file)
        writer.writerows(T_pred)
