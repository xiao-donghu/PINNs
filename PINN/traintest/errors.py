import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd


def calculate_errors(prediction_file = 'output1.csv', actual_file = 'prediction1.csv', abs_error_file='absolute_errors.csv', rel_error_file='relative_errors.csv'):
    pred_data = pd.read_csv(prediction_file, header=None).to_numpy()
    actual_data = pd.read_csv(actual_file, header=None).to_numpy()

    abs_errors = np.abs(pred_data - actual_data)
    rel_errors = np.abs((pred_data - actual_data) / actual_data)

    np.savetxt(abs_error_file, abs_errors, delimiter=',')
    np.savetxt(rel_error_file, rel_errors, delimiter=',')

# 计算并保存误差
calculate_errors('prediction1.csv', 'output.csv')