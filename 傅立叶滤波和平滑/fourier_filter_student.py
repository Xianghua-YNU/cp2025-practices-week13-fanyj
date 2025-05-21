#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
傅立叶滤波和平滑 - 道琼斯工业平均指数分析

本模块实现了对道Jones工业平均指数数据的傅立叶分析和滤波处理。
"""

import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    """
    加载道Jones工业平均指数数据
    
    参数:
        filename (str): 数据文件路径
    
    返回:
        numpy.ndarray: 指数数组
    """
    try:
        data = np.loadtxt(filename, delimiter=',', usecols=(1,))
        return data
    except Exception as e:
        print(f"数据加载失败: {e}")
        raise

def plot_data(data, title="Dow Jones Industrial Average"):
    """
    绘制时间序列数据
    
    参数:
        data (numpy.ndarray): 输入数据数组
        title (str): 图表标题
    
    返回:
        None
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data)
    plt.title(title)
    plt.xlabel("时间索引")
    plt.ylabel("道琼斯指数值")
    plt.grid(True)
    plt.show()

def fourier_filter(data, keep_fraction=0.1):
    """
    执行傅立叶变换并滤波
    
    参数:
        data (numpy.ndarray): 输入数据数组
        keep_fraction (float): 保留的傅立叶系数比例
    
    返回:
        tuple: (滤波后的数据数组, 原始傅立叶系数数组)
    """
    fft_coeff = np.fft.rfft(data)
    n_coeffs = len(fft_coeff)
    keep_count = int(keep_fraction * n_coeffs)
    
    # 创建滤波器系数
    filtered_coeff = np.zeros_like(fft_coeff)
    filtered_coeff[:keep_count] = fft_coeff[:keep_count]
    
    # 执行逆变换
    filtered_data = np.fft.irfft(filtered_coeff)
    return filtered_data, fft_coeff

def plot_comparison(original, filtered, title="Fourier Filter Result"):
    """
    绘制原始数据和滤波结果的比较
    
    参数:
        original (numpy.ndarray): 原始数据数组
        filtered (numpy.ndarray): 滤波后的数据数组
        title (str): 图表标题
    
    返回:
        None
    """
    plt.figure(figsize=(12, 6))
    plt.plot(original, label='原始数据', alpha=0.5)
    plt.plot(filtered, label='滤波数据', linewidth=2)
    plt.title(title)
    plt.xlabel("时间索引")
    plt.ylabel("道琼斯指数值")
    plt.legend()
    plt.grid(True)
    plt.show()
def calculate_y_values(self, x_values, energy, potential_func):
    """
    计算给定能量下薛定谔方程的波函数值
    
    参数:
        x_values: x坐标数组
        energy: 能量值
        potential_func: 势能函数
        
    返回:
        对应x值的波函数值数组
    """
    # 常数设置
    h_bar = 1.0  # 简化值，实际应用中可能需要更精确
    mass = 1.0   # 简化值
    
    # 计算势能
    V = potential_func(x_values)
    
    # 计算薛定谔方程的解
    y_values = np.sqrt(2.0 / (h_bar**2 * np.pi**2)) * np.sin(np.sqrt(2 * mass * (energy - V)) * x_values / h_bar)
    
    return y_values
    def find_energy_level_bisection(self, potential_func, x_range, n_points, energy_guess, tolerance=1e-6, max_iterations=100, is_ground_state=True):
    """
    使用二分法寻找给定势能的能级
    
    参数:
        potential_func: 势能函数
        x_range: x范围，如[x_min, x_max]
        n_points: 采样点数
        energy_guess: 初始能量猜测
        tolerance: 容忍度
        max_iterations: 最大迭代次数
        is_ground_state: 是否寻找基态
        
    返回:
        找到的能级值
    """
    x_min, x_max = x_range
    x = np.linspace(x_min, x_max, n_points)
    
    # 定义判别函数，用于判断能量是否太低
def is_energy_too_low(energy):
        y = self.calculate_y_values(x, energy, potential_func)
        # 对于基态，波函数在边界处应该接近零
        if is_ground_state:
            return (abs(y[0]) > tolerance or abs(y[-1]) > tolerance) and energy < 0
        else:
            # 对于激发态，波函数在至少一个边界处应该改变符号
            return (np.sign(y[0]) == np.sign(y[-1])) and energy < 0
    
    # 二分法寻找能级
    low = energy_guess
    high = energy_guess + 100  # 假设初始上界足够大
    
    for _ in range(max_iterations):
        mid = (low + high) / 2
        if is_energy_too_low(mid):
            low = mid
        else:
            high = mid
        
        if high - low < tolerance:
            break
    
    return (low + high) / 2
def plot_energy_functions(self, x_range, n_points, energies, potential_func, title="Energy Eigenstates"):
    """
    绘制不同能量的薛定谔方程波函数
    
    参数:
        x_range: x范围，如[x_min, x_max]
        n_points: 采样点数
        energies: 能量列表
        potential_func: 势能函数
        title: 图表标题
    """
    x = np.linspace(x_range[0], x_range[1], n_points)
    plt.figure(figsize=(10, 6))
    
    for i, energy in enumerate(energies):
        y = self.calculate_y_values(x, energy, potential_func)
        # 归一化波函数
        y = y / np.sqrt(np.trapz(y**2, x))
        plt.plot(x, y, label=f'E={energy:.2f}')
    
    plt.title(title)
    plt.xlabel('Position (x)')
    plt.ylabel('Wavefunction (ψ)')
    plt.legend()
    plt.grid(True)
    
    return plt.gcf()  # 返回图形对象
def main():
    # 任务1：数据加载与可视化
    data = load_data('dow.txt')
    plot_data(data, "道琼斯工业平均指数 - 原始数据")
    
    # 任务2：傅立叶变换与滤波（保留前10%系数）
    filtered_10, coeff = fourier_filter(data, 0.1)
    plot_comparison(data, filtered_10, "傅立叶滤波结果 (保留前10%系数)")
    
    # 任务3：修改滤波参数（保留前2%系数）
    filtered_2, _ = fourier_filter(data, 0.02)
    plot_comparison(data, filtered_2, "傅立叶滤波结果 (保留前2%系数)")

if __name__ == "__main__":
    main()
