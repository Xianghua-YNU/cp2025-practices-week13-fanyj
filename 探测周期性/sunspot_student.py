#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
太阳黑子周期性分析 - 完整实现

该程序用于分析太阳黑子数据，计算其主要周期，并进行可视化展示。
"""

import numpy as np
import matplotlib.pyplot as plt

def load_sunspot_data(url):
    """
    从本地文件读取太阳黑子数据
    
    参数:
        url (str): 本地文件路径
        
    返回:
        tuple: (years, sunspots) 年份和太阳黑子数
    """
    # 使用np.loadtxt读取数据，跳过标题行，通过usecols指定读取第2(年份)和3(太阳黑子数)列
    data = np.loadtxt(url, skiprows=1, usecols=(1, 2))
    years = data[:, 0]
    sunspots = data[:, 1]
    return years, sunspots

def plot_sunspot_data(years, sunspots):
    """
    绘制太阳黑子数据随时间变化图
    
    参数:
        years (numpy.ndarray): 年份数组
        sunspots (numpy.ndarray): 太阳黑子数数组
    """
    plt.figure(figsize=(12, 6))
    plt.plot(years, sunspots, 'b-', linewidth=1)
    plt.title('太阳黑子数随时间变化')
    plt.xlabel('年份')
    plt.ylabel('太阳黑子数')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compute_power_spectrum(sunspots):
    """
    计算太阳黑子数据的功率谱
    
    参数:
        sunspots (numpy.ndarray): 太阳黑子数数组
        
    返回:
        tuple: (frequencies, power) 频率数组和功率谱
    """
    # 执行傅里叶变换
    n = len(sunspots)
    fft_result = np.fft.fft(sunspots)
    
    # 计算功率谱 (只取正频率部分)
    power = np.abs(fft_result[:n//2])**2 / n
    frequencies = np.fft.fftfreq(n, d=1)[:n//2]  # 每月采样一次
    
    return frequencies, power

def plot_power_spectrum(frequencies, power):
    """
    绘制功率谱图
    
    参数:
        frequencies (numpy.ndarray): 频率数组
        power (numpy.ndarray): 功率谱数组
    """
    plt.figure(figsize=(12, 6))
    plt.plot(1/frequencies[1:], power[1:])  # 忽略零频率
    plt.xlabel('周期 (月)')
    plt.ylabel('功率')
    plt.title('太阳黑子数据功率谱')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def find_main_period(frequencies, power):
    """
    找出功率谱中的主周期
    
    参数:
        frequencies (numpy.ndarray): 频率数组
        power (numpy.ndarray): 功率谱数组
        
    返回:
        float: 主周期（月）
    """
    # 忽略零频率
    idx = np.argmax(power[1:]) + 1
    main_period = 1 / frequencies[idx]
    return main_period

def main():
    # 数据文件路径
    data = "sunspot_data.txt"
    
    # 1. 加载并可视化数据
    years, sunspots = load_sunspot_data(data)
    plot_sunspot_data(years, sunspots)
    
    # 2. 傅里叶变换分析
    frequencies, power = compute_power_spectrum(sunspots)
    plot_power_spectrum(frequencies, power)
    
    # 3. 确定主周期
    main_period = find_main_period(frequencies, power)
    print(f"\n太阳黑子周期的主周期: {main_period:.2f} 月")
    print(f"大约 {main_period/12:.2f} 年")

if __name__ == "__main__":
    main()
