#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拉格朗日点 - 地球-月球系统L1点位置计算

本模块实现了求解地球-月球系统L1拉格朗日点位置的数值方法。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# 物理常数
G = 6.674e-11  # 万有引力常数 (m^3 kg^-1 s^-2)
M = 5.974e24   # 地球质量 (kg)
m = 7.348e22   # 月球质量 (kg)
R = 3.844e8    # 地月距离 (m)
omega = 2.662e-6  # 月球角速度 (s^-1)


def lagrange_equation(r):
    """
    L1拉格朗日点位置方程
    
    参数:
        r (float): 从地心到L1点的距离 (m)
    
    返回:
        float: 方程左右两边的差值，当r是L1点位置时返回0
    """
    # 地球引力
    earth_gravity = G * M / r**2
    # 月球引力
    moon_gravity = G * m / (R - r)**2
    # 离心力
    centrifugal_force = omega**2 * r
    
    # 方程：地球引力 - 月球引力 = 离心力
    equation_value = earth_gravity - moon_gravity - centrifugal_force
    
    return equation_value


def lagrange_equation_derivative(r):
    """
    L1拉格朗日点位置方程的导数，用于牛顿法
    
    参数:
        r (float): 从地心到L1点的距离 (m)
    
    返回:
        float: 方程对r的导数值
    """
    # 地球引力项的导数: d/dr (G*M/r^2) = -2*G*M/r^3
    d_earth = -2 * G * M / r**3
    # 月球引力项的导数: d/dr (-G*m/(R-r)^2) = -2*G*m/(R-r)^3
    d_moon = -2 * G * m / (R - r)**3
    # 离心力项的导数: d/dr (omega^2*r) = omega^2
    d_centrifugal = omega**2
    
    # 方程的导数
    derivative_value = d_earth - d_moon - d_centrifugal
    
    return derivative_value


def newton_method(f, df, x0, tol=1e-8, max_iter=100):
    """
    使用牛顿法（切线法）求解方程f(x)=0
    
    参数:
        f (callable): 目标方程，形式为f(x)=0
        df (callable): 目标方程的导数
        x0 (float): 初始猜测值
        tol (float): 收敛容差
        max_iter (int): 最大迭代次数
    
    返回:
        tuple: (近似解, 迭代次数, 收敛标志)
    """
    x = x0
    converged = False
    
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        
        # 检查导数是否接近零
        if abs(dfx) < 1e-12:
            print(f"警告: 导数在迭代 {i} 时接近零")
            break
        
        # 牛顿法迭代公式
        x_next = x - fx / dfx
        
        # 检查收敛
        if abs(x_next - x) < tol:
            converged = True
            break
        
        x = x_next
    
    return x, i + 1, converged


def secant_method(f, a, b, tol=1e-8, max_iter=100):
    """
    使用弦截法求解方程f(x)=0
    
    参数:
        f (callable): 目标方程，形式为f(x)=0
        a (float): 区间左端点
        b (float): 区间右端点
        tol (float): 收敛容差
        max_iter (int): 最大迭代次数
    
    返回:
        tuple: (近似解, 迭代次数, 收敛标志)
    """
    x_prev = a
    x = b
    converged = False
    
    for i in range(max_iter):
        fx = f(x)
        fx_prev = f(x_prev)
        
        # 检查函数值是否接近零
        if abs(fx) < tol:
            converged = True
            break
        
        # 检查是否存在除零风险
        if abs(fx - fx_prev) < 1e-12:
            print(f"警告: 函数值在迭代 {i} 时过于接近")
            break
        
        # 弦截法迭代公式
        x_next = x - fx * (x - x_prev) / (fx - fx_prev)
        
        # 检查收敛
        if abs(x_next - x) < tol:
            converged = True
            break
        
        x_prev = x
        x = x_next
    
    return x, i + 1, converged


def plot_lagrange_equation(r_min, r_max, num_points=1000):
    """
    绘制L1拉格朗日点位置方程的函数图像
    
    参数:
        r_min (float): 绘图范围最小值 (m)
        r_max (float): 绘图范围最大值 (m)
        num_points (int): 采样点数
    
    返回:
        matplotlib.figure.Figure: 绘制的图形对象
    """
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 生成x轴数据
    r_values = np.linspace(r_min, r_max, num_points)
    
    # 计算对应的函数值
    f_values = np.array([lagrange_equation(r) for r in r_values])
    
    # 绘制函数曲线
    ax.plot(r_values, f_values, 'b-', label='拉格朗日方程')
    
    # 添加零水平线
    ax.axhline(y=0, color='r', linestyle='--', label='y=0')
    
    # 使用scipy求解精确解作为参考
    r_exact = optimize.fsolve(lagrange_equation, (r_min + r_max) / 2)[0]
    ax.axvline(x=r_exact, color='g', linestyle='--', label=f'精确解: {r_exact:.2e} m')
    
    # 添加标题和标签
    ax.set_title('地球-月球系统L1拉格朗日点方程')
    ax.set_xlabel('距离地球中心的距离 (m)')
    ax.set_ylabel('方程值')
    
    # 添加网格和图例
    ax.grid(True)
    ax.legend()
    
    # 设置y轴范围，更好地显示零点附近的情况
    y_min = min(f_values.min(), -1)
    y_max = max(f_values.max(), 1)
    ax.set_ylim(y_min, y_max)
    
    return fig


def main():
    """
    主函数，执行L1拉格朗日点位置的计算和可视化
    """
    # 1. 绘制方程图像，帮助选择初值
    r_min = 3.0e8  # 搜索范围下限 (m)，约为地月距离的80%
    r_max = 3.8e8  # 搜索范围上限 (m)，接近地月距离
    fig = plot_lagrange_equation(r_min, r_max)
    plt.savefig('lagrange_equation.png', dpi=300)
    plt.show()
    
    # 2. 使用牛顿法求解
    print("\n使用牛顿法求解L1点位置:")
    r0_newton = 3.5e8  # 初始猜测值 (m)，大约在地月距离的90%处
    r_newton, iter_newton, conv_newton = newton_method(lagrange_equation, lagrange_equation_derivative, r0_newton)
    if conv_newton:
        print(f"  收敛解: {r_newton:.8e} m")
        print(f"  迭代次数: {iter_newton}")
        print(f"  相对于地月距离的比例: {r_newton/R:.6f}")
    else:
        print("  牛顿法未收敛!")
    
    # 3. 使用弦截法求解
    print("\n使用弦截法求解L1点位置:")
    a, b = 3.2e8, 3.7e8  # 初始区间 (m)
    r_secant, iter_secant, conv_secant = secant_method(lagrange_equation, a, b)
    if conv_secant:
        print(f"  收敛解: {r_secant:.8e} m")
        print(f"  迭代次数: {iter_secant}")
        print(f"  相对于地月距离的比例: {r_secant/R:.6f}")
    else:
        print("  弦截法未收敛!")
    
    # 4. 使用SciPy的fsolve求解
    print("\n使用SciPy的fsolve求解L1点位置:")
    r0_fsolve = 3.5e8  # 初始猜测值 (m)
    r_fsolve = optimize.fsolve(lagrange_equation, r0_fsolve)[0]
    print(f"  收敛解: {r_fsolve:.8e} m")
    print(f"  相对于地月距离的比例: {r_fsolve/R:.6f}")
    
    # 5. 比较不同方法的结果
    if conv_newton and conv_secant:
        print("\n不同方法结果比较:")
        print(f"  牛顿法与弦截法的差异: {abs(r_newton-r_secant):.8e} m ({abs(r_newton-r_secant)/r_newton*100:.8f}%)")
        print(f"  牛顿法与fsolve的差异: {abs(r_newton-r_fsolve):.8e} m ({abs(r_newton-r_fsolve)/r_newton*100:.8f}%)")
        print(f"  弦截法与fsolve的差异: {abs(r_secant-r_fsolve):.8e} m ({abs(r_secant-r_fsolve)/r_secant*100:.8f}%)")


if __name__ == "__main__":
    main()
