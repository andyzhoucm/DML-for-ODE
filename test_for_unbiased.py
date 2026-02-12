import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.base import clone
from scipy.optimize import least_squares
from tqdm import tqdm

# ==========================================
# Part 1: 定义物理模型和数据生成过程
# ==========================================

def f_phys(D, theta):
    """
    物理模型 (Mechanism): 指数衰减模型
    Y = 10 * exp(-theta * D)
    """
    return 10 * np.exp(-theta * D)

def grad_f_phys(D, theta):
    """
    物理模型的梯度 (Jacobian)
    dY/dtheta = 10 * (-D) * exp(-theta * D)
    """
    return -10 * D * np.exp(-theta * D)

def generate_data(n=1000, true_theta=0.5, seed=None):
    """
    生成带混杂 (Confounding) 的数据
    Z -> D (Z 越大，D 越大)
    Z -> Y (Z 越大，Y 越大)
    如果不控制 Z，会误以为 D 导致 Y 没怎么衰减（低估 theta）
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 1. 干扰变量 Z
    Z = np.random.uniform(0, 3, n)
    
    # 2. Treatment D (受 Z 影响)
    # D = 1 + 0.5*Z + 随机噪音
    D = np.abs(np.random.normal(1 + 0.5*Z, 0.5, n))
    
    # 3. 干扰函数 g(Z) (非线性)
    g_Z = 2 * np.sin(Z * 2) + 1.5 * Z
    
    # 4. Outcome Y
    # Y = 物理规律 + 干扰 + 噪音
    U = np.random.normal(0, 1, n)
    Y = f_phys(D, true_theta) + g_Z + U
    
    return Y, D, Z

# ==========================================
# Part 2: 对比算法 1 - Naive NLS (反面教材)
# ==========================================

def solve_naive(Y, D):
    """
    直接做非线性最小二乘，完全忽略 Z。
    min sum (Y - f(D, theta))^2
    """
    def loss(theta):
        return Y - f_phys(D, theta[0])
    
    # 随便给个初值 0.1
    res = least_squares(loss, x0=[0.1])
    return res.x[0]

# ==========================================
# Part 3: 核心算法 - Iterative DML
# ==========================================

def solve_iterative_dml(Y, D, Z, theta_init=0.1, n_splits=2, max_iter=15):
    """
    Iterative DML for Non-linear Models
    """
    theta = theta_init
    n = len(Y)
    
    # 定义机器学习模型 (使用 Random Forest)
    # 限制深度防止过拟合
    rf_params = {'n_estimators': 50, 'max_depth': 5, 'min_samples_leaf': 10, 'n_jobs': -1}
    model_bias_base = RandomForestRegressor(**rf_params)
    model_grad_base = RandomForestRegressor(**rf_params)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # 迭代循环
    for t in range(max_iter):
        theta_old = theta
        
        # --- Step 2.1: 构造伪标签 (基于旧 theta) ---
        M = f_phys(D, theta)      # 物理预测
        J = grad_f_phys(D, theta) # 物理梯度
        R = Y - M                 # 物理模型解释不了的残差
        
        # 容器：存放 Cross-Fitting 的预测结果
        g_hat = np.zeros(n)
        h_hat = np.zeros(n)
        
        # --- Step 2.2: Cross-Fitting (核心) ---
        for train_idx, test_idx in kf.split(Y):
            # 切分数据
            Z_train, Z_test = Z[train_idx], Z[test_idx]
            R_train = R[train_idx]
            J_train = J[train_idx]
            
            # 训练 Bias Model: Z -> R
            # 注意：输入必须 reshape 成 (N, 1) 如果只有一个特征
            m_bias = clone(model_bias_base).fit(Z_train.reshape(-1, 1), R_train)
            g_hat[test_idx] = m_bias.predict(Z_test.reshape(-1, 1))
            
            # 训练 Gradient Model: Z -> J
            m_grad = clone(model_grad_base).fit(Z_train.reshape(-1, 1), J_train)
            h_hat[test_idx] = m_grad.predict(Z_test.reshape(-1, 1))
            
        # --- Step 2.3: 构造正交残差 ---
        Y_tilde = R - g_hat       # 去除了 Z 影响的结果残差
        G_tilde = J - h_hat       # 去除了 Z 影响的梯度残差
        
        # --- Step 2.4: 牛顿法更新 (相当于做线性回归) ---
        # minimize sum (Y_tilde - G_tilde * delta)^2
        # delta = (G^T G)^-1 G^T Y
        
        num = np.dot(G_tilde, Y_tilde)
        den = np.dot(G_tilde, G_tilde)
        
        # 加上一点点正则防止分母为0 (数值稳定性)
        delta_theta = num / (den + 1e-8)
        
        # 更新
        theta = theta + delta_theta
        
        # --- Step 2.5: 收敛检查 ---
        if np.abs(delta_theta) < 1e-6:
            break
            
    return theta

# ==========================================
# Part 4: 运行实验 (Monte Carlo Simulation)
# ==========================================

def run_experiment():
    print("开始蒙特卡洛模拟 (100 次)...")
    np.random.seed(123)
    
    n_sims = 100
    true_theta = 0.5
    
    res_naive = []
    res_dml = []
    
    for i in tqdm(range(n_sims), desc="蒙特卡洛模拟进度"):
        # 1. 生成数据
        Y, D, Z = generate_data(n=1000, true_theta=true_theta, seed=i)
        
        # 2. Naive 方法
        est_naive = solve_naive(Y, D)
        res_naive.append(est_naive)
        
        # 3. DML 方法
        est_dml = solve_iterative_dml(Y, D, Z, theta_init=0.1)
        res_dml.append(est_dml)
            
    # ==========================================
    # Part 5: 绘图
    # ==========================================
    plt.figure(figsize=(10, 6))
    
    # 绘制 Naive 分布
    sns.kdeplot(res_naive, fill=True, color='red', label='Naive NLS (Ignore Z)', alpha=0.3)
    
    # 绘制 DML 分布
    sns.kdeplot(res_dml, fill=True, color='blue', label='Iterative DML (With Z)', alpha=0.3)
    
    # 绘制真值
    plt.axvline(true_theta, color='black', linestyle='--', linewidth=2, label=f'True Theta ({true_theta})')
    
    plt.title('Monte Carlo Simulation: Naive vs Iterative DML')
    plt.xlabel('Estimated Theta')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 打印统计结果
    print("\n=== 实验结果统计 ===")
    print(f"真实值 (Truth): {true_theta}")
    print(f"Naive 均值: {np.mean(res_naive):.4f} (Bias: {np.mean(res_naive)-true_theta:.4f})")
    print(f"DML   均值: {np.mean(res_dml):.4f} (Bias: {np.mean(res_dml)-true_theta:.4f})")

# 运行主程序
if __name__ == "__main__":
    run_experiment()