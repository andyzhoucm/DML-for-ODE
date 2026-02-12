import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.base import clone
from tqdm import tqdm

# ==========================================
# Part 1: 物理模型与数据生成 (保持不变)
# ==========================================

def f_phys(D, theta):
    return 10 * np.exp(-theta * D)

def grad_f_phys(D, theta):
    return -10 * D * np.exp(-theta * D)

def generate_data(n=1000, true_theta=0.5, seed=None):
    if seed is not None:
        np.random.seed(seed)
    Z = np.random.uniform(0, 3, n)
    D = np.abs(np.random.normal(1 + 0.5*Z, 0.5, n))
    g_Z = 2 * np.sin(Z * 2) + 1.5 * Z
    U = np.random.normal(0, 1, n)
    Y = f_phys(D, true_theta) + g_Z + U
    return Y, D, Z

# ==========================================
# Part 2: 核心算法 - Iterative DML (含方差计算)
# ==========================================

def solve_iterative_dml_with_inference(Y, D, Z, theta_init=0.1, n_splits=2, max_iter=15):
    """
    Iterative DML，返回参数估计值 theta 和标准误 se
    """
    theta = theta_init
    n = len(Y)
    
    # 使用较轻量的 RF 参数以加速实验
    rf_params = {'n_estimators': 30, 'max_depth': 5, 'min_samples_leaf': 10, 'n_jobs': 1}
    model_bias_base = RandomForestRegressor(**rf_params)
    model_grad_base = RandomForestRegressor(**rf_params)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # 记录最后一次迭代的残差用于算方差
    final_Y_tilde = None
    final_G_tilde = None

    # --- 1. 迭代更新 Theta ---
    for t in range(max_iter):
        theta_old = theta
        
        # 构造伪标签
        M = f_phys(D, theta)
        J = grad_f_phys(D, theta)
        R = Y - M 
        
        g_hat = np.zeros(n)
        h_hat = np.zeros(n)
        
        # Cross-Fitting
        for train_idx, test_idx in kf.split(Y):
            Z_train, Z_test = Z[train_idx], Z[test_idx]
            R_train, J_train = R[train_idx], J[train_idx]
            
            # 拟合 nuisance
            m_bias = clone(model_bias_base).fit(Z_train.reshape(-1, 1), R_train)
            m_grad = clone(model_grad_base).fit(Z_train.reshape(-1, 1), J_train)
            
            g_hat[test_idx] = m_bias.predict(Z_test.reshape(-1, 1))
            h_hat[test_idx] = m_grad.predict(Z_test.reshape(-1, 1))
            
        # 构造正交残差
        Y_tilde = R - g_hat
        G_tilde = J - h_hat
        
        # 更新参数 (Newton Step / OLS)
        num = np.dot(G_tilde, Y_tilde)
        den = np.dot(G_tilde, G_tilde)
        delta_theta = num / (den + 1e-8)
        
        theta = theta + delta_theta
        
        # 保存这一轮的残差 (如果收敛了，这就是我们要用的)
        final_Y_tilde = Y_tilde
        final_G_tilde = G_tilde
        
        # 收敛检查
        if np.abs(delta_theta) < 1e-6:
            break
    
    # --- 2. 基于最终残差计算方差 (Sandwich Formula) ---
    # 公式: Var(theta) = (1/N) * J^-1 * Sigma * J^-1
    # 对于标量 theta:
    # J_hat = mean(G_tilde^2)  (Bread)
    # Sigma_hat = mean(G_tilde^2 * Y_tilde^2) (Meat)
    
    J_hat = np.mean(final_G_tilde ** 2)
    Sigma_hat = np.mean((final_G_tilde ** 2) * (final_Y_tilde ** 2))
    
    # 渐近方差 Omega = J^-2 * Sigma
    Omega_hat = Sigma_hat / (J_hat ** 2)
    
    # 估计量的方差 Var(theta_hat) = Omega / N
    variance_theta = Omega_hat / n
    se = np.sqrt(variance_theta)
            
    return theta, se

# ==========================================
# Part 3: 运行蒙特卡洛实验并验证正态性
# ==========================================

def run_inference_experiment():
    print("开始蒙特卡洛推断实验 (100 次)...")
    np.random.seed(42) # 固定种子以便复现
    
    n_sims = 100
    true_theta = 0.5
    
    estimates = []
    standard_errors = []
    t_stats = []
    
    for i in tqdm(range(n_sims), desc="Simulation Progress"):
        # 1. 生成数据
        Y, D, Z = generate_data(n=8000, true_theta=true_theta, seed=i)
        
        # 2. DML 求解 (获取 theta 和 SE)
        theta_hat, se_hat = solve_iterative_dml_with_inference(Y, D, Z, theta_init=0.1)
        
        # 3. 计算 t-statistic (标准化统计量)
        # 理论上应服从 N(0, 1)
        t_stat = (theta_hat - true_theta) / se_hat
        
        estimates.append(theta_hat)
        standard_errors.append(se_hat)
        t_stats.append(t_stat)
    
    estimates = np.array(estimates)
    t_stats = np.array(t_stats)
    
    # ==========================================
    # Part 4: 绘图验证
    # ==========================================
    plt.figure(figsize=(10, 6))
    
    # 1. 绘制 t-statistics 的直方图和 KDE
    sns.histplot(t_stats, stat="density", bins=15, kde=True, 
                 color="skyblue", label="Empirical Distribution of $\\frac{\\hat{\\theta} - \\theta_0}{\\hat{SE}}$",
                 edgecolor='white')
    
    # 2. 绘制标准正态分布 N(0, 1) 的理论曲线
    x = np.linspace(-4, 4, 100)
    plt.plot(x, stats.norm.pdf(x, 0, 1), 'k--', linewidth=2.5, label="Standard Normal $\\mathcal{N}(0, 1)$")
    
    plt.title("Validity of DML Inference: T-statistic Distribution")
    plt.xlabel("Standardized T-statistic")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-4, 4)
    
    plt.show()
    
    # 打印覆盖率 (Coverage Rate)
    # 95% 置信区间理论上应该让 t-stat 落在 [-1.96, 1.96] 之间
    in_ci = np.mean(np.abs(t_stats) < 1.96)
    print("\n=== 推断结果统计 ===")
    print(f"真实 Theta: {true_theta}")
    print(f"估计均值: {np.mean(estimates):.4f}")
    print(f"平均标准误 (SE): {np.mean(standard_errors):.4f}")
    print(f"95% CI 覆盖率 (理论值 0.95): {in_ci:.2f}")

if __name__ == "__main__":
    run_inference_experiment()