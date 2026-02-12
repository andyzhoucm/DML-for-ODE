import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.base import clone
from tqdm import tqdm

# ==========================================
# 1. 物理模型：米氏方程 (Michaelis-Menten)
# ==========================================

def f_phys(Y, D, theta):
    # dY/dt = - (Y * D) / (D + theta)
    denom = D + theta + 1e-8
    return - (Y * D) / denom

def grad_f_phys(Y, D, theta):
    # d(f)/d(theta) = (Y * D) / (D + theta)^2
    denom = (D + theta + 1e-8) ** 2
    return (Y * D) / denom

# ==========================================
# 2. 数据生成器 (含混杂 + 噪音)
# ==========================================

def generate_ode_data(N=2000, true_theta=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    # Z: 混杂变量 (免疫力)
    Z = np.random.uniform(1.5, 3.5, N)
    
    # D: 治疗 (受 Z 影响，正相关)
    D = 0.8 * Z + np.random.normal(0, 0.1, N)
    
    # Y: 初始状态
    Y = np.random.uniform(5.0, 15.0, N)
    
    # 真实物理导数
    physics = f_phys(Y, D, true_theta)
    
    # 混杂对导数的影响 g(Z)
    nuisance = -0.3 * Z**2 
    
    # 总导数 = 物理 + 混杂 + 过程噪音
    # 注意：这里的噪音 U 是方差来源的一部分
    U = np.random.normal(0, 0.2, N) 
    Y_dot_true = physics + nuisance + U
    
    # 观测噪音 (加在 Y_dot 上模拟测量误差或导数估算误差)
    Y_dot_obs = Y_dot_true + np.random.normal(0, 0.05, N)
    
    return Y, Y_dot_obs, D, Z

# ==========================================
# 3. 带推断功能的 DML Solver
# ==========================================

def solve_ode_dml_inference(Y, Y_dot, D, Z, theta_init=0.5, n_splits=2, max_iter=7):
    theta = theta_init
    n = len(Y)
    
    # 随机森林参数
    rf = RandomForestRegressor(n_estimators=50, max_depth=5, min_samples_leaf=20, n_jobs=-1)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # 用于存储最后一次迭代的残差和梯度
    final_Y_tilde = None # 去偏导数残差
    final_G_tilde = None # 去偏梯度残差
    
    for k in range(max_iter):
        M = f_phys(Y, D, theta)
        J = grad_f_phys(Y, D, theta)
        R = Y_dot - M
        
        g_hat = np.zeros(n)
        h_hat = np.zeros(n)
        
        for train_idx, test_idx in kf.split(Y):
            Z_train, Z_test = Z[train_idx], Z[test_idx]
            R_train, J_train = R[train_idx], J[train_idx]
            
            m_bias = clone(rf).fit(Z_train.reshape(-1, 1), R_train)
            g_hat[test_idx] = m_bias.predict(Z_test.reshape(-1, 1))
            
            m_grad = clone(rf).fit(Z_train.reshape(-1, 1), J_train)
            h_hat[test_idx] = m_grad.predict(Z_test.reshape(-1, 1))
            
        # 正交化
        Y_tilde = R - g_hat
        G_tilde = J - h_hat
        
        # 保存用于方差计算
        final_Y_tilde = Y_tilde
        final_G_tilde = G_tilde
        
        # Gauss-Newton 更新
        num = np.dot(G_tilde, Y_tilde)
        den = np.dot(G_tilde, G_tilde)
        
        delta_theta = num / (den + 1e-8)
        theta = theta + 0.8 * delta_theta # Damping
        
        if theta < 0.01: theta = 0.01 # 物理约束
        
        if np.abs(delta_theta) < 1e-5:
            break
            
    # === 关键：方差推导 (Sandwich Formula) ===
    # J_hat (Bread): 梯度的二阶矩，代表信息量
    J_hat = np.mean(final_G_tilde ** 2)
    
    # Sigma_hat (Meat): 残差与梯度的乘积方差，代表噪音
    Sigma_hat = np.mean((final_Y_tilde ** 2) * (final_G_tilde ** 2))
    
    # Asymptotic Variance Omega = J^-2 * Sigma
    Omega = Sigma_hat / (J_hat ** 2)
    
    # Standard Error = sqrt(Omega / N)
    se = np.sqrt(Omega / n)
    
    return theta, se

# ==========================================
# 4. 运行蒙特卡洛实验
# ==========================================

def run_inference_validation():
    TRUE_THETA = 1.0
    N_SAMPLES = 2000 # 样本量足够大以保证渐近正态性
    N_SIMS = 200     # 模拟次数 (建议 >100 以画出平滑的直方图)
    
    t_stats = []
    estimates = []
    
    print(f"🚀 正在验证 ODE 模型的统计推断 (N={N_SAMPLES}, Sims={N_SIMS})...")
    
    for i in tqdm(range(N_SIMS)):
        # 1. 生成数据
        Y, Y_dot, D, Z = generate_ode_data(N=N_SAMPLES, true_theta=TRUE_THETA, seed=i)
        
        # 2. DML 求解 (获取 theta 和 se)
        theta_hat, se_hat = solve_ode_dml_inference(Y, Y_dot, D, Z, theta_init=0.5)
        
        # 3. 计算 t-statistic
        # t = (Estimate - Truth) / SE
        t = (theta_hat - TRUE_THETA) / se_hat
        
        t_stats.append(t)
        estimates.append(theta_hat)
        
    t_stats = np.array(t_stats)
    
    # ==========================================
    # 5. 绘图 (你要求的代码)
    # ==========================================
    plt.figure(figsize=(10, 6))
    
    # 1. 绘制 t-statistics 的直方图和 KDE
    sns.histplot(t_stats, stat="density", bins=20, kde=True, 
                 color="skyblue", label=r"Empirical Distribution of $\frac{\hat{\theta} - \theta_0}{\hat{SE}}$",
                 edgecolor='white', alpha=0.6)
    
    # 2. 绘制标准正态分布 N(0, 1) 的理论曲线
    x = np.linspace(-4, 4, 100)
    plt.plot(x, stats.norm.pdf(x, 0, 1), 'k--', linewidth=2.5, label=r"Standard Normal $\mathcal{N}(0, 1)$")
    
    plt.title(f"Validity of ODE DML Inference (N={N_SAMPLES})", fontsize=14)
    plt.xlabel("Standardized T-statistic", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(-4, 4)
    
    plt.show()
    
    # --- 打印统计验证 ---
    # 计算覆盖率 (Coverage Rate)
    # 理论上 95% 的 t-stat 应该落在 [-1.96, 1.96] 之间
    coverage = np.mean(np.abs(t_stats) < 1.96)
    
    print("\n" + "="*40)
    print(f"真实参数: {TRUE_THETA}")
    print(f"估计均值: {np.mean(estimates):.4f}")
    print("-" * 40)
    print(f"95% CI 覆盖率 (目标 0.95): {coverage:.3f}")
    print(f"T-统计量均值 (目标 0.0):  {np.mean(t_stats):.3f}")
    print(f"T-统计量方差 (目标 1.0):  {np.var(t_stats):.3f}")
    print("="*40)

if __name__ == "__main__":
    run_inference_validation()