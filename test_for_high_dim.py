import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from tqdm import tqdm

# ==========================================
# 1. 物理模型
# ==========================================
def f_phys_vector(D, theta):
    interaction = np.dot(D, theta)
    # 纯物理计算，不加 clip (相信数据生成是健康的)
    return 10 * np.exp(-interaction)

def grad_f_phys_vector(D, theta):
    Y_pred = f_phys_vector(D, theta)
    return -Y_pred[:, np.newaxis] * D

# ==========================================
# 2. 数据生成 (保持健康设定)
# ==========================================
def generate_data_pure(n=5000, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    p_theta = 3
    p_z = 3
    
    true_theta = np.array([0.3, 0.5, 0.7])
    
    Z = np.random.uniform(0, 1, (n, p_z))
    
    D = np.zeros((n, p_theta))
    for j in range(p_theta):
        # 保证 D 在合理正数范围内波动
        D[:, j] = np.abs(np.random.normal(0.8 + 1.0 * Z[:, j], 0.4, n))
    
    Y_phys = f_phys_vector(D, true_theta)
    
    # 混杂项 g(Z)
    Z_centered = (Z - 0.5) * 2
    g_Z = 1.0 * np.sum(Z_centered, axis=1) + 0.5 * np.sum(Z_centered**2 - 0.33, axis=1)
    
    U = np.random.normal(0, 0.5, n)
    Y = Y_phys + g_Z + U
    
    return Y, D, Z, true_theta

# ==========================================
# 3. Solver (纯净版: 无正则化)
# ==========================================
def solve_dml_pure(Y, D, Z, n_splits=2, max_iter=20):
    n, p = D.shape
    
    # 初始值
    theta = np.ones(p) * 0.4 
    
    # 普通 ML (随机森林)
    rf = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=10, n_jobs=-1)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    step_size = 0.8 
    
    for t in range(max_iter):
        M = f_phys_vector(D, theta)
        J = grad_f_phys_vector(D, theta)
        R = Y - M
        
        g_hat = np.zeros(n)
        h_hat = np.zeros((n, p))
        
        for train_idx, test_idx in kf.split(Y):
            Z_train, Z_test = Z[train_idx], Z[test_idx]
            R_train, J_train = R[train_idx], J[train_idx]
            
            rf.fit(Z_train, R_train)
            g_hat[test_idx] = rf.predict(Z_test)
            
            rf.fit(Z_train, J_train)
            h_hat[test_idx] = rf.predict(Z_test)
            
        Y_tilde = R - g_hat
        G_tilde = J - h_hat
        
        # === 核心修改: 纯 OLS 更新 ===
        GTG = G_tilde.T @ G_tilde
        GTY = G_tilde.T @ Y_tilde
        
        # 直接求解，没有任何 + eye(p) * lambda
        # 只要样本量 N >> p，且 D 不完全共线，这里就是安全的
        try:
            delta_theta = np.linalg.solve(GTG, GTY)
        except np.linalg.LinAlgError:
            # 万一矩阵真的奇异了 (极小概率)，给个提示
            print("Warning: Singular matrix encountered in pure OLS.")
            delta_theta = np.zeros(p)
        
        # 依然保留一点步长控制，这是牛顿法收敛的必要条件，不是正则化
        theta_new = theta + step_size * delta_theta
        
        # 物理约束 theta > 0 是必须的，否则指数函数没法算
        theta = np.maximum(theta_new, 0.01)
        
        if np.linalg.norm(delta_theta) < 1e-6:
            break
            
    # === 方差计算 (纯净版) ===
    J_hat = (G_tilde.T @ G_tilde) / n
    # 直接求逆
    J_inv = np.linalg.inv(J_hat) 
    
    weighted_G = G_tilde * Y_tilde[:, np.newaxis]
    Sigma_hat = (weighted_G.T @ weighted_G) / n
    
    Omega = J_inv @ Sigma_hat @ J_inv
    se = np.sqrt(np.diag(Omega / n))
    
    return theta, se

# ==========================================
# 4. 运行
# ==========================================
def run_experiment_pure():
    N_SIMS = 100
    N_SAMPLES = 5000
    P_THETA = 3
    
    print(f"🚀 纯净版 OLS DML 验证 (N={N_SAMPLES})...")
    
    t_stats_list = []
    estimates_list = []
    
    for i in tqdm(range(N_SIMS)):
        Y, D, Z, truth = generate_data_pure(n=N_SAMPLES, seed=i)
        theta_hat, se_hat = solve_dml_pure(Y, D, Z)

        print(f"Sim {i+1}/{N_SIMS} | Estimates: {theta_hat} | SE: {se_hat}")
        
        t = (theta_hat - truth) / se_hat
        t_stats_list.append(t)
        estimates_list.append(theta_hat)
        
    t_stats = np.array(t_stats_list)
    estimates = np.array(estimates_list)
    
    # 绘图
    fig, axes = plt.subplots(1, P_THETA, figsize=(15, 5), sharey=True)
    x = np.linspace(-4, 4, 100)
    norm_pdf = stats.norm.pdf(x, 0, 1)
    colors = ['#FF9999', '#66B2FF', '#99FF99']
    
    for j in range(P_THETA):
        ax = axes[j]
        data = t_stats[:, j]
        
        sns.histplot(data, stat="density", bins=15, kde=True, ax=ax,
                     color=colors[j], edgecolor='white', alpha=0.7)
        ax.plot(x, norm_pdf, 'k--', lw=2, label='N(0,1)')
        
        bias = np.mean(estimates[:, j]) - truth[j]
        coverage = np.mean(np.abs(data) < 1.96)
        
        ax.set_title(f"$\\theta_{j+1}$ (True={truth[j]})\nBias: {bias:.4f} | Cov: {coverage:.2f}")
        ax.set_xlabel("T-statistic")
        ax.set_xlim(-4, 4)
        ax.grid(True, alpha=0.3)
        if j == 0: ax.set_ylabel("Density")
            
    plt.suptitle(f"Pure OLS DML Validation (No Regularization)", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment_pure()