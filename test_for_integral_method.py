import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from scipy import stats
from tqdm import tqdm

# ==========================================
# 1. 物理层：可微积分器 (Differentiable RK4)
# ==========================================
class ODESolver(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, y0, D, dt, theta):
        """
        RK4 积分器：计算从 t 到 t+dt 的状态演化
        y_next = y_curr + Integral(f(y, D, theta))
        """
        # k1
        k1 = self.func(y0, D, theta)
        # k2
        k2 = self.func(y0 + 0.5 * dt * k1, D, theta)
        # k3
        k3 = self.func(y0 + 0.5 * dt * k2, D, theta)
        # k4
        k4 = self.func(y0 + dt * k3, D, theta)
        
        y_next = y0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return y_next

def physics_model(y, D, theta):
    """
    物理方程：Michaelis-Menten 动力学
    dy/dt = - (Y * D) / (D + theta)
    """
    # 防止分母为 0 加一个小 epsilon
    denom = D + theta + 1e-6
    return - (y * D) / denom

# ==========================================
# 2. 数据生成：时变混杂 (Time-Varying Z)
# ==========================================
def generate_longitudinal_data(N=200, T=10, true_theta=1.0, dt=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    Y_list, D_list, Z_list = [], [], []

    # 真实的滋扰函数
    def true_g(z):
        return 0.5 * z + np.sin(z)

    # 物理方程 (用于数据生成)
    def true_physics_grad(y, d, theta):
        denom = d + theta + 1e-6
        return - (y * d) / denom

    for i in range(N):
        # 1. 生成 Z(t)
        base_z = np.random.uniform(1.0, 3.0)
        time_trend = np.linspace(0, 1, T+1)
        Z_t = base_z + 0.5 * np.sin(2 * np.pi * time_trend) + np.random.normal(0, 0.1, T+1)
        
        # 2. 生成 D(t)
        D_t = 0.5 * Z_t + np.random.normal(0.5, 0.1, T+1)
        
        # 3. 高精度积分生成 Y(t)
        y_traj = [10.0 + np.random.normal(0, 0.5)] 
        
        for t in range(T):
            y_curr = y_traj[-1]
            z_curr = Z_t[t]
            d_curr = D_t[t]
            
            # --- 核心修正：使用 RK4 生成数据 ---
            # 我们把 dynamics = physics + nuisance 看作一个整体
            
            def combined_dynamics(y_val, _d, _z):
                # _d 和 _z 在 dt 间隔内近似常数，或者你可以插值
                f_phys = true_physics_grad(y_val, _d, true_theta)
                f_nuis = true_g(_z)
                return f_phys + f_nuis

            # 手写 RK4 step
            k1 = combined_dynamics(y_curr, d_curr, z_curr)
            k2 = combined_dynamics(y_curr + 0.5*dt*k1, d_curr, z_curr)
            k3 = combined_dynamics(y_curr + 0.5*dt*k2, d_curr, z_curr)
            k4 = combined_dynamics(y_curr + dt*k3, d_curr, z_curr)
            
            dy = (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            # 加上观测噪声
            y_next = y_curr + dy + np.random.normal(0, 0.05)
            
            y_traj.append(y_next)
            
        Y_list.append(y_traj)
        D_list.append(D_t)
        Z_list.append(Z_t)

    return (
        torch.tensor(Y_list, dtype=torch.float32),
        torch.tensor(D_list, dtype=torch.float32),
        torch.tensor(Z_list, dtype=torch.float32)
    )
# ==========================================
# 3. 核心算法：带早停的 Integral DML
# ==========================================
class IntegralDML:
    def __init__(self, dt):
        self.dt = dt
        self.solver = ODESolver(physics_model)
        
    def fit_inference(self, Y, D, Z, n_splits=2, max_iter=20, tol=1e-4):
        """
        参数:
        - max_iter: 最大迭代次数 (比如 20)
        - tol: 收敛阈值 (比如 1e-4)
        """
        N, T_plus_1 = Y.shape
        T = T_plus_1 - 1
        
        # 数据拍扁 (Pooling)
        # 预测 Y(t+1) from Y(t)
        Y_curr_flat = Y[:, :-1].reshape(-1, 1)
        Y_next_flat = Y[:, 1:].reshape(-1, 1)
        D_curr_flat = D[:, :-1].reshape(-1, 1)
        Z_curr_flat = Z[:, :-1].reshape(-1, 1)
        
        # 记录受试者 ID 用于交叉验证 (防止 Time-leakage)
        subj_ids = np.repeat(np.arange(N), T)
        
        # 初始化参数
        theta_est = torch.tensor([0.5], requires_grad=True)
        
        final_epsilon = None
        final_G_tilde = None
        
        # 迭代优化循环
        for k in range(max_iter):
            # --- Step 1: 物理预测 & 雅可比计算 ---
            if theta_est.grad is not None: theta_est.grad.zero_()
            
            # 积分预测: Y_pred = Phi(Y_t, D_t, theta)
            Y_pred = self.solver(Y_curr_flat, D_curr_flat, self.dt, theta_est)
            
            # 计算雅可比 (梯度): J = d(Phi)/d(theta)
            # 这里用有限差分 (Finite Difference) 保证数值稳定性
            # 你也可以用 torch.autograd.grad，但在标量参数下 FD 往往更稳
            with torch.no_grad():
                delta_fd = 1e-4
                Y_pred_eps = self.solver(Y_curr_flat, D_curr_flat, self.dt, theta_est + delta_fd)
                J_raw = ((Y_pred_eps - Y_pred) / delta_fd).numpy().flatten()
            
            # 原始残差 R (包含 nuisance * dt)
            R_raw = (Y_next_flat - Y_pred).detach().numpy().flatten()
            
            # --- Step 2: 交叉拟合 (Cross-Fitting) ---
            g_hat_all = np.zeros_like(R_raw)
            h_hat_all = np.zeros_like(J_raw)
            
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42 + k) # 每次迭代随机种子变一下
            unique_subjs = np.unique(subj_ids)
            
            Z_numpy = Z_curr_flat.detach().numpy()
            
            for train_subj, val_subj in kf.split(unique_subjs):
                # 关键：按 Subject 切分 Mask
                train_mask = np.isin(subj_ids, unique_subjs[train_subj])
                val_mask = np.isin(subj_ids, unique_subjs[val_subj])
                
                # ML 1: 学习 Z -> Residual (估计累积漂移)
                # 树的数量不用太多，防止过拟合
                ml_g = RandomForestRegressor(n_estimators=20, max_depth=5, min_samples_leaf=10, n_jobs=-1)
                ml_g.fit(Z_numpy[train_mask], R_raw[train_mask])
                g_hat_all[val_mask] = ml_g.predict(Z_numpy[val_mask])
                
                # ML 2: 学习 Z -> Jacobian (估计梯度的条件期望)
                ml_h = RandomForestRegressor(n_estimators=20, max_depth=5, min_samples_leaf=10, n_jobs=-1)
                ml_h.fit(Z_numpy[train_mask], J_raw[train_mask])
                h_hat_all[val_mask] = ml_h.predict(Z_numpy[val_mask])
            
            # --- Step 3: 正交化 (Orthogonalization) ---
            epsilon_tilde = R_raw - g_hat_all
            G_tilde = J_raw - h_hat_all
            
            # 保存最后一步用于算方差
            final_epsilon = epsilon_tilde
            final_G_tilde = G_tilde
            
            # --- Step 4: 高斯-牛顿更新 & 终止条件 ---
            num = np.dot(G_tilde, epsilon_tilde)
            den = np.dot(G_tilde, G_tilde)
            
            # 计算步长
            delta_theta = num / (den + 1e-8)
            
            # === 核心修改：早停检查 ===
            if np.abs(delta_theta) < tol:
                # 调试时可以取消注释下面这行查看在第几次收敛
                # print(f"Converged at iter {k+1} with delta {delta_theta:.2e}")
                break
            
            # 更新参数 (带阻尼 0.8 防止震荡)
            new_val = theta_est.item() + 0.8 * delta_theta
            
            # 物理约束: theta 必须为正
            if new_val < 0.01: new_val = 0.01 
            
            theta_est = torch.tensor([new_val], requires_grad=True)
            
        # --- Step 5: 推断 (Sandwich Formula) ---
        n_obs = len(final_epsilon)
        
        # J_hat (Bread): 信息矩阵
        J_hat = np.mean(final_G_tilde ** 2)
        # Sigma_hat (Meat): 得分方差
        Sigma_hat = np.mean((final_epsilon ** 2) * (final_G_tilde ** 2))
        
        # 渐近方差 Omega
        Omega = Sigma_hat / (J_hat ** 2)
        se = np.sqrt(Omega / n_obs)
        
        return theta_est.item(), se

# ==========================================
# 4. 蒙特卡洛实验运行
# ==========================================
def run_simulation():
    TRUE_THETA = 1.0
    N_SIMS = 50         # 模拟次数
    N_SUBJECTS = 500    # 样本量
    T_STEPS = 10        # 时间步长
    DT = 0.1
    
    t_stats = []
    estimates = []
    
    print(f"🚀 Running Integral DML with Early Stopping (Sims={N_SIMS})...")
    
    # 实例化 Solver
    dml = IntegralDML(dt=DT)
    
    for i in tqdm(range(N_SIMS)):
        # 1. 生成数据
        Y, D, Z = generate_longitudinal_data(N=N_SUBJECTS, T=T_STEPS, true_theta=TRUE_THETA, dt=DT, seed=i)
        
        # 2. 拟合 (设置 max_iter=20, tol=1e-4)
        theta_hat, se = dml.fit_inference(Y, D, Z, max_iter=20, tol=1e-4)
        
        print(f"Sim {i+1}/{N_SIMS}: Theta_hat={theta_hat:.4f}, SE={se:.4f}")

        # 3. 统计
        t = (theta_hat - TRUE_THETA) / se
        t_stats.append(t)
        estimates.append(theta_hat)
        
    t_stats = np.array(t_stats)
    
    # ==========================================
    # 5. 绘图与结果
    # ==========================================
    plt.figure(figsize=(10, 6))
    
    # 直方图 + KDE
    sns.histplot(t_stats, stat="density", bins=15, kde=True, 
                 color="skyblue", label=r"Empirical Distribution",
                 edgecolor='white', alpha=0.6)
    
    # 标准正态分布参考线
    x = np.linspace(-4, 4, 100)
    plt.plot(x, stats.norm.pdf(x, 0, 1), 'k--', linewidth=2.5, label=r"Standard Normal $\mathcal{N}(0, 1)$")
    
    plt.title(f"Validity of Trajectory-Based Integral DML\n(Time-Varying Z, Early Stopping Enabled)", fontsize=14)
    plt.xlabel("Standardized T-statistic", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(-4, 4)
    plt.show()

    # 打印统计指标
    coverage = np.mean(np.abs(t_stats) < 1.96)
    print("\n" + "="*40)
    print(f"True Theta: {TRUE_THETA}")
    print(f"Mean Estimate: {np.mean(estimates):.4f}")
    print("-" * 40)
    print(f"95% CI Coverage (Target 0.95): {coverage:.3f}")
    print(f"T-stat Mean (Target 0.0):      {np.mean(t_stats):.3f}")
    print(f"T-stat Var (Target 1.0):       {np.var(t_stats):.3f}")
    print("="*40)

if __name__ == "__main__":
    run_simulation()