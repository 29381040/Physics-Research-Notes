#%% 导入与全局设置
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "text.latex.preamble": r"\usepackage{newtxtext,newtxmath}",
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
})
custom_colors = [
    '#66b2c8',
    '#0095b7',
    '#ff7f7f',
    '#e63946',
]
from matplotlib.colors import LinearSegmentedColormap
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', custom_colors, N=256)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

script_dir = "/Users/huangyiding/Desktop/Python Figure/Altermagnet/Intro/spin_wave"
print(f"图片将保存到: {script_dir}")


#%% 模型参数

a = 1.0            #晶格常数
eta = 0.3          #J_0/J
delta = 0.5        #
n_segment = 500    #采样点数

# 图像尺寸：aspect_ratio = 宽度 / 高度
figure_width = 9.0
aspect_ratio = 1.7
figure_height = figure_width / aspect_ratio


#%% 构造动量路径
#(0,0) -> (pi,pi) -> (pi,-pi) -> (0,0)

def momentum_segment(start, end, number):
    """在两个高对称点之间进行线性插值。"""
    kx = np.linspace(start[0], end[0], number)
    ky = np.linspace(start[1], end[1], number)
    return np.column_stack((kx, ky))


Gamma = np.array([0.0, 0.0])
M_plus = np.array([np.pi / a, np.pi / a])
M_minus = np.array([np.pi / a, -np.pi / a])

path_1 = momentum_segment(Gamma, M_plus, n_segment)
path_2 = momentum_segment(M_plus, M_minus, n_segment)[1:]
path_3 = momentum_segment(M_minus, Gamma, n_segment)[1:]

k_path = np.vstack((path_1, path_2, path_3))

kx = k_path[:, 0]
ky = k_path[:, 1]

#横坐标取动量空间中的累计路径长度
dk = np.sqrt(np.diff(kx)**2 + np.diff(ky)**2)
path_coordinate = np.concatenate(([0.0], np.cumsum(dk)))

boundary_indices = [
    0,
    n_segment - 1,
    2 * n_segment - 2,
    3 * n_segment - 3,
]

boundary_positions = path_coordinate[boundary_indices]


#%% 计算AM自旋波色散

gamma_k = np.cos(kx * a) + np.cos(ky * a)

D_k = (
    2- eta * (2 - np.cos((kx + ky) * a) - np.cos((kx - ky) * a))
)

common_part = np.sqrt(np.maximum(D_k**2 - gamma_k**2, 0.0))

#手性劈裂项
chiral_splitting = ( 2 * eta * delta * np.sin(kx * a) * np.sin(ky * a) )

#纵轴单位为JS，即绘制 hbar*omega/(JS)
energy_left = 2 * (common_part + chiral_splitting)
energy_right = 2 * (common_part - chiral_splitting)


#%% 绘制自旋波激发劈裂

fig, ax = plt.subplots(figsize=(figure_width, figure_height))

#正自旋/左旋分支：深红色
ax.plot(
    path_coordinate,
    energy_left,
    color=custom_colors[3],
    linewidth=3.0,
    label=r'$\circlearrowleft\;\omega_{+}$ branch',
)

#反自旋/右旋分支：深蓝色
ax.plot(
    path_coordinate,
    energy_right,
    color=custom_colors[1],
    linewidth=2.7,
    label=r'$\circlearrowright\;\omega_{-}$ branch',
)

#标出高对称路径的分界位置
for position in boundary_positions[1:-1]:
    ax.axvline(
        position,
        color='black',
        linewidth=0.8,
        linestyle='--',
        alpha=0.55,
    )

ax.axhline(0, color='black', linewidth=0.8)

ax.set_xticks(boundary_positions)
ax.set_xticklabels([
    r'$(0,0)$',
    r'$(\pi,\pi)$',
    r'$(\pi,-\pi)$',
    r'$(0,0)$',
])
ax.set_xlabel(r'$(k_x a, k_y a)$', fontsize=24)

ax.set_ylabel(
    r'$\hbar\omega/JS$',
    fontsize=22,
)

ax.set_xlim(path_coordinate[0], path_coordinate[-1])
ax.set_ylim(bottom=0)

ax.tick_params(
    axis='both',
    which='both',
    direction='out',
    length=6,
    width=1.2,
)

ax.legend(
    frameon=False,
    fontsize=18,
    loc='upper center',
)

for spine in ax.spines.values():
    spine.set_linewidth(1.2)

plt.tight_layout()

plt.savefig(
    os.path.join(script_dir, 'AM_spin_wave_chiral_splitting.pdf'),
    dpi=600,
    bbox_inches='tight'
)

plt.show()
# %%
