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
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', custom_colors, N=256)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

script_dir = "/Users/huangyiding/Desktop/Python Figure/Altermagnet/Intro/p-wave"
print(f"图片将保存到: {script_dir}")


#%% 模型参数与色散关系

t = 1.0                   # 能量单位
t_J = 0.4 * t             # 题目给定的自旋相关跃迁强度
a = 1.0                   # 晶格常数
mu = -1.1 * t              # 化学势；该值给出清晰的闭合费米面

COLOR_PLUS = custom_colors[3]     # sigma_z=+1：深红色
COLOR_MINUS = custom_colors[1]    # sigma_z=-1：深蓝色

# 色散关系中的三个有效参数
t_x = t #/ 4.0 - 3.0 * t_J**2 / (8.0 * t)
t_y = t #+ t_J**2 / (2.0 * t)
J = t_J**2 / t

print(f"t_x/t = {t_x/t:.2f}")
print(f"t_y/t = {t_y/t:.2f}")
print(f"J/t   = {J/t:.2f}")


def energy(kx, ky, sigma_z):
    """
    严格对应：
    epsilon_sigma(k)
      = t_x cos(k_x a)
      + t_y cos(k_y a)
      - J sigma_z sin(k_x a) cos(k_y a)

    sigma_z = +1：正自旋
    sigma_z = -1：负自旋
    """
    return (
        -t_x * np.cos(kx * a)
        -t_y * np.cos(ky * a)
        - J * sigma_z
        * np.sin(kx * a)
        * np.cos(ky * a)
    )

#%% 路径能带线图

n_segment = 400     # 每段路径的采样数；越大则曲线越光滑

path_vertices = np.array([
    [0.0, 0.0],
    [np.pi, np.pi],
    [-np.pi, np.pi],
    [0.0, 0.0],
])


def construct_k_path(vertices, number):
    k_path_parts = []
    path_coordinate_parts = []
    node_positions = [0.0]

    accumulated_length = 0.0

    for i in range(len(vertices) - 1):
        start = vertices[i]
        end = vertices[i + 1]

        interpolation = np.linspace(0.0, 1.0, number)

        segment = (
            start[None, :]
            + interpolation[:, None] * (end - start)[None, :]
        )

        segment_length = np.linalg.norm(end - start)

        segment_coordinate = (
            accumulated_length
            + interpolation * segment_length
        )

        # 相邻两段共享的端点只保留一次
        if i > 0:
            segment = segment[1:]
            segment_coordinate = segment_coordinate[1:]

        k_path_parts.append(segment)
        path_coordinate_parts.append(segment_coordinate)

        accumulated_length += segment_length
        node_positions.append(accumulated_length)

    return (
        np.vstack(k_path_parts),
        np.concatenate(path_coordinate_parts),
        np.array(node_positions),
    )


k_path, path_s, node_s = construct_k_path(
    path_vertices,
    n_segment,
)

kx_path = k_path[:, 0]
ky_path = k_path[:, 1]

E_path_plus = energy(kx_path, ky_path, +1) / t
E_path_minus = energy(kx_path, ky_path, -1) / t


fig, ax = plt.subplots(figsize=(11.5, 6.2))

ax.plot(
    path_s,
    E_path_minus,
    color=COLOR_MINUS,
    linewidth=2.5,
    label=r"spin-down $\downarrow$",
)

ax.plot(
    path_s,
    E_path_plus,
    color=COLOR_PLUS,
    linewidth=2.5,
    label=r"spin-up $\uparrow$",
)

# 高对称点处的竖直虚线
for position in node_s[1:-1]:
    ax.axvline(
        position,
        color="0.65",
        linestyle="--",
        linewidth=1.0,
        alpha=0.65,
        zorder=0,
    )

# 用85个灰色散点标出化学势
fermi_x = np.linspace(node_s[0], node_s[-1], 85)

ax.scatter(
    fermi_x,
    np.full_like(fermi_x, mu / t),
    s=28,
    color="0.35",
    alpha=0.65,
    edgecolors="none",
    zorder=1,
    label=rf"$\mu={mu/t:.1f}t$",
)

ax.set_xlim(node_s[0], node_s[-1])
ax.set_ylim(-2.5, 3)

ax.set_xticks(node_s)
ax.set_xticklabels([
    r"$(0,0)$",
    r"$(\pi,\pi)$",
    r"$(-\pi,\pi)$",
    r"$(0,0)$",
])

ax.set_xlabel(r"$(k_xa,k_ya)$", fontsize=22)
ax.set_ylabel(r"$\varepsilon/t$", fontsize=22)

ax.tick_params(
    axis="both",
    direction="out",
    length=5,
    width=1.0,
)

ax.legend(
    fontsize=15,
    frameon=False,
    loc="lower center",
    ncol=3,
    columnspacing=1.8,
)

for spine in ax.spines.values():
    spine.set_linewidth(1.1)
    spine.set_color("0.25")

plt.tight_layout()

plt.savefig(
    os.path.join(script_dir, 'p_wave_band_path_cos.pdf'),
    dpi=600,
    bbox_inches='tight',
)

plt.show()

#%% 第一Brillouin区内的费米面

n_k = 800     # 动量网格密度；增大后费米面轮廓更加光滑

kx = np.linspace(-np.pi / a, np.pi / a, n_k)
ky = np.linspace(-np.pi / a, np.pi / a, n_k)

KX, KY = np.meshgrid(kx, ky)

E_plus = energy(KX, KY, +1) / t
E_minus = energy(KX, KY, -1) / t


fig, ax = plt.subplots(figsize=(7.5, 7.0))

# 正自旋费米面：epsilon_+(k)=mu
ax.contour(
    KX * a,
    KY * a,
    E_plus,
    levels=[mu / t],
    colors=[COLOR_PLUS],
    linewidths=2.8,
    linestyles="solid",
)

# 负自旋费米面：epsilon_-(k)=mu
ax.contour(
    KX * a,
    KY * a,
    E_minus,
    levels=[mu / t],
    colors=[COLOR_MINUS],
    linewidths=2.8,
    linestyles="solid",
)

# 空曲线用于生成图例
ax.plot(
    [],
    [],
    color=COLOR_PLUS,
    linewidth=2.8,
    label=r"spin-up $\uparrow$",
)

ax.plot(
    [],
    [],
    color=COLOR_MINUS,
    linewidth=2.8,
    label=r"spin-down $\downarrow$",
)

ax.grid(
    True,
    color="0.75",
    linewidth=0.7,
    alpha=0.25,
)

ax.set_xlim(-np.pi/2-0.1, np.pi/2+0.1)
ax.set_ylim(-np.pi/2-0.1, np.pi/2+0.1)
ax.set_aspect("equal")

momentum_ticks = [
    -np.pi,
    -np.pi / 2.0,
    0.0,
    np.pi / 2.0,
    np.pi,
]

momentum_labels = [
    r"$-\pi$",
    r"$-\pi/2$",
    r"$0$",
    r"$\pi/2$",
    r"$\pi$",
]

ax.set_xticks(momentum_ticks)
ax.set_xticklabels(momentum_labels)

ax.set_yticks(momentum_ticks)
ax.set_yticklabels(momentum_labels)

ax.set_xlabel(r"$k_xa$", fontsize=22)
ax.set_ylabel(r"$k_ya$", fontsize=22)

ax.tick_params(
    axis="both",
    direction="out",
    length=5,
    width=1.0,
)

ax.legend(
    fontsize=15,
    frameon=False,
    loc="upper right",
)

for spine in ax.spines.values():
    spine.set_linewidth(1.1)
    spine.set_color("0.25")

plt.tight_layout()

plt.savefig(
    os.path.join(script_dir, 'p_wave_fermi_surface_cos.pdf'),
    dpi=600,
    bbox_inches='tight',
)

plt.show()