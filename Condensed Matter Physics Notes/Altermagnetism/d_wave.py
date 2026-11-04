#%% 导入与全局设置
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib.ticker import FuncFormatter

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "text.latex.preamble": r"\usepackage{newtxtext,newtxmath}",
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
})
custom_colors = [
    '#0095b7',
    '#66b2c8',
    '#ff7f7f',
    '#e63946',
]
from matplotlib.colors import LinearSegmentedColormap
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', custom_colors, N=256)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

script_dir = "/Users/huangyiding/Desktop/Python Figure/Altermagnet/Intro/e_0+Delta"
print(f"图片将保存到: {script_dir}")


#%% 模型参数与二维能带
#晶格常数
a = 1.0

#最近邻跃迁
t = 1.0

#对角跃迁的自旋无关部分t_0=t_+ + t_-
t0 = 0.70

#交错磁性劈裂强度t_J=t_+ - t_-
tJ = 0.50

#费米能epsilon_F=0
epsilon_F = 0.0

#第一布里渊区每个方向的采样点数
N_k = 401

kxa = np.linspace(-np.pi, np.pi, N_k)
kya = np.linspace(-np.pi, np.pi, N_k)
KXA, KYA = np.meshgrid(kxa, kya)


def epsilon_0(kxa_value, kya_value):
    """正方晶格最近邻公共色散。"""
    return -2.0 * t * (np.cos(kxa_value) + np.cos(kya_value))


def epsilon_sigma(kxa_value, kya_value, sigma_z):
    """sigma_z=+1/-1表示上/下自旋能带。"""
    return (
        epsilon_0(kxa_value, kya_value)
        - 2.0 * t0 * np.cos(kxa_value) * np.cos(kya_value)
        + 2.0 * sigma_z * tJ * np.sin(kxa_value) * np.sin(kya_value)
    )


energy_up = epsilon_sigma(KXA, KYA, sigma_z=+1) / t
energy_down = epsilon_sigma(KXA, KYA, sigma_z=-1) / t
epsilon_F_plot = epsilon_F / t

#所有能带图共用能量范围，以便直接比较上下自旋
energy_min = min(energy_up.min(), energy_down.min())
energy_max = max(energy_up.max(), energy_down.max())

#上自旋统一使用红色系，下自旋统一使用蓝色系
up_color = '#e63946'
down_color = '#0095b7'
up_cmap = plt.get_cmap('Reds')
down_cmap = plt.get_cmap('Blues')

#%% epsilon_F=0时同时包含两种自旋的费米面
fig, ax = plt.subplots(figsize=(8.6, 8.0))

ax.contour(
    KXA,
    KYA,
    energy_up,
    levels=[epsilon_F_plot],
    colors=[up_color],
    linewidths=3.2,
)
ax.contour(
    KXA,
    KYA,
    energy_down,
    levels=[epsilon_F_plot],
    colors=[down_color],
    linewidths=3.2,
)

#用空数据线建立图例，避免 contour 图例样式不统一
ax.plot([], [], color=up_color, linewidth=3.2,
        label=r'spin-up $\uparrow$')
ax.plot([], [], color=down_color, linewidth=3.2,
        label=r'spin-down $\downarrow$')

ax.set_xlabel(r'$k_xa$', fontsize=24)
ax.set_ylabel(r'$k_ya$', fontsize=24)
ax.set_title(r'Fermi surface at $\varepsilon_F=0$', fontsize=24, pad=14)
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-np.pi, np.pi)
ax.set_aspect('equal')
ax.set_xticks(
    [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
    [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'],
)
ax.set_yticks(
    [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
    [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'],
)
ax.legend(fontsize=18, loc='upper right', frameon=False)
ax.grid(alpha=0.15)

plt.savefig(os.path.join(script_dir, 'altermagnetic_fermi_surface_EF0.pdf'), dpi=600, bbox_inches='tight')
plt.show()


#%% 沿菱形动量路径的能带截面
#截面路径：(0,0) -> (-pi/2,pi/2) -> (0,pi) -> (pi/2,pi/2) -> (0,0)
path_vertices = np.array([
    [0.0, 0.0],
    [-np.pi / 2.0, np.pi / 2.0],
    [0.0, np.pi],
    [np.pi / 2.0, np.pi / 2.0],
    [0.0, 0.0],
])

#截面路径采样点数
N_segment = 250

path_kx = []
path_ky = []
path_coordinate = []

for segment_index in range(len(path_vertices) - 1):
    start_point = path_vertices[segment_index]
    end_point = path_vertices[segment_index + 1]

    #最后一段保留终点，其余段不重复保存相邻顶点
    endpoint = segment_index == len(path_vertices) - 2
    interpolation = np.linspace(0.0, 1.0, N_segment, endpoint=endpoint)

    segment_points = (
        (1.0 - interpolation[:, None]) * start_point
        + interpolation[:, None] * end_point
    )
    path_kx.append(segment_points[:, 0])
    path_ky.append(segment_points[:, 1])
    path_coordinate.append(segment_index + interpolation)

path_kx = np.concatenate(path_kx)
path_ky = np.concatenate(path_ky)
path_coordinate = np.concatenate(path_coordinate)

section_up = epsilon_sigma(path_kx, path_ky, sigma_z=+1) / t
section_down = epsilon_sigma(path_kx, path_ky, sigma_z=-1) / t

fig, ax = plt.subplots(figsize=(14.5, 7.6))

ax.plot(path_coordinate, section_up, color=up_color, linewidth=4.0,
        label=r'spin-up $\uparrow$')
ax.plot(path_coordinate, section_down, color=down_color, linewidth=4.0,
        label=r'spin-down $\downarrow$')

#费米能 epsilon_F=0
fermi_dots_x = np.linspace(0.0, 4.0, 70)
ax.scatter(
    fermi_dots_x,
    np.full_like(fermi_dots_x, epsilon_F_plot),
    s=26,
    color='0.35',
    alpha=0.65,
    zorder=1,
)

#路径节点处的竖直辅助线
for boundary in [1.0, 2.0, 3.0]:
    ax.axvline(boundary, color='0.78', linewidth=1.0, linestyle='--', zorder=0)

path_labels = [
    r'$(0,0)$',
    r'$(-\pi/2,\pi/2)$',
    r'$(0,\pi)$',
    r'$(\pi/2,\pi/2)$',
    r'$(0,0)$',
]
ax.set_xticks(np.arange(5), path_labels)
ax.set_xlim(0.0, 4.0)
ax.set_ylim(energy_min - 0.25, energy_max + 0.25)
ax.set_xlabel(r'$(k_x a, k_y a)$', fontsize=24)
ax.set_ylabel(r'$\varepsilon/t$', fontsize=24)
ax.legend(fontsize=22, loc='upper center', ncol=2, frameon=False)
ax.tick_params(direction='out', length=8, width=1.8)
for spine in ax.spines.values():
    spine.set_linewidth(1.6)

#内嵌图：同时显示两种自旋的费米线和截面所取的菱形路径
ax_inset = ax.inset_axes([0.37, 0.10, 0.28, 0.45])
ax_inset.contour(
    KXA,
    KYA,
    energy_up,
    levels=[epsilon_F_plot],
    colors=[up_color],
    linewidths=2.4,
)
ax_inset.contour(
    KXA,
    KYA,
    energy_down,
    levels=[epsilon_F_plot],
    colors=[down_color],
    linewidths=2.4,
)

#用带箭头的黑色虚线标出截面方向
for start_point, end_point in zip(path_vertices[:-1], path_vertices[1:]):
    ax_inset.annotate(
        '',
        xy=end_point,
        xytext=start_point,
        arrowprops=dict(
            arrowstyle='->',
            color='black',
            linewidth=1.4,
            linestyle=(0, (4, 4)),
            shrinkA=3,
            shrinkB=3,
        ),
    )

ax_inset.set_xlim(-np.pi, np.pi)
ax_inset.set_ylim(-np.pi, np.pi)
ax_inset.set_aspect('equal')
ax_inset.set_xticks([-np.pi, 0, np.pi], [r'$-\pi$', r'$0$', r'$\pi$'])
ax_inset.set_yticks([-np.pi, 0, np.pi], [r'$-\pi$', r'$0$', r'$\pi$'])
ax_inset.xaxis.set_label_position('top')
ax_inset.xaxis.tick_top()
ax_inset.set_xlabel(r'$k_xa$', fontsize=18, labelpad=4)
ax_inset.set_ylabel(r'$k_ya$', fontsize=18, labelpad=3)
ax_inset.tick_params(labelsize=15, direction='in', length=4)
for spine in ax_inset.spines.values():
    spine.set_linewidth(1.4)

plt.savefig(os.path.join(script_dir, 'altermagnetic_band_cross_section.pdf'), dpi=600, bbox_inches='tight')
plt.show()

#%% 能带截面：(pi,0) -> (pi,pi) -> (0,0) -> (-pi,pi) -> (0,pi)

section_vertices = np.array([
    [np.pi, 0.0],
    [np.pi, np.pi],
    [0.0, 0.0],
    [-np.pi, np.pi],
    [0.0, np.pi],
])

N_section_segment = 300

section_kx_list = []
section_ky_list = []
section_coordinate_list = []

#路径节点在横坐标上的位置
section_tick_positions = [0.0]
cumulative_distance = 0.0

for segment_index in range(len(section_vertices) - 1):
    start_point = section_vertices[segment_index]
    end_point = section_vertices[segment_index + 1]

    #当前路径段在动量空间中的实际长度
    segment_length = np.linalg.norm(end_point - start_point)

    #除最后一段外不保存终点
    endpoint = segment_index == len(section_vertices) - 2
    interpolation = np.linspace(
        0.0,
        1.0,
        N_section_segment,
        endpoint=endpoint,
    )

    segment_points = (
        (1.0 - interpolation[:, None]) * start_point
        + interpolation[:, None] * end_point
    )

    section_kx_list.append(segment_points[:, 0])
    section_ky_list.append(segment_points[:, 1])

    section_coordinate_list.append(
        cumulative_distance + interpolation * segment_length
    )

    cumulative_distance += segment_length
    section_tick_positions.append(cumulative_distance)


#拼接完整动量路径
section_kx = np.concatenate(section_kx_list)
section_ky = np.concatenate(section_ky_list)
section_coordinate = np.concatenate(section_coordinate_list)

#计算上、下自旋能带
section_up = epsilon_sigma(
    section_kx,
    section_ky,
    sigma_z=+1,
) / t

section_down = epsilon_sigma(
    section_kx,
    section_ky,
    sigma_z=-1,
) / t


fig, ax = plt.subplots(figsize=(14.5, 7.5))

ax.plot(
    section_coordinate,
    section_down,
    color=down_color,
    linewidth=4.0,
    label=r'spin-down $\downarrow$',
    zorder=2,
)
ax.plot(
    section_coordinate,
    section_up,
    color=up_color,
    linewidth=3.7,
    label=r'spin-up $\uparrow$',
    zorder=3,
)

#费米能epsilon_F=0
fermi_dots_x = np.linspace(
    section_tick_positions[0],
    section_tick_positions[-1],
    85,
)

ax.scatter(
    fermi_dots_x,
    np.full_like(fermi_dots_x, epsilon_F_plot),
    s=28,
    color='0.35',
    alpha=0.65,
    zorder=1,
)

# 各段路径的分界处的竖直虚线
for boundary_position in section_tick_positions[1:-1]:
    ax.axvline(
        boundary_position,
        color='0.76',
        linewidth=1.1,
        linestyle='--',
        zorder=0,
    )

#路径节点
section_tick_labels = [
    r'$(\pi,0)$',
    r'$(\pi,\pi)$',
    r'$(0,0)$',
    r'$(\textrm{-}\pi,\pi)$',
    r'$(0,\pi)$',
]

ax.set_xticks(
    section_tick_positions,
    section_tick_labels,
)
ax.set_xlim(
    section_tick_positions[0],
    section_tick_positions[-1],
)

#自动设置纵轴
section_energy_min = min(section_up.min(), section_down.min())
section_energy_max = max(section_up.max(), section_down.max())

ax.set_ylim(
    section_energy_min - 0.25,
    section_energy_max + 1,
)
ax.yaxis.set_major_formatter(
    FuncFormatter(
        lambda value, _: (
            rf'$\textrm{{-}}{abs(value):g}$'
            if value < 0
            else rf'${value:g}$'
        )
    )
)

ax.set_xlabel(r'$(k_x a, k_y a)$', fontsize=24)
ax.set_ylabel(r'$\varepsilon/t$', fontsize=24)

ax.legend(
    fontsize=21,
    loc='upper center',
    ncol=2,
    frameon=False,
)
ax.tick_params(
    direction='out',
    length=8,
    width=1.8,
)
for spine in ax.spines.values():
    spine.set_linewidth(1.6)

plt.savefig(
    os.path.join(
        script_dir,
        'altermagnetic_band_section2.pdf',
    ),
    dpi=600,
    bbox_inches='tight',
)

plt.show()
# %%
