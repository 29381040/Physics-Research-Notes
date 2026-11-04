#%% 导入与全局设置
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
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

script_dir = "/Users/huangyiding/Desktop/Python Figure/Altermagnet/Intro/Hubbard"
print(f"图片将保存到: {script_dir}")


#%% 模型参数与色散关系

#取 t=1
t = 1.0
t0 = 0.6 * t
tJ_ratio = 0.9
tJ = tJ_ratio * t0
U = 3.5 * t
m = 0.209

#化学势（同时作为费米能）
mu = 0.44254 * t
E_F = mu

#若希望只预览类似论文 Fig. 1 的金属费米口袋，
#可以暂时改为以下参数，但这不再是题目指定的 m=0.35：
#m = 0.32
#E_F = 0.06 * t

#令晶格常数a=1
a = 1.0

#同自旋采用同一色系；+ 支深色，- 支浅色
red_dark = "#f50e21"
red_light = "#faa5a5"
blue_dark = "#007ad6"
blue_light = "#7ac3d7"


def dispersion(kx, ky, zeta_s):
    """
    返回指定自旋的正负两支能带。

    zeta_s = +1：spin-up
    zeta_s = -1：spin-down
    """
    f_k = np.cos(kx * a) * np.cos(ky * a)

    epsilon_0 = -2.0 * t * (
        np.cos(kx * a) + np.cos(ky * a)
    )

    Delta_k = np.sin(kx * a) * np.sin(ky * a)

    band_center = -2.0 * t0 * f_k

    band_splitting = np.sqrt(
        epsilon_0**2
        + (2.0 * tJ * Delta_k + zeta_s * U * m)**2
    )

    epsilon_plus = band_center + band_splitting
    epsilon_minus = band_center - band_splitting

    return epsilon_plus, epsilon_minus

#%% (0,0) -> (-pi/2,pi/2) -> (0,pi) -> (pi/2,pi/2) -> (0,0)路径能带与费米面

#B.Z.路径
high_symmetry_points = np.array([
    [0.0, 0.0],
    [-np.pi / 2.0, np.pi / 2.0],
    [0.0, np.pi],
    [np.pi / 2.0, np.pi / 2.0],
    [0.0, 0.0],
])

#采样点数
N_segment = 400

kx_path = []
ky_path = []
path_coordinate = []
tick_positions = [0.0]

current_distance = 0.0

for i in range(len(high_symmetry_points) - 1):
    start = high_symmetry_points[i]
    end = high_symmetry_points[i + 1]

    u = np.linspace(0.0, 1.0, N_segment, endpoint=True)

    kx_segment = start[0] + u * (end[0] - start[0])
    ky_segment = start[1] + u * (end[1] - start[1])

    segment_length = np.linalg.norm(end - start)
    coordinate_segment = current_distance + u * segment_length

    kx_path.append(kx_segment)
    ky_path.append(ky_segment)
    path_coordinate.append(coordinate_segment)

    current_distance += segment_length
    tick_positions.append(current_distance)

kx_path = np.concatenate(kx_path)
ky_path = np.concatenate(ky_path)
path_coordinate = np.concatenate(path_coordinate)

#上、下自旋的两支能带
E_up_plus, E_up_minus = dispersion(kx_path, ky_path, +1)
E_down_plus, E_down_minus = dispersion(kx_path, ky_path, -1)

#能量基准平移E_F
E_up_plus = (E_up_plus - E_F) / t
E_up_minus = (E_up_minus - E_F) / t
E_down_plus = (E_down_plus - E_F) / t
E_down_minus = (E_down_minus - E_F) / t


fig, ax = plt.subplots(figsize=(13.0, 7.8))

#spin-up
ax.plot(
    path_coordinate, E_up_plus,
    color=red_dark, linewidth=5.0,
    label=r'spin-up $\uparrow$, $+$ branch'
)
ax.plot(
    path_coordinate, E_up_minus,
    color=red_light, linewidth=5.0,
    label=r'spin-up $\uparrow$, $-$ branch'
)

#spin-down
ax.plot(
    path_coordinate, E_down_plus,
    color=blue_dark, linewidth=5.0,
    label=r'spin-down $\downarrow$, $+$ branch'
)
ax.plot(
    path_coordinate, E_down_minus,
    color=blue_light, linewidth=5.0,
    label=r'spin-down $\downarrow$, $-$ branch'
)

#费米能E_F=___
ax.axhline(
    y=0.0,
    color='0.45',
    linewidth=4.8,#圆点大小
    linestyle=(0, (0.1, 2)),#第二个数越大，圆点间距越大
    dash_capstyle='round', #端点变成圆形
    alpha=0.65,
    zorder=1
)

#竖直辅助线
for position in tick_positions:
    ax.axvline(
        position,
        color='0.82',
        linewidth=1.0,
        zorder=0
    )

tick_labels = [
    r'$(0,0)$',
    r'$(-\pi/2,\pi/2)$',
    r'$(0,\pi)$',
    r'$(\pi/2,\pi/2)$',
    r'$(0,0)$',
]

ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, fontsize=18)

ax.set_xlim(tick_positions[0], tick_positions[-1])
ax.set_ylim(-6.1, 3.5)

ax.set_xlabel(r'$(k_x a, k_y a)$', fontsize=24)
ax.set_ylabel(r'$(\varepsilon-E_F)/t$', fontsize=24)

ax.legend(
    loc='upper center',
    ncol=2,
    fontsize=13,
    frameon=False,
    handlelength=2.8,
    columnspacing=1.5
)

ax.tick_params(direction='out', length=7, width=1.4)
for spine in ax.spines.values():
    spine.set_linewidth(1.4)


#嵌入费米面

#插图[左边界, 下边界, 宽度, 高度]
axins = ax.inset_axes([0.28, 0.02, 0.44, 0.44])

#
fermi_display_limit = 1.04 * np.pi

#费米面网格密度
N_fermi = 801

kx_fermi = np.linspace(
    -fermi_display_limit,
    fermi_display_limit,
    N_fermi
)
ky_fermi = np.linspace(
    -fermi_display_limit,
    fermi_display_limit,
    N_fermi
)

KX, KY = np.meshgrid(kx_fermi, ky_fermi)

EF_up_plus, EF_up_minus = dispersion(KX, KY, +1)
EF_down_plus, EF_down_minus = dispersion(KX, KY, -1)

#B.Z.边界#
diamond_x = np.array([
    0.0, np.pi, 0.0, -np.pi, 0.0
])
diamond_y = np.array([
    np.pi, 0.0, -np.pi, 0.0, np.pi
])

#背景灰色
diamond_x = np.array([0.0, np.pi, 0.0, -np.pi, 0.0])
diamond_y = np.array([np.pi, 0.0, -np.pi, 0.0, np.pi])

axins.fill(
    diamond_x, diamond_y,
    color='#b9bfd0',
    alpha=0.78,
    zorder=0
)


#费米面：spin-up
axins.contour(
    KX,
    KY,
    EF_up_plus,
    levels=[E_F],
    colors=[red_dark],
    linewidths=2.7,
    zorder=4
)

axins.contour(
    KX,
    KY,
    EF_up_minus,
    levels=[E_F],
    colors=[red_light],
    linewidths=2.7,
    zorder=4
)

#spin-down
axins.contour(
    KX,
    KY,
    EF_down_plus,
    levels=[E_F],
    colors=[blue_dark],
    linewidths=2.7,
    zorder=4
)

axins.contour(
    KX,
    KY,
    EF_down_minus,
    levels=[E_F],
    colors=[blue_light],
    linewidths=2.7,
    zorder=4
)

# 黑色辅助线：(0,0) -> (-pi/2,pi/2) -> (0,pi)-> (pi/2,pi/2) -> (0,0)示意图
path_points = np.array([
    [0.0,           0.0],
    [-np.pi / 2.0,  np.pi / 2.0],
    [0.0,           np.pi],
    [np.pi / 2.0,   np.pi / 2.0],
    [0.0,           0.0]
])
#方向箭头
for start_point, end_point in zip(
    path_points[:-1],
    path_points[1:]
):
    arrow_start = (
        start_point
        + 0.45 * (end_point - start_point)
    )
    arrow_end = (
        start_point
        + 0.55 * (end_point - start_point)
    )

    axins.annotate(
        '',
        xy=arrow_end,
        xytext=arrow_start,
        arrowprops=dict(
            arrowstyle='->',
            color='black',
            linewidth=1.5,
            mutation_scale=10,   
            shrinkA=0,
            shrinkB=0
        ),
        zorder=8
    )

#B.Z.路径
axins.plot(
    path_points[:, 0],
    path_points[:, 1],
    color='black',
    linewidth=1.5,
    linestyle='--',
    solid_capstyle='round',
    solid_joinstyle='round',
    alpha = 0.8,
    zorder=6
)


#坐标轴设置

axins.set_xlim(
    -fermi_display_limit,
    fermi_display_limit
)
axins.set_ylim(
    -fermi_display_limit,
    fermi_display_limit
)
axins.set_aspect('equal')

axins.set_xticks([
    -np.pi, 0.0, np.pi
])
axins.set_yticks([
    -np.pi, 0.0, np.pi
])

axins.set_xticklabels(
    [r'$-\pi$', r'$0$', r'$\pi$'],
    fontsize=14
)
axins.set_yticklabels(
    [r'$-\pi$', r'$0$', r'$\pi$'],
    fontsize=14
)

axins.set_xlabel(
    r'$k_xa$',
    fontsize=16,
    labelpad=7
)
axins.set_ylabel(
    r'$k_ya$',
    fontsize=16,
    labelpad=-6
)

axins.xaxis.set_label_position('top')
axins.xaxis.tick_top()

axins.tick_params(
    direction='out',
    length=4,
    width=1.0,
    pad=1
)

for spine in axins.spines.values():
    spine.set_linewidth(1.2)

plt.tight_layout()
plt.savefig(
    os.path.join(
        script_dir,
        'altermagnetic_band_and_fermi_surface.pdf'
    ),
    dpi=600,
    bbox_inches='tight'
)
plt.show()

#%% 三维连续渐变能带面与完整费米面

# 三维曲面的网格密度；增大可使曲面更加光滑
N_surface = 301

# 能带面延伸至第一布里渊区之外；坐标范围再多留一圈空白缓冲
surface_plot_limit = 1.25 * np.pi
surface_display_limit = 1.38 * np.pi
kx_surface = np.linspace(
    -surface_plot_limit,
    surface_plot_limit,
    N_surface
)
ky_surface = np.linspace(
    -surface_plot_limit,
    surface_plot_limit,
    N_surface
)

KX3D, KY3D = np.meshgrid(kx_surface, ky_surface)

# spin-up 能带用于绘制能带面
E3D_up_plus, E3D_up_minus = dispersion(KX3D, KY3D, +1)

# spin-down 能带用于补齐费米面
E3D_down_plus, E3D_down_minus = dispersion(KX3D, KY3D, -1)

# 能量零点移至 mu，并用 t 作为单位
E3D_up_plus = (E3D_up_plus - E_F) / t
E3D_up_minus = (E3D_up_minus - E_F) / t
E3D_down_plus = (E3D_down_plus - E_F) / t
E3D_down_minus = (E3D_down_minus - E_F) / t

# 只使用上面设定的颜色：+ 能带为红色渐变，- 能带为蓝色渐变
plus_cmap = LinearSegmentedColormap.from_list(
    'plus_red_cmap',
    [red_light, red_dark],
    N=256
)
minus_cmap = LinearSegmentedColormap.from_list(
    'minus_blue_cmap',
    [blue_dark, blue_light],
    N=256
)

# 两支分别归一化，使红、蓝两套渐变都能完整显示
plus_norm = Normalize(
    vmin=np.min(E3D_up_plus),
    vmax=np.max(E3D_up_plus)
)
minus_norm = Normalize(
    vmin=np.min(E3D_up_minus),
    vmax=np.max(E3D_up_minus)
)

plus_facecolors = plus_cmap(
    plus_norm(E3D_up_plus)
)
minus_facecolors = minus_cmap(
    minus_norm(E3D_up_minus)
)


# 清掉交互环境中残留的旧图窗和旧图例
plt.close('all')

# 直接控制显示窗口的真实尺寸与坐标区位置；这对 plt.show() 和导出都生效
fig = plt.figure(figsize=(8.6, 5.4))
ax = fig.add_axes(
    [0.105, 0.015, 0.635, 0.97],
    projection='3d'
)

# + 支：红色渐变
ax.plot_surface(
    KX3D,
    KY3D,
    E3D_up_plus,
    facecolors=plus_facecolors,
    rstride=1,
    cstride=1,
    linewidth=0,
    antialiased=True,
    shade=False,
    alpha=0.97
)

# - 支：蓝色渐变
ax.plot_surface(
    KX3D,
    KY3D,
    E3D_up_minus,
    facecolors=minus_facecolors,
    rstride=1,
    cstride=1,
    linewidth=0,
    antialiased=True,
    shade=False,
    alpha=0.95
)

# mu 对应 z=0 的半透明平面
ax.plot_surface(
    KX3D,
    KY3D,
    np.zeros_like(KX3D),
    color='0.35',
    alpha=0.10,
    linewidth=0,
    shade=False
)

# 在 mu=0.44254t 平面上绘制完整费米面。
# spin-up 用红色，spin-down 用蓝色；+ 支深色，- 支浅色。
#fermi_surface_z = 0.015
#fermi_contours = (
#    (E3D_up_plus, red_dark),
#    (E3D_up_minus, red_light),
#    (E3D_down_plus, blue_dark),
#    (E3D_down_minus, blue_light),
#)

#for fermi_energy, fermi_color in fermi_contours:
#    ax.contour(
#        KX3D,
#        KY3D,
#        fermi_energy,
#        levels=[0.0],
#        zdir='z',
#        offset=fermi_surface_z,
#        colors=[fermi_color],
#        linewidths=3.0
#    )

# 第一磁性布里渊区边界投影到费米平面，扩展区域仍保留以展示闭合口袋
#ax.plot(
#    diamond_x,
#    diamond_y,
#    np.full_like(diamond_x,fermi_surface_z),
#    color='0.30',
#    linestyle='--',
#    linewidth=1.4,
#    alpha=0.75
#)

ax.set_xlabel(r'$k_xa$', fontsize=23, labelpad=14)
ax.set_ylabel(r'$k_ya$', fontsize=23, labelpad=14)

# 使用整张画布的坐标固定能量轴标题，与 z 轴刻度保留独立间距
ax.set_zlabel('')
fig.text(
    0.018,
    0.50,
    r'$(\varepsilon-\mu)/t$',
    rotation=90,
    va='center',
    ha='center',
    fontsize=23
)

ax.set_xlim(-surface_display_limit, surface_display_limit)
ax.set_ylim(-surface_display_limit, surface_display_limit)

ax.set_xticks([-np.pi, 0.0, np.pi])
ax.set_yticks([-np.pi, 0.0, np.pi])

ax.set_xticklabels(
    [r'$-\pi$', r'$0$', r'$\pi$']
)
ax.set_yticklabels(
    [r'$-\pi$', r'$0$', r'$\pi$']
)

# 三维观察角度：
# elev 控制俯视角，azim 控制绕 z 轴旋转角
ax.view_init(elev=0, azim=-115)

# 控制三维盒子的长宽高比例
# 缩小横向投影、保持竖向高度，避免三维框与左侧色条重合
ax.set_box_aspect((1.0, 1.0, 0.78), zoom=1.28)

# 两条色标分别使用用户设定的红、蓝色系
plus_mappable = ScalarMappable(
    norm=plus_norm,
    cmap=plus_cmap
)
plus_mappable.set_array([])

# 色条使用独立坐标区，不再让 fig.colorbar 自动挤压三维图
plus_colorbar_ax = fig.add_axes([0.885, 0.16, 0.022, 0.68])
plus_colorbar = fig.colorbar(
    plus_mappable,
    cax=plus_colorbar_ax
)
plus_colorbar.set_label(
    r'$+$ branch, $(\varepsilon-\mu)/t$',
    fontsize=17
)
plus_colorbar.ax.tick_params(labelsize=14)

minus_mappable = ScalarMappable(
    norm=minus_norm,
    cmap=minus_cmap
)
minus_mappable.set_array([])

minus_colorbar_ax = fig.add_axes([0.755, 0.16, 0.022, 0.68])
minus_colorbar = fig.colorbar(
    minus_mappable,
    cax=minus_colorbar_ax
)
minus_colorbar.set_label(
    r'$-$ branch, $(\varepsilon-\mu)/t$',
    fontsize=17
)
minus_colorbar.ax.tick_params(labelsize=14)

ax.tick_params(
    axis='both',
    which='major',
    labelsize=17,
    pad=2
)

# 确保三维坐标轴上没有残留图例；颜色条不受影响
if ax.get_legend() is not None:
    ax.get_legend().remove()

plt.savefig(
    os.path.join(
        script_dir,
        'altermagnetic_3D_band_and_fermisphere_15.pdf'
    ),
    dpi=600,
    facecolor='white'
)

plt.show()

# %%
