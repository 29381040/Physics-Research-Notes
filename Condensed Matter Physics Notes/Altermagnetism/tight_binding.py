#%% 导入与全局设置
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib.transforms import Bbox

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
from matplotlib.colors import LinearSegmentedColormap, Normalize

#四个颜色在完整能量范围内均匀渐变
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', custom_colors, N=256)
color_norm = Normalize(vmin=-4, vmax=4)


def tex_number(value, number_format='g'):
    """避开 newtxmath 在部分 PDF 后端中丢失数学负号的问题。"""
    sign = r'\mbox{-}' if value < 0 else ''
    return rf'${sign}{format(abs(value), number_format)}$'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

script_dir = "/Users/huangyiding/Desktop/Python Figure/Altermagnet/Intro/e_0"
print(f"图片将保存到: {script_dir}")

#%% energy band
t = 1.0          # 最近邻跃迁强度
a = 1.0          # 晶格常数
N = 300          # k空间网格数

# 使用无量纲变量 Kx = k_x a, Ky = k_y a
Kx = np.linspace(-np.pi, np.pi, N)
Ky = np.linspace(-np.pi, np.pi, N)
KX, KY = np.meshgrid(Kx, Ky)

# 能带：epsilon(k) = -2t(cos k_x a + cos k_y a)
epsilon = -2 * t * (np.cos(KX) + np.cos(KY))

#%% 绘图
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(
    KX,
    KY,
    epsilon / t,
    cmap=custom_cmap,
    norm=color_norm,
    linewidth=0,
    edgecolor='none',
    antialiased=False,
    alpha=0.95
)

ax.set_xlabel(r'$k_x a$', fontsize=24, labelpad=12)
ax.set_ylabel(r'$k_y a$', fontsize=24, labelpad=12)
ax.set_zlabel(r'$\varepsilon(\mathbf{k})/t$', fontsize=24, labelpad=12)

ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax.set_xticklabels([r'$\mbox{-}\pi$', r'$\mbox{-}\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
ax.set_yticklabels([r'$\mbox{-}\pi$', r'$\mbox{-}\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
ax.set_zticks([-4, -2, 0, 2, 4])
ax.set_zticklabels([tex_number(value) for value in [-4, -2, 0, 2, 4]])

ax.view_init(elev=20, azim=45)  #elev俯仰角，azim水平旋转角
ax.set_box_aspect(None, zoom=1.10)

cbar = fig.colorbar(surf, ax=ax, shrink=0.72, pad=0.035)
cbar.set_label(r'$\varepsilon/t$', fontsize=22)
cbar_ticks = cbar.get_ticks()
cbar.set_ticks(cbar_ticks, labels=[tex_number(value) for value in cbar_ticks])

fig.subplots_adjust(left=0.06, right=0.92, bottom=0.06, top=0.98)
fig.canvas.draw()
tight_bbox = fig.get_tightbbox(fig.canvas.get_renderer())
cropped_bbox = Bbox.from_extents(
    tight_bbox.x0,
    tight_bbox.y0,
    tight_bbox.x1,
    tight_bbox.y1 - 0.42,
)
plt.savefig(
    os.path.join(script_dir, 'square_lattice_band_3d.pdf'),
    dpi=600,
    bbox_inches=cropped_bbox,
    pad_inches=0.04,
)
plt.show()


#%% Fermi sphere
t = 1.0          #最近邻跃迁强度
a = 1.0          #晶格常数
N = 500          #k空间网格数

Kx = np.linspace(-np.pi, np.pi, N)
Ky = np.linspace(-np.pi, np.pi, N)
KX, KY = np.meshgrid(Kx, Ky)

epsilon = -2 * t * (np.cos(KX) + np.cos(KY))

#%% 绘图
fig, ax = plt.subplots(figsize=(7, 7))

#整个能带在二维Brillouin区中的能量分布
im = ax.contourf(
    KX,
    KY,
    epsilon / t,
    levels=100,
    cmap=custom_cmap,
    norm=color_norm
)

# epsilon = 0 的费米面：cos(k_x a) + cos(k_y a) = 0
fermi = ax.contour(
    KX,
    KY,
    epsilon,
    levels=[0],
    colors='black',
    linewidths=2.5
)

# 非零 Fermi 能级：与 epsilon_F = 0 一起以 t 为步长等间距排列
nonzero_fermi_levels = np.array([-3, -2, -1, 1, 2, 3]) * t
ax.contour(
    KX,
    KY,
    epsilon,
    levels=nonzero_fermi_levels,
    colors='gray',
    linestyles='--',
    linewidths=1.5
)

ax.set_xlabel(r'$k_x a$', fontsize=24)
ax.set_ylabel(r'$k_y a$', fontsize=24)

ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax.set_xticklabels([r'$\mbox{-}\pi$', r'$\mbox{-}\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
ax.set_yticklabels([r'$\mbox{-}\pi$', r'$\mbox{-}\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])

ax.set_aspect('equal')
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-np.pi, np.pi)

cbar = fig.colorbar(im, ax=ax, shrink=0.82)
cbar.set_label(r'$\varepsilon/t$', fontsize=22)
cbar_ticks = cbar.get_ticks()
cbar.set_ticks(cbar_ticks, labels=[tex_number(value, '.2f') for value in cbar_ticks])

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'square_lattice_fermi_surface.pdf'), dpi=600, bbox_inches='tight')
plt.show()
