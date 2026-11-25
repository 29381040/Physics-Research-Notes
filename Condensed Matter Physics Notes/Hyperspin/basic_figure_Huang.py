#%% 导入与全局设置
# 在 macOS 终端安装绘图环境所需的 Python 包。
# python3 -m pip install matplotlib numpy pandas scipy
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

# 脚本和 VS Code 单元格共用这个 PDF 输出目录。
script_dir = "/Users/huangyiding/Desktop/Python Figure/Altermagnet/Hyperspin"
print(f"图片将保存到: {script_dir}")

from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

# 为 PDF 导出指定可显示的负号和大写 Γ 字形。
plt.rcParams["text.latex.preamble"] += (
    r"\DeclareSymbolFont{plotminus}{TS1}{ntxtlf}{m}{n}"
    r"\DeclareMathSymbol{-}{\mathbin}{plotminus}{61}"
    r"\DeclareSymbolFont{plotgreek}{U}{psy}{m}{n}"
    r"\DeclareMathSymbol{\plotGamma}{\mathord}{plotgreek}{71}"
)


#%% 共面螺旋模型与绘图参数
exchange_over_t = 1  # Δ/t = |Um|/t，取正值。Γ_p 处两支相隔 2Δ。
fermi_energy_over_t = 0.0  # 半填充费米能，同时决定能带参考线和费米轮廓。
cut_energy_over_t = -3.830  # 小窗切能，取下支最低能和 Γ_p 处下支能量之间可得到两个口袋。

scan_samples = 1001  # 对角线扫描点数。取奇数以采样 Γ_p。
contour_samples = 801  # 等能轮廓的每轴采样点数，增加点数可细化轮廓。
inset_parallel_limit = 1.82  # 小窗沿螺旋方向的 p_parallel 显示半宽。
inset_perpendicular_limit = 0.73  # 小窗垂直螺旋方向的 p_perp 显示半高。

figure_height = 6.0  # 三张画布的高度，单位为英寸。
figure_width_to_height = 1.5  # 画布和 PDF 页面的宽高比。
figure_size = (figure_height * figure_width_to_height, figure_height)
curve_linewidth = 2.8  # 能带和自旋曲线的线宽，单位为 pt。
contour_linewidth = 2.6  # 费米轮廓和等能轮廓的线宽，单位为 pt。
axis_label_size = 22  # 主图坐标轴标签的字号，单位为 pt。

# 色标从深蓝到浅灰再到深红，对应 S_z/hbar 从 -1/2 到 +1/2。
spin_cmap = LinearSegmentedColormap.from_list(
    "spin_z",
    [custom_colors[1], custom_colors[0], "0.88", custom_colors[2], custom_colors[3]],
    N=257,
)
spin_norm = Normalize(vmin=-0.5, vmax=0.5)
spin_label = r"$\langle\hat{S}_z\rangle/\hbar$"
gap_label = rf"$Um/t={exchange_over_t:g}$"

# 取晶格常数 a=1。能带与自旋图沿 p_x=p_y 从 -π 扫描到 π。
scan_momentum = np.linspace(-np.pi, np.pi, scan_samples)
momentum_ticks = np.linspace(-np.pi, np.pi, 5)
momentum_tick_labels = [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]
scan_tick_labels = [r"$-M$", r"$-\pi/2$", r"$\plotGamma$", r"$\pi/2$", r"$M$"]
branches = [(1, r"$E_+$"), (-1, r"$E_-$")]

# 两张扫描图共用主坐标框的四个边距，数值为画布宽高的比例。
scan_margins = dict(left=0.150, right=0.815, bottom=0.15, top=0.90)


def energy_and_spin(px, py, branch):
    """按式 (120) 和 (128) 返回 E/t 与 <S_z>/hbar。"""
    average = -np.sqrt(2) * (np.cos(px) + np.cos(py))
    difference = -np.sqrt(2) * (np.sin(px) + np.sin(py))
    radius = np.hypot(difference, exchange_over_t)
    return average + branch * radius, branch * difference / (2 * radius)


#%% 公共绘图函数
def add_spin_line(ax, x, y, spin, linewidth):
    """每一小段的颜色由两端自旋期望值的平均值决定。"""
    points = np.column_stack((x, y))
    segments = np.stack((points[:-1], points[1:]), axis=1)
    line = LineCollection(
        segments, cmap=spin_cmap, norm=spin_norm, linewidths=linewidth,
        capstyle="round", zorder=3,
    )
    line.set_array((spin[:-1] + spin[1:]) / 2)
    ax.add_collection(line)


def add_spin_contour(ax, x, y, branch, level, *, spiral_axes=False):
    """提取指定能量的轮廓，并在轮廓顶点重新求自旋。"""
    def evaluate(p1, p2):
        if spiral_axes:
            # 小窗的横轴沿 (1,1)，纵轴沿 (1,-1)。转回笛卡尔坐标后代入能带公式。
            p1, p2 = (p1 + p2) / np.sqrt(2), (p1 - p2) / np.sqrt(2)
        return energy_and_spin(p1, p2, branch)

    grid_x, grid_y = np.meshgrid(x, y, sparse=True)
    energy, _ = evaluate(grid_x, grid_y)
    contour = ax.contour(x, y, energy, levels=[level])
    paths = contour.allsegs[0]
    contour.remove()
    for points in paths:
        px, py = points.T
        _, spin = evaluate(px, py)
        add_spin_line(ax, px, py, spin, contour_linewidth)


def add_spin_colorbar(fig, left=0.865):
    # 色条位置以画布宽高的比例给出，left 控制色条距左边缘的距离。
    colorbar_ax = fig.add_axes([left, 0.31, 0.021, 0.40])
    scale = plt.cm.ScalarMappable(norm=spin_norm, cmap=spin_cmap)
    colorbar = fig.colorbar(scale, cax=colorbar_ax)
    colorbar.set_ticks([-0.5, 0.0, 0.5], labels=[r"$-1/2$", r"$0$", r"$1/2$"])
    colorbar.ax.tick_params(labelsize=15)
    colorbar.set_label(spin_label, fontsize=18, labelpad=8)


def format_scan_axis(ax, gamma_guide_top=1.0):
    ax.set_xticks(momentum_ticks, scan_tick_labels)
    ax.set_xlim(scan_momentum[0], scan_momentum[-1])
    ax.set_xlabel(r"$p_x=p_y$", fontsize=axis_label_size)
    ax.axvline(0, ymax=gamma_guide_top, color="0.74", lw=1.0, ls="--", zorder=1)


def save_figure(fig, pdf_filename):
    # 把完整画布纳入 tight 边界，确保导出的 PDF 也保持 1.5:1。
    fig.savefig(
        os.path.join(script_dir, pdf_filename), dpi=600, bbox_inches='tight',
        pad_inches=0, bbox_extra_artists=[fig.patch],
    )


#%% 能带图与双口袋等能小窗
fig_band, ax_band = plt.subplots(figsize=figure_size)
fig_band.subplots_adjust(**scan_margins)
# Γ_p 的竖虚线止于小窗轴标签下方，避免穿过文字。
format_scan_axis(ax_band, gamma_guide_top=0.69)

for branch, label in branches:
    energy, spin = energy_and_spin(scan_momentum, scan_momentum, branch)
    add_spin_line(ax_band, scan_momentum, energy, spin, curve_linewidth)
    gamma_energy, _ = energy_and_spin(0, 0, branch)
    ax_band.annotate(
        label, (0, gamma_energy), xytext=(6, 23 if branch == 1 else 11),
        textcoords="offset points",
        fontsize=16, color="0.24",
    )

ax_band.axhline(fermi_energy_over_t, color="0.53", lw=1.25, ls=":", zorder=1)
ax_band.axhline(cut_energy_over_t, color="0.61", lw=1.0, ls="--", zorder=1)
ax_band.annotate(
    r"$E_{\mathrm{cut}}$", (0.035, cut_energy_over_t), xycoords=("axes fraction", "data"),
    xytext=(0, 4), textcoords="offset points", fontsize=14, color="0.36",
)
ax_band.margins(y=0.065)
ax_band.autoscale_view(scalex=False)
ax_band.set_ylabel(r"$E/t$", fontsize=axis_label_size)
ax_band.set_title(f"Band structure at {gap_label}", fontsize=23, pad=12)

# 小窗位置按主坐标框的 [左、下、宽、高] 比例给出。
ax_cut = ax_band.inset_axes([0.30, 0.75, 0.40, 0.20])
parallel = np.linspace(-inset_parallel_limit, inset_parallel_limit, contour_samples)
perpendicular = np.linspace(-inset_perpendicular_limit, inset_perpendicular_limit, contour_samples)
add_spin_contour(ax_cut, parallel, perpendicular, -1, cut_energy_over_t, spiral_axes=True)
ax_cut.axhline(0, color="0.88", lw=0.75, zorder=1)
ax_cut.set_xlim(parallel[0], parallel[-1])
ax_cut.set_ylim(perpendicular[0], perpendicular[-1])
ax_cut.set_aspect("equal", adjustable="box")
ax_cut.set_xticks([])
ax_cut.set_yticks([])
ax_cut.set_xlabel(r"$p_\parallel$", fontsize=13, labelpad=1)
ax_cut.set_ylabel(r"$p_\perp$", fontsize=13, labelpad=2)
ax_cut.set_title(rf"$E_{{\mathrm{{cut}}}}/t={cut_energy_over_t:.2f}$", fontsize=14, pad=3)
add_spin_colorbar(fig_band)
save_figure(fig_band, "hyperspin_huang_bands.pdf")


#%% 零能费米面
fig_fermi, ax_fermi = plt.subplots(figsize=figure_size)
# 主坐标框按正方形显示，在右侧留出色标的位置。
fig_fermi.subplots_adjust(left=0.12, right=0.77, bottom=0.15, top=0.89)
momentum = np.linspace(-np.pi, np.pi, contour_samples)
for branch in (1, -1):
    add_spin_contour(ax_fermi, momentum, momentum, branch, fermi_energy_over_t)

# 浅灰对角线给出能带和 S_z 的共同扫描方向。
ax_fermi.plot([-np.pi, np.pi], [-np.pi, np.pi], color="0.78", lw=1.0, ls="--", zorder=1)
ax_fermi.set_xticks(momentum_ticks, momentum_tick_labels)
ax_fermi.set_yticks(momentum_ticks, momentum_tick_labels)
ax_fermi.set_xlim(-np.pi, np.pi)
ax_fermi.set_ylim(-np.pi, np.pi)
ax_fermi.set_aspect("equal", adjustable="box")
ax_fermi.set_axisbelow(True)
ax_fermi.grid(color="0.92", lw=0.75)
ax_fermi.set_xlabel(r"$p_x$", fontsize=axis_label_size)
ax_fermi.set_ylabel(r"$p_y$", fontsize=axis_label_size)
ax_fermi.set_title(
    rf"Fermi surface at $E_{{\mathrm{{F}}}}/t={fermi_energy_over_t:g}$, {gap_label}",
    fontsize=22, pad=12,
)
add_spin_colorbar(fig_fermi, left=0.78)
save_figure(fig_fermi, "hyperspin_huang_fermi_surface.pdf")


#%% 普通自旋 S_z 扫描
fig_spin, ax_spin = plt.subplots(figsize=figure_size)
fig_spin.subplots_adjust(**scan_margins)
format_scan_axis(ax_spin)

for branch, label in branches:
    _, spin = energy_and_spin(scan_momentum, scan_momentum, branch)
    add_spin_line(ax_spin, scan_momentum, spin, spin, curve_linewidth)
    # 在左半段的极值附近标出能带分支，文字朝向图内。
    label_momentum = -np.pi / 2
    _, label_spin = energy_and_spin(label_momentum, label_momentum, branch)
    ax_spin.annotate(
        label, (label_momentum, label_spin), xytext=(0, -10 * branch),
        textcoords="offset points", ha="center", va="top" if branch == 1 else "bottom",
        fontsize=16, color="0.24",
    )

for bound in (-0.5, 0.5):
    ax_spin.axhline(bound, color="0.61", lw=1.0, ls="--", zorder=1)
ax_spin.axhline(0, color="0.53", lw=1.25, ls=":", zorder=1)
# 给 ±1/2 的辅助线留边距，使它们与边框分开。
ax_spin.set_ylim(-0.55, 0.55)
ax_spin.set_yticks([-0.5, -0.25, 0.0, 0.25, 0.5])
ax_spin.set_ylabel(spin_label, fontsize=axis_label_size)
ax_spin.set_title(f"Spin polarization at {gap_label}", fontsize=23, pad=12)
add_spin_colorbar(fig_spin)
save_figure(fig_spin, "hyperspin_huang_spin_z.pdf")
