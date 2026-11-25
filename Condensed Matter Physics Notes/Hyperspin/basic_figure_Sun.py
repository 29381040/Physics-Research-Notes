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

# PDF 的保存目录。
script_dir = "/Users/huangyiding/Desktop/Python Figure/Altermagnet/Hyperspin"
print(f"图片将保存到: {script_dir}")

from matplotlib.lines import Line2D


# 使用 newtxtext 的 TS1 负号字形，避免 PDF 导出时负号缺失。
plt.rcParams["text.latex.preamble"] += (
    r"\DeclareSymbolFont{plotminus}{TS1}{ntxtlf}{m}{n}"
    r"\DeclareMathSymbol{-}{\mathbin}{plotminus}{61}"
)


#%% 模型与公共参数
hopping_ratio = 0.40  # t_J/t，控制超自旋分裂和费米口袋的形状。
electrons_per_cell = 2.0  # 每个磁性晶胞的电子数。半填充时费米能为零。

# (k_x a, k_y a) 的扫描节点以 π 为单位。修改节点时同步更新符号和坐标标签。
path_nodes = np.pi * np.array([(0, 0), (1, 0), (1, 1), (0, 1)])
path_labels = [r"$\Gamma$", r"$X$", r"$M$", r"$Y$"]
path_coordinates = [r"$(0,0)$", r"$(\pi,0)$", r"$(\pi,\pi)$", r"$(0,\pi)$"]

path_tick_labels = [
    f"{label} {coordinates}" for label, coordinates in zip(path_labels, path_coordinates)
]

# 闭合扫描路径沿上方区边经过 Y 后返回 Γ。节点坐标以 π 为单位。
closed_path_nodes = np.pi * np.array([(0, 0), (1, 1), (0, 1), (-1, 1), (0, 0)])
closed_path_tick_labels = [
    r"$\Gamma\,(0,0)$", r"$M\,(\pi,\pi)$", r"$Y\,(0,\pi)$",
    r"$M$'$(-\pi,\pi)$", r"$\Gamma\,(0,0)$",
]

# 截图中的扫描路径，节点以 π 为单位。顺序同时决定两张图的横轴。
reference_path_nodes = np.pi * np.array([(1, 0), (1, 1), (0, 0), (-1, 1), (0, 1)])
reference_path_tick_labels = [
    r"$X\,(\pi,0)$", r"$M\,(\pi,\pi)$", r"$\Gamma\,(0,0)$",
    r"$M$'$(-\pi,\pi)$", r"$Y\,(0,\pi)$",
]

# 按 (η, λ) 指定能带、自旋曲线及费米面等值线的颜色。
band_colors = {
    (1, 1): custom_colors[3],    # 深红
    (1, -1): custom_colors[2],   # 浅红
    (-1, 1): custom_colors[1],   # 深蓝
    (-1, -1): custom_colors[0],  # 浅蓝
}
band_styles = {1: "-", -1: "-"}  # 按 λ 指定线型，两图共用，当前均取实线。
band_linewidth = 2.2  # 能带曲线和图例样线的线宽，单位 pt。
fermi_linewidth = 1.9  # 费米面等值线的线宽，单位 pt。
axis_label_size = 22  # 动量和能量轴标签的字号，单位 pt。
# 以 k_x a=π 处幺正变换后的 A/B 分量为分支命名，t_J 换号时 A/B 的能量次序对调。
edge_sublattices = {1: "B", -1: "A"} if hopping_ratio >= 0 else {1: "A", -1: "B"}
hyperspin_labels = {1: r"+\hbar/2", -1: r"-\hbar/2"}
branch_handles = [
    Line2D(
        [], [], color=color, lw=band_linewidth, ls=band_styles[band_sign],
        label=(
            rf"$\widetilde{{{edge_sublattices[band_sign]}}},\ "
            r"\langle\hat{\mathcal{J}}_z\rangle="
            rf"{hyperspin_labels[eta]}$"
        ),
    )
    for (eta, band_sign), color in band_colors.items()
]


def splitting_over_t(kx_a, ky_a, eta):
    """式 (100) 根号内能量的平方根，以 t 为单位。"""
    half_kx = kx_a / 2
    return np.hypot(
        np.cos(half_kx),
        hopping_ratio * (eta * np.cos(ky_a) - np.sin(half_kx)),
    )


def energy_over_t(kx_a, ky_a, eta, band_sign):
    """式 (100) 的 E/t，取 t>0，输入动量为 k_x a 与 k_y a。"""
    return 2 * (np.cos(ky_a) + band_sign * splitting_over_t(kx_a, ky_a, eta))


def spin_z_over_hbar(kx_a, ky_a, eta, band_sign):
    """式 (111) 的面外普通自旋期望值，单位为 hbar。"""
    splitting = splitting_over_t(kx_a, ky_a, eta)
    # 上下支简并时自旋期望取决于所选本征态。用 NaN 将这些点留空。
    return np.divide(
        0.5 * eta * band_sign * np.cos(kx_a / 2),
        splitting,
        out=np.full_like(splitting, np.nan),
        where=splitting > 1e-12,
    )


#%% 零温费米能
fermi_filling_samples = 1000  # 粒子数计算时第一布里渊区每个方向的网格数。


def calculate_fermi_energy(filling, samples):
    """给定每个磁性晶胞的电子数，通过零温粒子数约束求费米能 E_F/t。"""
    # k_y a 平移 π 后能谱整体变号，半填充的精确解是零能。
    if filling == len(band_colors) / 2:
        return 0.0

    # 在单个布里渊区采用等权的网格中点，避免重复采样周期边界。
    momenta = (np.arange(samples) + 0.5) * (2 * np.pi / samples) - np.pi
    kx, ky = np.meshgrid(momenta, momenta, sparse=True)
    state_energies = np.concatenate([
        energy_over_t(kx, ky, eta, band_sign).ravel()
        for eta, band_sign in band_colors
    ])
    # T=0 时占据所有低于费米能的态，分位点由目标占据比例确定。
    return np.quantile(state_energies, filling / len(band_colors), overwrite_input=True)


fermi_energy = calculate_fermi_energy(electrons_per_cell, fermi_filling_samples)
print(f"每个磁性晶胞 n={electrons_per_cell:g} 个电子，零温费米能 E_F/t={fermi_energy:.6f}")


#%% 公共绘图参数与函数
figure_height = 6.0  # 所有画布的高度，单位英寸。
figure_width_to_height = 1.5  # 所有画布及导出的 PDF 页面的宽高比。
figure_size = (figure_width_to_height * figure_height, figure_height)
path_samples = 500  # 每段扫描路径的采样间隔数，增大可使曲线更平滑。
fermi_samples = 1001  # 单个动量周期内每轴的网格点数，增大可提高等值线精度。


def save_figure(fig, pdf_filename):
    """按统一的宽高比导出 PDF。"""
    # 把完整画布纳入 tight 边界并取消额外白边，使 PDF 保持设定的宽高比。
    fig.savefig(
        os.path.join(script_dir, pdf_filename), dpi=600, bbox_inches='tight',
        pad_inches=0, bbox_extra_artists=[fig.patch],
    )


def spin_bound_positions(nodes, path_ticks):
    """求扫描路径上普通自旋达到 ±hbar/2 的累积路程。"""
    tolerance = 1e-12
    if hopping_ratio == 0:
        # 无交换跃迁时，各段的非简并本征态均已完全极化，取中点示意。
        kx_midpoints = (nodes[:-1, 0] + nodes[1:, 0]) / 2
        midpoints = (path_ticks[:-1] + path_ticks[1:]) / 2
        return midpoints[np.abs(np.cos(kx_midpoints / 2)) > tolerance]

    positions = []
    # 合并两个超自旋扇区后，饱和条件为 k_x a/2 ± k_y a = (n+1/2)π。
    for ky_sign in (-1, 1):
        phases = nodes[:, 0] / 2 + ky_sign * nodes[:, 1]
        for index, (start_phase, end_phase) in enumerate(zip(phases[:-1], phases[1:])):
            phase_step = end_phase - start_phase
            if abs(phase_step) <= tolerance:
                if abs(np.cos(start_phase)) > tolerance:
                    continue
                # 整段均饱和时，用一个中点表示。
                fractions = np.array([0.5])
            else:
                low, high = sorted((start_phase, end_phase))
                first_n = int(np.ceil((low - tolerance) / np.pi - 0.5))
                last_n = int(np.floor((high + tolerance) / np.pi - 0.5))
                targets = (np.arange(first_n, last_n + 1) + 0.5) * np.pi
                fractions = np.clip((targets - start_phase) / phase_step, 0.0, 1.0)

            kx_start, kx_end = nodes[index:index + 2, 0]
            kx = kx_start + fractions * (kx_end - kx_start)
            # cos(k_x a/2)=0 时式 (111) 的饱和值条件落在简并点。
            fractions = fractions[np.abs(np.cos(kx / 2)) > tolerance]
            start, stop = path_ticks[index:index + 2]
            positions.extend(start + fractions * (stop - start))

    return np.unique(np.round(positions, 12))


def plot_path(nodes, tick_labels, pdf_filename, *, spin=False):
    """沿给定节点采样。spin 为 True 时绘制普通自旋，否则绘制能带。"""
    segment_lengths = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
    path_ticks = np.r_[0.0, np.cumsum(segment_lengths)]
    path_distance = np.concatenate([
        np.linspace(start, stop, path_samples, endpoint=False)
        for start, stop in zip(path_ticks[:-1], path_ticks[1:])
    ] + [path_ticks[-1:]])
    kx_path = np.interp(path_distance, path_ticks, nodes[:, 0])
    ky_path = np.interp(path_distance, path_ticks, nodes[:, 1])

    if spin:
        observable = spin_z_over_hbar
        ylabel = r"$\langle\hat{S}_z\rangle/\hbar$"
        title = rf"Spin polarization at $t_J/t={hopping_ratio:g}$"
        reference_level = 0.0
    else:
        observable = energy_over_t
        ylabel = r"$E/t$"
        title = (
            rf"Band structure at $t_J/t={hopping_ratio:g},\quad "
            rf"\varepsilon_\mathrm{{F}}/t={fermi_energy:.4f}$"
        )
        reference_level = fermi_energy

    fig, ax = plt.subplots(figsize=figure_size)
    # top 为标题和单行图例留空，bottom 为路径标签和横轴标题留空。
    fig.subplots_adjust(left=0.13, right=0.94, bottom=0.16, top=0.80)
    fig.suptitle(title, y=0.96, fontsize=22)

    # 灰色竖虚线标记路径节点，水平点线标记费米能或零自旋。
    for node_distance in path_ticks[1:-1]:
        ax.axvline(node_distance, color="0.75", lw=0.85, ls="--", zorder=0.5)
    ax.axhline(reference_level, color="0.5", lw=1.6, ls=":", zorder=1)

    for (eta, band_sign), color in band_colors.items():
        ax.plot(
            path_distance,
            observable(kx_path, ky_path, eta, band_sign),
            color=color,
            lw=band_linewidth,
            ls=band_styles[band_sign],
        )

    if spin:
        # 两条灰色横虚线标出归一化自旋的上下界，绘在彩色曲线下方。
        bound_positions = spin_bound_positions(nodes, path_ticks)

        # 两个相邻段从同一侧靠近简并节点时，±1/2 的两侧极限一致。
        kx_turns, ky_turns = nodes[1:-1].T
        same_side = (nodes[:-2, 0] - kx_turns) * (nodes[2:, 0] - kx_turns) > 0
        smaller_splitting = np.minimum(
            splitting_over_t(kx_turns, ky_turns, 1),
            splitting_over_t(kx_turns, ky_turns, -1),
        )
        limit_positions = path_ticks[1:-1][same_side & (smaller_splitting <= 1e-12)]
        for spin_bound in (-0.5, 0.5):
            ax.axhline(spin_bound, color="0.70", lw=1.0, ls="--", zorder=0.75)
            if bound_positions.size:
                ax.scatter(
                    bound_positions, np.full(bound_positions.shape, spin_bound),
                    s=29, color="0.42", edgecolors="none", zorder=4,
                )  # s 控制灰点面积，单位 pt²。
            if limit_positions.size:
                # 空心灰点表示简并处沿扫描路径的共同极限。
                ax.scatter(
                    limit_positions, np.full(limit_positions.shape, spin_bound),
                    s=29, facecolors="white", edgecolors="0.42", linewidths=1.0, zorder=4,
                )
        # 两条路径的纵轴统一覆盖普通自旋的完整取值范围。
        ax.set_yticks([-0.5, -0.25, 0.0, 0.25, 0.5])
        ax.set_ylim(-0.55, 0.55)
    else:
        energy_limit = 1.10 * np.max(np.abs(ax.dataLim.intervaly))  # 能带上下各留 10% 空间。
        ax.set_ylim(-energy_limit, energy_limit)

    ax.set_xlim(path_ticks[0], path_ticks[-1])
    # 高对称点标签用 Matplotlib 自带的 STIX 字形，确保 Γ 在 PDF 中完整显示。
    ax.set_xticks(path_ticks, tick_labels, usetex=False, math_fontfamily="stix")
    ax.set_xlabel(r"$(k_x a,k_y a)$", fontsize=axis_label_size)
    ax.set_ylabel(ylabel, fontsize=axis_label_size)
    # bbox_to_anchor 的第二项控制图例距坐标框上沿的距离。
    ax.legend(
        handles=branch_handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncols=4, frameon=False, fontsize=13, handlelength=1.6, columnspacing=0.8,
    )

    save_figure(fig, pdf_filename)
    return fig, ax


def plot_fermi_surface(kx_limits, ky_limits, kx_tick_labels, ky_tick_labels, samples, pdf_filename):
    """绘制 E(k_x,k_y)=E_F 的等值线，费米能取自零温粒子数约束。"""
    kx_momentum = np.linspace(*kx_limits, num=samples)
    ky_momentum = np.linspace(*ky_limits, num=samples)
    kx_grid, ky_grid = np.meshgrid(kx_momentum, ky_momentum, sparse=True)

    fig, ax = plt.subplots(figsize=figure_size)
    # 右侧预留图例空间，左侧为等比例的动量坐标框。
    fig.subplots_adjust(left=0.12, right=0.67, bottom=0.15, top=0.90)
    for (eta, band_sign), color in band_colors.items():
        ax.contour(
            kx_momentum,
            ky_momentum,
            energy_over_t(kx_grid, ky_grid, eta, band_sign),
            levels=[fermi_energy],
            colors=[color],
            linewidths=fermi_linewidth,
            linestyles=[band_styles[band_sign]],
        )

    # 坐标刻度在各自的动量区间内等距分布。
    ax.set_xticks(np.linspace(*kx_limits, num=len(kx_tick_labels)), kx_tick_labels)
    ax.set_yticks(np.linspace(*ky_limits, num=len(ky_tick_labels)), ky_tick_labels)
    ax.set_xlim(*kx_limits)
    ax.set_ylim(*ky_limits)
    ax.set_box_aspect(1)  # 内部动量坐标框保持正方形。
    ax.set_axisbelow(True)
    ax.grid(color="0.90", lw=0.75)
    ax.set_xlabel(r"$k_x a$", fontsize=axis_label_size)
    ax.set_ylabel(r"$k_y a$", fontsize=axis_label_size)
    ax.set_title(
        rf"$\varepsilon_\mathrm{{F}}/t={fermi_energy:.4f},\quad t_J/t={hopping_ratio:g}$",
        fontsize=22, pad=11,
    )
    # bbox_to_anchor 的第一项控制图例距坐标框右侧的距离。
    ax.legend(
        handles=branch_handles, loc="center left", bbox_to_anchor=(1.08, 0.5),
        ncols=1, frameon=False, fontsize=15, handlelength=1.9,
        labelspacing=0.8, borderaxespad=0,
    )

    save_figure(fig, pdf_filename)
    return fig, ax




#%% 能带图 Γ → M → Y → (-π, π) → Γ
plot_path(closed_path_nodes, closed_path_tick_labels, "hyperspin_bands_gamma_m_minus_pi_pi_gamma.pdf")


#%% 能带图 X → M → Γ → (-π, π) → Y
plot_path(
    reference_path_nodes, reference_path_tick_labels,
    "hyperspin_bands_x_m_gamma_minus_pi_pi_y.pdf",
)




#%% 普通自旋 S_z Γ → M → Y → (-π, π) → Γ
plot_path(closed_path_nodes, closed_path_tick_labels, "hyperspin_spin_z_gamma_m_minus_pi_pi_gamma.pdf", spin=True)


#%% 普通自旋 S_z X → M → Γ → (-π, π) → Y
plot_path(
    reference_path_nodes, reference_path_tick_labels,
    "hyperspin_spin_z_x_m_gamma_minus_pi_pi_y.pdf", spin=True,
)


#%% 费米面图 一个动量周期
# 这两个区间使 X 与 M 周围的口袋位于窗口内部。修改区间时同步调整刻度标签。
kx_limits = (0.0, 2 * np.pi)
ky_limits = (-np.pi / 2, 3 * np.pi / 2)
kx_tick_labels = [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
ky_tick_labels = [r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$"]
fig_fermi, ax_fermi = plot_fermi_surface(
    kx_limits, ky_limits, kx_tick_labels, ky_tick_labels, fermi_samples,
    "hyperspin_fermi_surface.pdf",
)


#%% 费米面图 以 Γ 为中心的扩展区
extended_limits = (-2 * np.pi, 2 * np.pi)  # x、y 两轴均显示两个完整的动量周期。
extended_tick_labels = [r"$-2\pi$", r"$-\pi$", r"$0$", r"$\pi$", r"$2\pi$"]
extended_samples = 2 * fermi_samples - 1  # 区间加倍后维持原有的动量采样间距。
plot_fermi_surface(
    extended_limits, extended_limits, extended_tick_labels, extended_tick_labels,
    extended_samples, "hyperspin_fermi_surface_extended.pdf",
)
