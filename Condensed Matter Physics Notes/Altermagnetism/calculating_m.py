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
custom_cmap = LinearSegmentedColormap.from_list(
    'custom_cmap',
    custom_colors,
    N=256
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

script_dir = "/Users/huangyiding/Desktop/Python Figure/Altermagnet/Intro/Hubbard"
print(f"图片将保存到: {script_dir}")


#%% 模型参数与磁性布里渊区网格

#能量单位取 t=1
t = 1.0
t0_over_t = 0.6
t0 = t0_over_t * t
delta = 0.9
tJ = delta * t0

#单点自洽计算使用的相互作用强度
U_single_over_t = 3.5

#文章附录采用200×200的磁性布里渊区网格
N_k = 200

#文章附录规定的自洽收敛精度
convergence_tolerance = 1.0e-6

#文章附录给出的mixing迭代方法：m_next = 0.4*m_in + 0.6*m_out
old_solution_weight = 0.40

#非零初值用于避免m=0作为严格不动点
m_initial = 0.35

#最大迭代次数
max_iterations = 5000


#使用磁性布里渊区的倒格矢坐标：
#k = u*pi*(1,1) + v*pi*(1,-1)，u,v∈[-1/2,1/2)

#在 kx-ky 平面中对应菱形区域，|kx|+|ky| <= pi
u = (np.arange(N_k) + 0.5) / N_k - 0.5
v = (np.arange(N_k) + 0.5) / N_k - 0.5

U_grid, V_grid = np.meshgrid(u, v, indexing='ij')

kx = np.pi * (U_grid + V_grid)
ky = np.pi * (U_grid - V_grid)

#N对应附录公式中的磁性晶胞总数
N_unit_cells = N_k * N_k


#与动量有关、但不随 m 和 U 改变的量
epsilon_0 = -2.0 * t * (
    np.cos(kx) + np.cos(ky)
)

band_center = -2.0 * t0 * (
    np.cos(kx) * np.cos(ky)
)

Delta_k = (
    np.sin(kx) * np.sin(ky)
)


#%% 能带及附录S2中Q算符的期望值

def calculate_bands_and_Q(m_value, U_over_t):
    """
    计算四条能带以及每个本征态的 Q 期望值。

    基底为
        (A-up, B-up, A-down, B-down)

    附录 S2 中
        Q = diag(1,-1,-1,1)。

    这里使用2×2自旋分块的解析本征值与本征矢期望值，
    与逐点调用 np.linalg.eigh 得到的结果完全相同，但扫描相图更快。
    """
    U_value = U_over_t * t

    energy_list = []
    Q_expectation_list = []

    #zeta=+1：spin-up
    #zeta=-1：spin-down
    for zeta in (+1.0, -1.0):

        magnetic_term = (
            2.0 * tJ * Delta_k
            + zeta * U_value * m_value
        )

        splitting = np.sqrt(
            epsilon_0**2
            + magnetic_term**2
        )

        energy_minus = band_center - splitting
        energy_plus = band_center + splitting

        #Q在两条本征能带上的期望值。
        #下带对应 +zeta*magnetic_term/splitting，
        #上带取相反数
        Q_minus = np.divide(
            zeta * magnetic_term,
            splitting,
            out=np.zeros_like(splitting),
            where=splitting > 1.0e-14
        )
        Q_plus = -Q_minus

        energy_list.extend([
            energy_minus,
            energy_plus
        ])

        Q_expectation_list.extend([
            Q_minus,
            Q_plus
        ])

    energies = np.stack(energy_list, axis=0)
    Q_expectations = np.stack(
        Q_expectation_list,
        axis=0
    )

    return energies, Q_expectations


#%% T=0、半填充条件下的化学势与占据数

def half_filling_occupations(energies):
    """
    在 T=0、n=1 时占据全部 4N 状态中的最低 2N 个状态。

    若费米能恰好穿过有限网格中的简并能级，
    则对该简并能级采用等权分数占据，以保持晶格对称性。
    """
    flattened_energies = energies.ravel()

    number_occupied = 2 * N_unit_cells

    partitioned = np.partition(
        flattened_energies,
        [number_occupied - 1, number_occupied]
    )

    highest_occupied = partitioned[
        number_occupied - 1
    ]
    lowest_empty = partitioned[
        number_occupied
    ]

    chemical_potential = 0.5 * (
        highest_occupied + lowest_empty
    )

    occupations = (
        energies < chemical_potential
    ).astype(float)

    #处理费米能上的精确简并
    if np.isclose(
        highest_occupied,
        lowest_empty,
        rtol=0.0,
        atol=1.0e-13 * t
    ):
        fermi_level_states = np.isclose(
            energies,
            chemical_potential,
            rtol=0.0,
            atol=1.0e-13 * t
        )

        remaining_particles = (
            number_occupied
            - np.sum(occupations)
        )

        occupations[fermi_level_states] = (
            remaining_particles
            / np.sum(fermi_level_states)
        )

    return chemical_potential, occupations


#%% 单个 U/t 下的 m 自洽迭代

def solve_self_consistent_m(
    U_over_t,
    initial_m=m_initial
):
    """
    按照附录 S2-S3 迭代计算 m 和化学势。

    每一轮：
    1. 用 m_in 构造四条能带；
    2. 调整 mu 使总填充为 n=1；
    3. 由 S2 计算 m_out；
    4. 用 40% 旧解和 60% 新解进行混合。
    """
    m_in = initial_m

    for iteration in range(1, max_iterations + 1):

        energies, Q_expectations = (
            calculate_bands_and_Q(
                m_in,
                U_over_t
            )
        )

        chemical_potential, occupations = (
            half_filling_occupations(energies)
        )

        #附录公式 S2：
        #m_out = (1/4N) sum_{k,alpha}<psi|Q|psi> f(epsilon-mu)
        m_out = np.sum(
            Q_expectations * occupations
        ) / (4.0 * N_unit_cells)

        difference = np.abs(m_out - m_in)

        if difference < convergence_tolerance:
            m_self_consistent = m_out

            #用最终 m 重新计算一次能带和化学势
            final_energies, _ = (
                calculate_bands_and_Q(
                    m_self_consistent,
                    U_over_t
                )
            )

            final_mu, _ = (
                half_filling_occupations(
                    final_energies
                )
            )

            #每个k点把四条能带从低到高排序。
            #间接带隙定义为min_k E_3(k) - max_k E_2(k)
            sorted_energies = np.sort(
                final_energies,
                axis=0
            )

            indirect_gap = (
                np.min(sorted_energies[2])
                - np.max(sorted_energies[1])
            )

            return (
                m_self_consistent,
                final_mu,
                indirect_gap,
                iteration
            )

        #附录所说的“混入40%的旧结果”
        m_in = (
            old_solution_weight * m_in
            + (1.0 - old_solution_weight) * m_out
        )


#%% 指定参数下单独计算m

m_single, mu_single, gap_single, iteration_single = (
    solve_self_consistent_m(
        U_single_over_t,
        initial_m=m_initial
    )
)

print(
    rf"delta = {delta:.2f}, "
    rf"t0/t = {t0_over_t:.2f}, "
    rf"U/t = {U_single_over_t:.2f}"
)

print(
    rf"自洽磁矩 m = {m_single:.8f}"
)

print(
    rf"化学势 mu/t = {mu_single / t:.8f}"
)

print(
    rf"间接带隙/t = {gap_single / t:.8f}"
)

print(
    rf"迭代次数 = {iteration_single}"
)


#%% 扫描 U/t，计算相图

#相互作用扫描范围，取0到4
U_min_over_t = 0.0
U_max_over_t = 5.0

#U/t采样点数
N_U = 201

U_values = np.linspace(
    U_min_over_t,
    U_max_over_t,
    N_U
)

m_values = np.zeros_like(U_values)
mu_values = np.zeros_like(U_values)
indirect_gaps = np.zeros_like(U_values)

for index, U_over_t in enumerate(U_values):

    (
        m_values[index],
        mu_values[index],
        indirect_gaps[index],
        _
    ) = solve_self_consistent_m(
        U_over_t,
        initial_m=m_initial
    )

    mu_values[index] /= t
    indirect_gaps[index] /= t

    print(
        rf"U/t = {U_over_t:.3f}, "
        rf"m = {m_values[index]:.6f}, "
        rf"gap/t = {indirect_gaps[index]:.6f}"
    )


#%% 根据m和间接带隙确定相边界

#相变阈值判断
magnetic_phase_threshold = 5.0e-3

insulating_gap_threshold = 5.0e-3


def interpolate_crossing(x_values, y_values, threshold):
    """
    在线性插值下求 y 首次越过给定阈值的位置。
    """
    crossing_index = np.where(
        y_values >= threshold
    )[0][0]

    x_left = x_values[crossing_index - 1]
    x_right = x_values[crossing_index]

    y_left = y_values[crossing_index - 1]
    y_right = y_values[crossing_index]

    return x_left + (
        threshold - y_left
    ) * (x_right - x_left) / (
        y_right - y_left
    )


#NM -> AMM：m首次超过数值磁性阈值
U_NM_AMM = interpolate_crossing(
    U_values,
    m_values,
    magnetic_phase_threshold
)

#只在磁性区域内寻找AMM -> AMI边界
magnetic_indices = np.where(
    U_values >= U_NM_AMM
)[0]

first_insulating_index = magnetic_indices[
    np.where(
        indirect_gaps[magnetic_indices]
        >= insulating_gap_threshold
    )[0][0]
]

U_left = U_values[
    first_insulating_index - 1
]
U_right = U_values[
    first_insulating_index
]

gap_left = indirect_gaps[
    first_insulating_index - 1
]
gap_right = indirect_gaps[
    first_insulating_index
]

U_AMM_AMI = U_left + (
    insulating_gap_threshold - gap_left
) * (U_right - U_left) / (
    gap_right - gap_left
)

print(
    rf"NM -> AMM: U/t = {U_NM_AMM:.4f}"
)

print(
    rf"AMM -> AMI: U/t = {U_AMM_AMI:.4f}"
)


#%% 绘制相变图

fig, ax = plt.subplots(
    figsize=(11.5,7)
)

#NM区域不填充颜色，保持空白背景

#AMM：使用配色设置中的浅蓝色
ax.axvspan(
    U_NM_AMM,
    U_AMM_AMI,
    color=custom_colors[1],
    alpha=0.30,
    linewidth=0,
    zorder=0
)

#AMI：使用配色设置中的浅红色
ax.axvspan(
    U_AMM_AMI,
    U_max_over_t,
    color=custom_colors[3],
    alpha=0.28,
    linewidth=0,
    zorder=0
)

#自洽磁矩曲线统一使用黑色
ax.plot(
    U_values,
    m_values,
    color='black',
    linewidth=3.0,
    solid_capstyle='round',
    zorder=3
)

#相边界辅助线
ax.axvline(
    U_NM_AMM,
    color='0.35',
    linewidth=1.5,
    linestyle=(0, (4, 3)),
    zorder=2
)

ax.axvline(
    U_AMM_AMI,
    color='0.35',
    linewidth=1.5,
    linestyle=(0, (4, 3)),
    zorder=2
)

#相名称的位置可通过 phase_label_height调节
phase_label_height = 0.27

ax.text(
    0.5 * (U_min_over_t + U_NM_AMM),
    phase_label_height,
    r'\textbf{NM}',
    fontsize=27,
    ha='center',
    va='center'
)

ax.text(
    0.5 * (U_NM_AMM + U_AMM_AMI),
    phase_label_height,
    r'\textbf{AMM}',
    fontsize=27,
    ha='center',
    va='center'
)

ax.text(
    0.5 * (U_AMM_AMI + U_max_over_t),
    phase_label_height,
    r'\textbf{AMI}',
    fontsize=27,
    ha='center',
    va='center'
)

#左上角参数标注
ax.text(
    0.04,
    0.94,
    (
        rf'$\delta=t_J/t_0={delta:.1f},\quad '
        rf't_0/t={t0_over_t:.1f},\quad T=0$'
    ),
    transform=ax.transAxes,
    fontsize=20,
    ha='left',
    va='top'
)

ax.set_xlim(
    U_min_over_t,
    U_max_over_t
)

#修改上限可以控制曲线顶部留白
ax.set_ylim(0.0, 0.4)

ax.set_xlabel(
    r'$U/t$',
    fontsize=25
)

ax.set_ylabel(
    r'order parameter $m$',
    fontsize=25
)

ax.tick_params(
    direction='in',
    length=7,
    width=1.4,
    top=True,
    right=True
)

for spine in ax.spines.values():
    spine.set_linewidth(1.5)

plt.tight_layout()

plt.savefig(
    os.path.join(
        script_dir,
        'phase_transition_diagram_delta=0.9.pdf'
    ),
    dpi=600,
    bbox_inches='tight'
)

plt.show()

#%% 自定义 U/t 查询
#
U_query_over_t = 3.50

m_query, mu_query, gap_query, iteration_query = (
    solve_self_consistent_m(
        U_query_over_t,
        initial_m=m_initial
    )
)

print(
    rf"查询 U/t = {U_query_over_t:.4f}, "
    rf"m = {m_query:.8f}, "
    rf"mu/t = {mu_query / t:.8f}, "
    rf"gap/t = {gap_query / t:.8f}, "
    rf"迭代次数 = {iteration_query}"
)

# %%
