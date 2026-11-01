# fvmkit

径向瞬态扩散的守恒型有限体积内核，以及判断这些数值结果能不能用的校验工具。

求解器支持平板、圆柱、球三种几何，物性可随解变化，边界为表格数据驱动的
Robin 条件，可同时求解多个耦合场，也可在材料坐标下处理收缩域。纯 Python
实现，核心不依赖任何第三方库。

校验部分是这个包独立存在的主要理由。它提供三样东西：冻结物性的同类问题的
闭式解、实测收敛阶的估计方法，以及能区分"格式真的不守恒"与"记账口径不一致"
的守恒审计。

```
                         solve                     prove
  tabulated env  -->  fvmkit.kernel  -->  fvmkit.verify
  properties     -->  (theta, Picard) -->  (series, order, residual)
```

## 模块

| 模块 | 内容 |
|---|---|
| `fvmkit.geometry` | `[0, R]` 上 `slab` / `cylinder` / `sphere` 的控制体积划分 |
| `fvmkit.kernel` | theta 法 + Picard 迭代、块 Gauss-Seidel 耦合、材料坐标 |
| `fvmkit.envtab` | 保形插值、表格曲线的 C1 延拓、连续化阈值时刻 |
| `fvmkit.verify` | Bessel 与三角级数解、Richardson 外推、实测收敛阶、自收敛 |
| `fvmkit.audit` | 单因子扫描、反解标定、记账口径对比、薄层模型拟合 |
| `fvmkit.docsync` | 报告中的陈旧数字扫描、同名异源产物检测 |

## 安装

```bash
pip install -e .
```

可选依赖：`pip install -e ".[reference]"` 启用 scipy 的 Bessel 函数，
`.[plot]` 装 matplotlib，`.[dev]` 装 pytest。

## 最小示例

```python
from fvmkit import Field, Grid, solve_coupled

grid = Grid(R=0.02, N=80, geom="cylinder")

fields = [
    Field("c", init=2.0,
          conductivity=lambda s: [D_of(s["c"][i], s["T"][i]) for i in range(grid.N + 1)],
          robin=8e-7, env=lambda t: 0.09),
    Field("T", init=301.15,
          conductivity=lambda s: [k_of(s["c"][i]) for i in range(grid.N + 1)],
          robin=25.0, env=lambda t: 340.0,
          capacity=lambda s: [grid.v[i] * rho_cp(s["c"][i]) for i in range(grid.N + 1)]),
]

history, stats = solve_coupled(grid, fields, dt=8.0, nsteps=1800, theta=0.5)

print(stats["fields"]["c"]["residual"])   # 良态问题下约 1e-14
```

`Field.capacity` 是**体积积分后的容量**，不是密度：浓度场取控制体体积，
温度场取 `v * rho * cp`。若误传 `rho * cp`，格式仍然守恒，但所有量级都会变，
而且不容易从结果上看出来。

## 校验

`examples/02_verification.py` 用有闭式解的冻结物性问题跑三项检查。本机实测：

| 几何 | N=40 误差 | N=80 | N=160 | 实测阶 |
|---|---:|---:|---:|---:|
| slab | 1.97e-04 | 4.92e-05 | 1.23e-05 | 1.997 |
| cylinder | 3.88e-04 | 9.71e-05 | 2.43e-05 | 1.998 |
| sphere | 5.74e-04 | 1.44e-04 | 3.59e-05 | 1.998 |

固定 `N=80` 只加密时间步，相邻差分为 `1.64e-07, 4.09e-08, 1.02e-08`，
实测阶 2.000。守恒残差（相对边界通量积分）在 3e-14 到 2e-13 之间。

这些数字是当前代码的测量值，不是设计目标。改动内核后应重新跑一遍。

## 精度与开销

空间与时间均为二阶。原点处用半控制体，直接得到 L'Hôpital 极限系数
`2(p+1)/dr^2`，不需要 `r = 0` 分支，也就不存在写错这个分支的机会。

开销是纯 Python 的开销。80 个控制体的圆柱跑 1800 步约 2 秒；320 个控制体
跑 259200 步需要几分钟。如果需要更快，内核足够小，可以移植或用 numpy 包裹
内层循环，但改完要重新核对收敛数字。

## 边界

覆盖范围：一维径向的线性扩散算子、随解变化的非线性系数、Robin 与对称边界、
多个耦合场、给定半径历史。

不覆盖、需要调用方自行处理：源项、潜热等交叉耦合闭合、各向异性介质、
截面速度不均匀的移动边界。材料坐标变换依赖"刚性且均匀收缩"这一前提，
才能使对流项与容量项相消；该前提在 `solve_moving` 的文档里明确写出。

## 文档

- [`docs/methodology.md`](docs/methodology.md) —— 完整的校验流程，以及它
  要捕捉的失效特征。
- [`README.md`](README.md) —— 英文版。

## 许可

MIT。
