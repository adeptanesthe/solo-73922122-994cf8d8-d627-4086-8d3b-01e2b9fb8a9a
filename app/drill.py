"""超温报警演练：条件高斯过程轨迹抽样与首次超温判定。

演练的有状态部分（当前轨迹、基点、已插入时刻）由前端保存，本模块只做无状态纯计算：

1. 每条轨迹以 **实测读数**（带观测噪声，进入后验）和 **该轨迹已有取值** 为条件；
2. 在新增时刻上，按条件高斯过程 **联合** 抽样全部新点，不是逐点边缘抽样后拼接；
3. 已模拟时刻的取值一次抽出后固定不变，之后只追加；
4. 观测噪声只作用于实测读数，不对轨迹额外加噪声；
5. 零噪声时，已测时刻上的模拟值精确等于实测温度。

三条轨迹使用同一后验、不同随机流，因而在给定数据下相互独立。
随机数使用显式种子的 xoshiro128**，抽样流程为
“标准正态偏差（Box-Muller）→ Cholesky 下三角 → 均值 + L·z”，结果可复现。
"""
from __future__ import annotations

import math

import numpy as np

from .gp import (
    GPError,
    NonFiniteInputError,
    _SAME_TIME_ATOL,
    _SAME_TIME_RTOL,
    _as_finite_1d,
    _finite_scalar,
    posterior_mean_cov,
)

__all__ = [
    "DrillError",
    "DuplicateDrillTimeError",
    "TooManyDrillTimesError",
    "MAX_DRILL_TIMES",
    "N_TRAJECTORIES",
    "first_alarm_time",
    "sample_joint_extension",
    "sample_new_trajectories",
    "extend_trajectories",
    "recheck_alarm",
    "count_distinct_times",
    "rng_standard_normals",
    "derive_seed",
]

MAX_DRILL_TIMES = 200
"""演练中累计的不同时刻数上限（演练时刻的并集，不含仅用于条件的实测读数）。"""

N_TRAJECTORIES = 3
"""每次演练固定的轨迹条数。"""

_ALARM_RTOL = 1e-12
"""超温判定容差：模拟值须严格大于报警温度（仅吸收双精度浮点噪声）。"""


class DrillError(GPError):
    """演练计算错误（API 返回 422；前端保留已有轨迹与标记）。"""

    code = "DRILL_ERROR"


class DuplicateDrillTimeError(DrillError):
    """演练时刻重复（新时刻已模拟过或批量内重复）。"""

    code = "DUPLICATE_DRILL_TIME"


class TooManyDrillTimesError(DrillError):
    """累计时刻数超过上限。"""

    code = "TOO_MANY_DRILL_TIMES"


# ---------------------------------------------------------------------------
# 确定性 PRNG（xoshiro128**，64 位输出），种子可复现
# ---------------------------------------------------------------------------

_MASK64 = (1 << 64) - 1


def _rotl64(x: int, k: int) -> int:
    return ((x << k) | (x >> (64 - k))) & _MASK64


class _SplitMix64:
    """把 64 位种子扩展成高质量随机序列（也用于派生种子）。"""

    def __init__(self, seed: int) -> None:
        self.state = seed & _MASK64

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
        return z ^ (z >> 31)


def derive_seed(seed: int, step: int = 1) -> int:
    """由基点种子派生第 step 段独立随机流的种子（确定性、可复现）。"""
    sm = _SplitMix64((int(seed) + 0x9E3779B97F4A7C15 * (int(step) & 0xFFFFFFFF)) & _MASK64)
    return sm.next()


class _Xoshiro128StarStar:
    """xoshiro128**，输出 64 位无符号整数。"""

    def __init__(self, seed: int) -> None:
        sm = _SplitMix64(seed)
        self.s = [sm.next() & 0xFFFFFFFF for _ in range(4)]

    def next64(self) -> int:
        s0, s1, s2, s3 = self.s
        result = (_rotl64((s1 * 5) & 0xFFFFFFFF, 7) * 9) & 0xFFFFFFFF
        t = (s1 << 9) & 0xFFFFFFFF
        s2 ^= s0
        s3 ^= s1
        s1 ^= s2
        s0 ^= s3
        s2 ^= t
        s3 = _rotl64(s3, 11)
        self.s = [s0, s1, s2, s3]
        return ((s1 << 32) | result) & _MASK64


def rng_standard_normals(seed: int, n: int) -> np.ndarray:
    """由 64 位种子生成 n 个标准正态偏差（Box-Muller，两两一对）。"""
    if not isinstance(seed, int) or seed < 0 or seed > _MASK64:
        raise DrillError("随机种子必须是 0 到 2^64-1 之间的整数")
    rng = _Xoshiro128StarStar(seed)
    out = np.empty(n, dtype=float)
    for k in range(0, n, 2):
        u1 = ((rng.next64() >> 11) + 0.5) / float(1 << 53)
        u2 = ((rng.next64() >> 11) + 0.5) / float(1 << 53)
        r = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2
        out[k] = r * math.cos(theta)
        if k + 1 < n:
            out[k + 1] = r * math.sin(theta)
    return out


# ---------------------------------------------------------------------------
# 时刻工具
# ---------------------------------------------------------------------------


def _match_index(t: float, grid: np.ndarray) -> int | None:
    idx = np.where(np.isclose(t, grid, rtol=_SAME_TIME_RTOL, atol=_SAME_TIME_ATOL))[0]
    return int(idx[0]) if idx.size else None


def _assert_unique_times(times: np.ndarray, label: str) -> None:
    if times.size > 1:
        iu0, iu1 = np.triu_indices(times.size, k=1)
        if np.any(np.isclose(times[iu0], times[iu1], rtol=_SAME_TIME_RTOL, atol=_SAME_TIME_ATOL)):
            raise DuplicateDrillTimeError(f"{label}中存在重复时刻，请去重后再试")


def _union_times(groups) -> np.ndarray:
    """按“同一时刻”容差合并多组时刻，返回升序去重数组。"""
    all_times = np.sort(np.concatenate([np.asarray(g, dtype=float) for g in groups if len(g)]))
    if all_times.size == 0:
        return all_times
    merged = [all_times[0]]
    for t in all_times[1:]:
        if not math.isclose(float(t), float(merged[-1]), rel_tol=_SAME_TIME_RTOL,
                            abs_tol=_SAME_TIME_ATOL):
            merged.append(t)
    return np.asarray(merged, dtype=float)


def count_distinct_times(trajectory_times) -> int:
    """演练中累计的不同时刻数（各轨迹时刻的并集，容差去重）。

    只统计演练时刻；实测读数作为条件数据，不计入 200 个时刻的上限。
    """
    return int(_union_times(list(trajectory_times)).size)


# ---------------------------------------------------------------------------
# 联合抽样
# ---------------------------------------------------------------------------


def _cholesky_with_floor(cov: np.ndarray, signal_variance: float):
    """条件协方差 Cholesky；数值上非正定时做特征值下限修正。

    返回 (L, adjusted)。adjusted 只在存在 **显著负** 特征值（数值错误）时为真；
    零噪声下已测点带来的精确约束会让协方差半正定（有一个 0 特征值），这是
    数学上正确的情形，只补下限、不提示。
    """
    cov = 0.5 * (cov + cov.T)
    try:
        chol = np.linalg.cholesky(cov)
        pivot_scale = math.sqrt(max(float(np.max(np.diag(cov))), np.finfo(float).tiny))
        if np.all(np.diag(chol) > 1e-10 * pivot_scale):
            return chol, False
    except np.linalg.LinAlgError:
        pass
    vals, vecs = np.linalg.eigh(cov)
    scale = max(signal_variance, 1.0)
    adjusted = bool(np.min(vals) < -1e-8 * scale)
    floor = 1e-12 * scale
    vals = np.clip(vals, floor, None)
    rebuilt = (vecs * vals) @ vecs.T
    return np.linalg.cholesky(0.5 * (rebuilt + rebuilt.T)), adjusted


def _validate_params(length_scale, noise_std, signal_variance):
    length_scale = _finite_scalar(length_scale, "length_scale")
    noise_std = _finite_scalar(noise_std, "noise_std")
    signal_variance = _finite_scalar(signal_variance, "signal_variance")
    if length_scale <= 0:
        raise DrillError("核长度尺度必须 > 0")
    if noise_std < 0:
        raise DrillError("观测噪声必须 >= 0")
    if signal_variance <= 0:
        raise DrillError("核信号方差必须 > 0")
    return length_scale, noise_std, signal_variance


def _measured_arrays(measured_times, measured_values):
    t_obs = _as_finite_1d(measured_times, "已测时刻")
    y_obs = _as_finite_1d(measured_values, "已测温度")
    if t_obs.size != y_obs.size:
        raise DrillError("已测时刻与已测温度数量不一致")
    return t_obs, y_obs


def sample_joint_extension(
    *,
    measured_times,
    measured_values,
    simulated_times,
    simulated_values,
    new_times,
    length_scale: float,
    noise_std: float,
    signal_variance: float = 1.0,
    seed: int,
) -> dict:
    """为一条轨迹在新时刻上联合抽样。

    条件集合 = 实测读数（带观测噪声）+ 该轨迹已有取值（按精确函数值处理）。
    新时刻上的条件分布为多元正态，一次性联合抽样。

    返回 {"times", "values", "n_conditioned", "numerically_adjusted"}。
    """
    length_scale, noise_std, signal_variance = _validate_params(
        length_scale, noise_std, signal_variance
    )
    t_obs, y_obs = _measured_arrays(measured_times, measured_values)
    t_sim = _as_finite_1d(simulated_times, "已模拟时刻")
    y_sim = _as_finite_1d(simulated_values, "已模拟值")
    if t_sim.size != y_sim.size:
        raise DrillError("该轨迹已模拟时刻与取值数量不一致")
    t_new = _as_finite_1d(new_times, "新增时刻")

    _assert_unique_times(t_sim, "已模拟时刻")
    _assert_unique_times(t_new, "新增时刻")
    for t in t_new:
        if _match_index(float(t), t_sim) is not None:
            raise DuplicateDrillTimeError(
                f"时刻 {t:g} 已模拟过：该时刻的轨迹取值已固定，不能重复抽样"
            )

    n_sim = int(t_sim.size)
    n_new = int(t_new.size)
    if n_new == 0:
        return {
            "times": [],
            "values": [],
            "n_conditioned": int(t_obs.size + n_sim),
            "numerically_adjusted": False,
        }

    # 一次性后验覆盖“已有模拟点 + 新点”，两部分的均值/协方差来自同一条件分布。
    t_all = np.concatenate([t_sim, t_new])
    mean_all, cov_all = posterior_mean_cov(
        t_obs, y_obs, t_all,
        length_scale=length_scale, noise_std=noise_std, signal_variance=signal_variance,
    )
    mean_all = np.asarray(mean_all, dtype=float)
    cov_all = np.asarray(cov_all, dtype=float)
    if not (np.all(np.isfinite(mean_all)) and np.all(np.isfinite(cov_all))):
        raise NonFiniteInputError("后验均值或协方差含非有限值，无法抽样")

    mu_s, mu_n = mean_all[:n_sim], mean_all[n_sim:]
    cov_nn = cov_all[n_sim:, n_sim:]
    cov_ns = cov_all[n_sim:, :n_sim]

    adjusted = False
    if n_sim:
        cov_ss = cov_all[:n_sim, :n_sim]
        # 已模拟时刻理论上互不相同；极小 jitter 只用于数值求解稳定。
        jitter = 1e-12 * max(signal_variance, 1.0)
        try:
            chol_ss = np.linalg.cholesky(cov_ss + jitter * np.eye(n_sim))
        except np.linalg.LinAlgError as exc:
            raise DrillError("已模拟时刻的协方差矩阵数值上非正定，无法追加抽样") from exc
        alpha = np.linalg.solve(chol_ss, y_sim - mu_s)
        cond_mean = mu_n + cov_ns @ np.linalg.solve(chol_ss.T, alpha)
        w = np.linalg.solve(chol_ss, cov_ns.T)          # (n_sim, n_new)
        cond_cov = cov_nn - w.T @ w
    else:
        cond_mean = mu_n
        cond_cov = cov_nn

    cond_cov = 0.5 * (cond_cov + cond_cov.T)
    diag = np.diag(cond_cov)
    neg_tol = 1e-8 * max(signal_variance, noise_std * noise_std, 1.0)
    if np.any(diag < -neg_tol):
        raise DrillError("条件协方差对角线出现显著负值，无法抽样")

    chol, adjusted = _cholesky_with_floor(cond_cov, signal_variance)
    if not (np.all(np.isfinite(chol)) and np.all(np.isfinite(cond_mean))):
        raise DrillError("条件分布参数含非有限值，无法抽样")

    z = rng_standard_normals(int(seed), n_new)
    values = cond_mean + chol @ z

    # 零噪声时，已测时刻的模拟值精确固定为实测温度（消除浮点误差）。
    if noise_std == 0.0 and t_obs.size:
        for k, t in enumerate(t_new):
            hit = _match_index(float(t), t_obs)
            if hit is not None:
                values[k] = float(y_obs[hit])

    if not np.all(np.isfinite(values)):
        raise DrillError("抽样结果含非有限值")

    return {
        "times": [float(v) for v in t_new],
        "values": [float(v) for v in values],
        "n_conditioned": int(t_obs.size + n_sim),
        "numerically_adjusted": bool(adjusted),
    }


def _trajectory_seed(base_seed: int, step: int, traj_index: int) -> int:
    # 每条轨迹、每个批次使用互不重叠的派生种子，保证独立且可复现。
    return derive_seed(base_seed, step * 64 + traj_index + 1)


def _make_result(trajectories, alarm_temperature) -> dict:
    markers = []
    for tr in trajectories:
        markers.append(first_alarm_time(tr["times"], tr["values"], alarm_temperature))
    n_distinct = count_distinct_times([tr["times"] for tr in trajectories])
    return {
        "trajectories": [
            {
                "times": [float(t) for t in tr["times"]],
                "values": [float(v) for v in tr["values"]],
                "alarm": marker,
            }
            for tr, marker in zip(trajectories, markers)
        ],
        "n_distinct_times": n_distinct,
        "max_distinct_times": MAX_DRILL_TIMES,
    }


def sample_new_trajectories(
    *,
    measured_times,
    measured_values,
    drill_times,
    length_scale: float,
    noise_std: float,
    signal_variance: float = 1.0,
    alarm_temperature: float,
    seed: int,
) -> dict:
    """演练开始：在给定的一批时刻上，从后验独立抽出 3 条轨迹。"""
    length_scale, noise_std, signal_variance = _validate_params(
        length_scale, noise_std, signal_variance
    )
    alarm = _finite_scalar(alarm_temperature, "alarm_temperature")
    t_obs, y_obs = _measured_arrays(measured_times, measured_values)
    t_drill = _as_finite_1d(drill_times, "演练时刻")
    _assert_unique_times(t_drill, "演练时刻")
    if t_drill.size == 0:
        raise DrillError("请至少填写一个演练时刻")

    n_total = count_distinct_times([t_drill])
    if n_total > MAX_DRILL_TIMES:
        raise TooManyDrillTimesError(
            f"演练时刻 {n_total} 个，超过上限 {MAX_DRILL_TIMES} 个；请减少演练时刻"
        )

    trajectories = []
    adjusted = False
    for k in range(N_TRAJECTORIES):
        res = sample_joint_extension(
            measured_times=t_obs,
            measured_values=y_obs,
            simulated_times=[],
            simulated_values=[],
            new_times=t_drill,
            length_scale=length_scale,
            noise_std=noise_std,
            signal_variance=signal_variance,
            seed=_trajectory_seed(int(seed), 0, k),
        )
        trajectories.append({"times": res["times"], "values": res["values"]})
        adjusted = adjusted or res["numerically_adjusted"]
    result = _make_result(trajectories, alarm)
    result["numerically_adjusted"] = adjusted
    return result


def extend_trajectories(
    *,
    measured_times,
    measured_values,
    trajectories,
    new_times,
    length_scale: float,
    noise_std: float,
    signal_variance: float = 1.0,
    alarm_temperature: float,
    seed: int,
    step: int,
) -> dict:
    """在已有轨迹上插入新时刻：每条轨迹以自身已有取值为条件独立联合抽样。

    任一步失败都会抛异常，调用方据此保留上一次的轨迹和标记。
    """
    length_scale, noise_std, signal_variance = _validate_params(
        length_scale, noise_std, signal_variance
    )
    alarm = _finite_scalar(alarm_temperature, "alarm_temperature")
    t_obs, y_obs = _measured_arrays(measured_times, measured_values)
    t_new = _as_finite_1d(new_times, "新增时刻")
    _assert_unique_times(t_new, "新增时刻")
    if t_new.size == 0:
        raise DrillError("请填写要插入的新时刻")
    if not isinstance(trajectories, (list, tuple)) or len(trajectories) != N_TRAJECTORIES:
        raise DrillError(f"演练轨迹必须是 {N_TRAJECTORIES} 条")
    if not isinstance(step, int) or step < 0:
        raise DrillError("步数必须是非负整数")

    # 新时刻不得与任一条轨迹的已模拟时刻重合
    existing = []
    for k, tr in enumerate(trajectories):
        t_sim = _as_finite_1d(tr.get("times", []), f"轨迹 {k + 1} 已模拟时刻")
        _assert_unique_times(t_sim, f"轨迹 {k + 1} 已模拟时刻")
        existing.append(t_sim)
        for t in t_new:
            if _match_index(float(t), t_sim) is not None:
                raise DuplicateDrillTimeError(
                    f"时刻 {t:g} 已模拟过：该时刻的轨迹取值已固定，不能重复抽样"
                )

    n_total = count_distinct_times(existing + [t_new])
    if n_total > MAX_DRILL_TIMES:
        raise TooManyDrillTimesError(
            f"累计不同时刻将达 {n_total} 个，超过上限 {MAX_DRILL_TIMES} 个；"
            f"本次加点已取消，原有轨迹与标记保持不变"
        )

    extended = []
    adjusted = False
    for k, tr in enumerate(trajectories):
        res = sample_joint_extension(
            measured_times=t_obs,
            measured_values=y_obs,
            simulated_times=tr.get("times", []),
            simulated_values=tr.get("values", []),
            new_times=t_new,
            length_scale=length_scale,
            noise_std=noise_std,
            signal_variance=signal_variance,
            seed=_trajectory_seed(int(seed), step + 1, k),
        )
        extended.append({
            "times": list(tr.get("times", [])) + res["times"],
            "values": list(tr.get("values", [])) + res["values"],
        })
        adjusted = adjusted or res["numerically_adjusted"]
    result = _make_result(extended, alarm)
    result["numerically_adjusted"] = adjusted
    return result


def recheck_alarm(*, trajectories, alarm_temperature) -> dict:
    """仅更新报警标记（沿用原轨迹，不重新抽样）。"""
    alarm = _finite_scalar(alarm_temperature, "alarm_temperature")
    if not isinstance(trajectories, (list, tuple)) or len(trajectories) != N_TRAJECTORIES:
        raise DrillError(f"演练轨迹必须是 {N_TRAJECTORIES} 条")
    parsed = []
    for k, tr in enumerate(trajectories):
        times = _as_finite_1d(tr.get("times", []), f"轨迹 {k + 1} 时刻")
        values = _as_finite_1d(tr.get("values", []), f"轨迹 {k + 1} 取值")
        if times.size != values.size:
            raise DrillError(f"轨迹 {k + 1} 的时刻与取值数量不一致")
        parsed.append({"times": times, "values": values})
    result = _make_result(parsed, alarm)
    result["numerically_adjusted"] = False  # 沿用轨迹，未重新抽样
    return result


# ---------------------------------------------------------------------------
# 首次超温
# ---------------------------------------------------------------------------


def first_alarm_time(times, values, alarm_temperature: float) -> dict:
    """在已模拟时刻中找最早的超温点（模拟值 > 报警温度）。

    返回 {"first_time", "first_value", "index", "exceeded"}；未超温时前三项为 None。
    """
    alarm = _finite_scalar(alarm_temperature, "alarm_temperature")
    t = _as_finite_1d(times, "模拟时刻")
    y = _as_finite_1d(values, "模拟值")
    if t.size != y.size:
        raise DrillError("模拟时刻与模拟值数量不一致")
    best_idx: int | None = None
    best_t = math.inf
    for k in range(t.size):
        if float(y[k]) > alarm + _ALARM_RTOL * max(1.0, abs(alarm)):
            if float(t[k]) < best_t:
                best_t = float(t[k])
                best_idx = k
    if best_idx is None:
        return {"first_time": None, "first_value": None, "index": None, "exceeded": False}
    return {
        "first_time": float(t[best_idx]),
        "first_value": float(y[best_idx]),
        "index": int(best_idx),
        "exceeded": True,
    }
