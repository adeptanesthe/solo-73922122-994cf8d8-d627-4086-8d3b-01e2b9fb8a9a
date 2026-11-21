"""超温报警演练核心逻辑测试。"""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.drill import (
    MAX_DRILL_TIMES,
    DrillError,
    DuplicateDrillTimeError,
    TooManyDrillTimesError,
    count_distinct_times,
    derive_seed,
    extend_trajectories,
    first_alarm_time,
    recheck_alarm,
    rng_standard_normals,
    sample_joint_extension,
    sample_new_trajectories,
)
from app.gp import posterior_mean_cov

MEASURED_T = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0]
MEASURED_Y = [25.3, 196.1, 330.8, 433.5, 578.2, 695.1, 769.9]
PARAMS = dict(length_scale=2.0, noise_std=5.0, signal_variance=100000.0)
DRILL_TIMES = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 9.0, 10.5, 11.5, 12.5, 14.0, 15.5]


# ---------------------------------------------------------------------------
# 随机流
# ---------------------------------------------------------------------------


def test_rng_is_reproducible_and_nontrivial():
    a = rng_standard_normals(12345, 10)
    b = rng_standard_normals(12345, 10)
    c = rng_standard_normals(12346, 10)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.all(np.isfinite(a))


def test_rng_moments_reasonable():
    z = np.concatenate([rng_standard_normals(k, 2000) for k in range(5)])
    assert abs(float(np.mean(z))) < 0.05
    assert abs(float(np.var(z)) - 1.0) < 0.1


def test_derive_seed_deterministic_and_distinct():
    assert derive_seed(7, 1) == derive_seed(7, 1)
    seeds = {derive_seed(7, s * 64 + k + 1) for s in range(5) for k in range(3)}
    assert len(seeds) == 15


# ---------------------------------------------------------------------------
# 首次超温标记
# ---------------------------------------------------------------------------


def test_first_alarm_picks_earliest_exceedance():
    res = first_alarm_time([5.0, 1.0, 3.0, 9.0], [100.0, 50.0, 700.0, 900.0], 600.0)
    assert res["exceeded"]
    assert res["first_time"] == pytest.approx(3.0)
    assert res["first_value"] == pytest.approx(700.0)


def test_first_alarm_none_when_all_below():
    res = first_alarm_time([1.0, 2.0], [10.0, 20.0], 100.0)
    assert res["exceeded"] is False
    assert res["first_time"] is None and res["first_value"] is None and res["index"] is None


def test_first_alarm_strictly_greater_than_threshold():
    # 恰好等于报警温度不算超温
    res = first_alarm_time([1.0, 2.0], [100.0, 100.0], 100.0)
    assert not res["exceeded"]
    res = first_alarm_time([1.0, 2.0], [100.0, 100.0000001], 100.0)
    assert res["exceeded"] and res["first_time"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 轨迹抽样
# ---------------------------------------------------------------------------


def test_start_returns_three_independent_reproducible_trajectories():
    a = sample_new_trajectories(measured_times=MEASURED_T, measured_values=MEASURED_Y,
                                drill_times=DRILL_TIMES, alarm_temperature=700.0,
                                seed=42, **PARAMS)
    b = sample_new_trajectories(measured_times=MEASURED_T, measured_values=MEASURED_Y,
                                drill_times=DRILL_TIMES, alarm_temperature=700.0,
                                seed=42, **PARAMS)
    assert len(a["trajectories"]) == 3
    for ta, tb in zip(a["trajectories"], b["trajectories"]):
        assert ta["times"] == DRILL_TIMES
        assert np.allclose(ta["values"], tb["values"])       # 同种子可复现
    # 三条轨迹不共享随机流
    v0, v1, v2 = (np.array(t["values"]) for t in a["trajectories"])
    assert not np.allclose(v0, v1)
    assert not np.allclose(v1, v2)
    # 每条轨迹都带有标记字段
    for tr in a["trajectories"]:
        assert set(tr["alarm"]) >= {"first_time", "first_value", "index", "exceeded"}


def test_start_trajectories_condition_on_measured_values():
    """离实测数据较远的轨迹应围绕后验均值波动，且零噪声时在已测点精确命中。"""
    times = [0.0, 1.0, 2.0, 5.0]
    values = [100.0, 200.0, 300.0, 500.0]
    res = sample_joint_extension(
        measured_times=times, measured_values=values,
        simulated_times=[], simulated_values=[], new_times=times,
        length_scale=1.0, noise_std=0.0, signal_variance=1e4, seed=3,
    )
    assert np.array_equal(np.array(res["values"]), np.array(values))


def test_zero_noise_exact_at_subset_of_measured_times():
    res = sample_joint_extension(
        measured_times=MEASURED_T, measured_values=MEASURED_Y,
        simulated_times=[], simulated_values=[], new_times=[0.0, 2.0, 13.0, 4.0],
        length_scale=2.0, noise_std=0.0, signal_variance=100000.0, seed=9,
    )
    assert res["values"][0] == MEASURED_Y[0]
    assert res["values"][1] == MEASURED_Y[2]
    assert res["values"][2] == MEASURED_Y[6]
    assert res["values"][3] != MEASURED_Y[2]  # 非实测点不是简单照搬


def test_noise_only_affects_measured_values():
    """零噪声时，轨迹在已测点等于实测温度；加噪声后允许偏离。"""
    kwargs = dict(
        measured_times=MEASURED_T, measured_values=MEASURED_Y,
        simulated_times=[], simulated_values=[], new_times=[1.0, 8.0],
        length_scale=2.0, signal_variance=100000.0, seed=11,
    )
    exact = sample_joint_extension(noise_std=0.0, **kwargs)
    noisy = sample_joint_extension(noise_std=50.0, **kwargs)
    assert exact["values"] == pytest.approx([MEASURED_Y[1], MEASURED_Y[5]], abs=0)
    assert noisy["values"] != pytest.approx([MEASURED_Y[1], MEASURED_Y[5]], abs=1e-9)


def test_joint_sampling_matches_conditional_gp_distribution():
    """用不同种子重复抽样，验证样本均值/协方差与条件 GP 理论值一致。"""
    measured_t = [0.0, 2.0, 6.0]
    measured_y = [100.0, 400.0, 800.0]
    sim_t = [1.0, 4.0]
    sim_y = [250.0, 600.0]
    new_t = [3.0, 5.0, 7.5]
    ls, nstd, sv = 1.5, 10.0, 5e4

    t_all = sim_t + new_t
    mean_all, cov_all = posterior_mean_cov(
        measured_t, measured_y, t_all,
        length_scale=ls, noise_std=nstd, signal_variance=sv,
    )
    n_s = len(sim_t)
    cov_ss = cov_all[:n_s, :n_s]
    cov_sn = cov_all[:n_s, n_s:]
    ref_mean = mean_all[n_s:] + cov_sn.T @ np.linalg.solve(
        cov_ss, np.array(sim_y) - mean_all[:n_s])
    ref_cov = cov_all[n_s:, n_s:] - cov_sn.T @ np.linalg.solve(cov_ss, cov_sn)

    draws = np.array([
        sample_joint_extension(
            measured_times=measured_t, measured_values=measured_y,
            simulated_times=sim_t, simulated_values=sim_y, new_times=new_t,
            length_scale=ls, noise_std=nstd, signal_variance=sv, seed=s,
        )["values"]
        for s in range(3000)
    ])
    assert draws.shape == (3000, 3)
    np.testing.assert_allclose(draws.mean(axis=0), ref_mean, rtol=0, atol=0.06 * math.sqrt(sv))
    np.testing.assert_allclose(np.cov(draws, rowvar=False), ref_cov,
                               rtol=0, atol=0.15 * sv)
    # 新点之间存在由核与条件集诱导的相关性（联合抽样而非独立边缘）
    corr = np.corrcoef(draws, rowvar=False)
    assert np.any(np.abs(corr - np.eye(3)) > 0.05)


def test_extension_keeps_existing_values_exactly():
    res = sample_new_trajectories(measured_times=MEASURED_T, measured_values=MEASURED_Y,
                                  drill_times=DRILL_TIMES, alarm_temperature=700.0,
                                  seed=5, **PARAMS)
    ext = extend_trajectories(
        measured_times=MEASURED_T, measured_values=MEASURED_Y,
        trajectories=res["trajectories"], new_times=[6.2, 12.8],
        alarm_temperature=700.0, seed=5, step=0, **PARAMS,
    )
    for old, new in zip(res["trajectories"], ext["trajectories"]):
        assert new["times"][:len(old["times"])] == old["times"]
        assert new["values"][:len(old["values"])] == old["values"]
        assert new["times"][len(old["times"]):] == [6.2, 12.8]
        assert len(new["times"]) == len(old["times"]) + 2
    assert ext["n_distinct_times"] == len(DRILL_TIMES) + 2


def test_extension_only_samples_new_points():
    """追加抽样必须只产生新点；已有时间点的取值不受新随机流影响。"""
    base = sample_new_trajectories(measured_times=MEASURED_T, measured_values=MEASURED_Y,
                                   drill_times=[0.5, 1.5], alarm_temperature=1e9,
                                   seed=8, **PARAMS)
    ext = extend_trajectories(measured_times=MEASURED_T, measured_values=MEASURED_Y,
                              trajectories=base["trajectories"], new_times=[10.0],
                              alarm_temperature=1e9, seed=8, step=0, **PARAMS)
    for old, new in zip(base["trajectories"], ext["trajectories"]):
        assert np.array_equal(np.array(old["values"]), np.array(new["values"][:2]))


def test_recheck_alarm_keeps_values_and_updates_markers():
    res = sample_new_trajectories(measured_times=MEASURED_T, measured_values=MEASURED_Y,
                                  drill_times=DRILL_TIMES, alarm_temperature=1e9,
                                  seed=6, **PARAMS)
    low = recheck_alarm(trajectories=res["trajectories"], alarm_temperature=100.0)
    high = recheck_alarm(trajectories=res["trajectories"], alarm_temperature=1e9)
    for a, b, orig in zip(low["trajectories"], high["trajectories"], res["trajectories"]):
        assert a["values"] == b["values"] == orig["values"]
        assert a["times"] == b["times"] == orig["times"]
    assert any(t["alarm"]["exceeded"] for t in low["trajectories"])
    assert not any(t["alarm"]["exceeded"] for t in high["trajectories"])


# ---------------------------------------------------------------------------
# 失败情形（调用方保留上次轨迹）
# ---------------------------------------------------------------------------


def test_duplicate_new_time_rejected():
    res = sample_new_trajectories(measured_times=MEASURED_T, measured_values=MEASURED_Y,
                                  drill_times=[0.5, 1.5], alarm_temperature=700.0,
                                  seed=1, **PARAMS)
    with pytest.raises(DuplicateDrillTimeError):
        extend_trajectories(measured_times=MEASURED_T, measured_values=MEASURED_Y,
                            trajectories=res["trajectories"], new_times=[1.5],
                            alarm_temperature=700.0, seed=1, step=0, **PARAMS)
    with pytest.raises(DuplicateDrillTimeError):
        extend_trajectories(measured_times=MEASURED_T, measured_values=MEASURED_Y,
                            trajectories=res["trajectories"], new_times=[3.0, 3.0],
                            alarm_temperature=700.0, seed=1, step=0, **PARAMS)


def test_start_rejects_duplicate_drill_times():
    with pytest.raises(DuplicateDrillTimeError):
        sample_new_trajectories(measured_times=MEASURED_T, measured_values=MEASURED_Y,
                                drill_times=[1.0, 1.0], alarm_temperature=700.0,
                                seed=1, **PARAMS)


def test_limit_200_distinct_times():
    res = sample_new_trajectories(measured_times=[], measured_values=[],
                                  drill_times=[0.0], alarm_temperature=700.0,
                                  seed=1, **PARAMS)
    fill = [10.0 + k for k in range(MAX_DRILL_TIMES - 1)]
    ok = extend_trajectories(measured_times=[], measured_values=[],
                             trajectories=res["trajectories"], new_times=fill,
                             alarm_temperature=700.0, seed=1, step=0, **PARAMS)
    assert ok["n_distinct_times"] == MAX_DRILL_TIMES
    with pytest.raises(TooManyDrillTimesError):
        extend_trajectories(measured_times=[], measured_values=[],
                            trajectories=ok["trajectories"], new_times=[999.0],
                            alarm_temperature=700.0, seed=1, step=1, **PARAMS)


def test_limit_200_on_start():
    with pytest.raises(TooManyDrillTimesError):
        sample_new_trajectories(measured_times=[], measured_values=[],
                                drill_times=list(range(MAX_DRILL_TIMES + 1)),
                                alarm_temperature=700.0, seed=1, **PARAMS)


def test_count_distinct_times_uses_same_time_tolerance():
    assert count_distinct_times([[0.0, 1.0], [1.0 + 1e-13, 2.0]]) == 3


def test_drill_errors_have_codes():
    for exc, code in [(DuplicateDrillTimeError("x"), "DUPLICATE_DRILL_TIME"),
                      (TooManyDrillTimesError("x"), "TOO_MANY_DRILL_TIMES"),
                      (DrillError("x"), "DRILL_ERROR")]:
        assert exc.code == code
