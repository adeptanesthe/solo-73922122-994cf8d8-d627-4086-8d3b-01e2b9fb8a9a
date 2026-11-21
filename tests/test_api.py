"""API 层测试：正常流程、页面服务与整批拒绝行为。"""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "measured_times": [0, 1, 2, 3, 5, 8, 13],
    "measured_values": [25.3, 196.1, 330.8, 433.5, 578.2, 695.1, 769.9],
    "candidate_times": [0.5, 1.5, 2.5, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5,
                        8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 14, 14.5, 15, 15.5],
    "length_scale": 2.0,
    "noise_std": 5.0,
    "signal_variance": 100000.0,
    "min_interval": 1.0,
}


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_index_page_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "高温炉温度采样选点" in r.text


def test_select_ok():
    r = client.post("/api/select", json=VALID_PAYLOAD)
    assert r.status_code == 200, r.text
    data = r.json()
    # 选中两个不同的候选时刻
    assert len(data["selected_times"]) == 2
    assert data["selected_times"][0] != data["selected_times"][1]
    assert len(data["selected_indices"]) == 2
    # 总量自洽
    assert data["total_variance_before"] > data["total_variance_after"]
    assert abs(data["total_variance_before"] - data["total_variance_after"]
               - data["total_reduction"]) < 1e-6 * data["total_variance_before"]
    # 候选明细完整且选后方差不超过选前
    cands = data["candidates"]
    assert len(cands) == len(VALID_PAYLOAD["candidate_times"])
    for c in cands:
        assert c["variance_after"] <= c["variance_before"] + 1e-6
    # 已测时刻 5.0 对应的候选点被排除
    c5 = next(c for c in cands if c["time"] == 5.0)
    assert not c5["eligible"] and c5["exclusion_reason"]
    # 选中的点必须合法
    for idx in data["selected_indices"]:
        assert cands[idx]["eligible"] and cands[idx]["selected"]
    assert 0 < data["n_pairs_feasible"] <= data["n_pairs_total"]


def test_nan_in_candidates_rejected():
    payload = dict(VALID_PAYLOAD)
    raw = json.dumps(payload).replace("0.5", "NaN", 1)  # 构造含 NaN 的非法 JSON 字面量
    r = client.post("/api/select", content=raw,
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert r.json()["detail"]["code"] in ("INVALID_INPUT", "NON_FINITE_INPUT")


def test_non_psd_kernel_rejected():
    payload = dict(VALID_PAYLOAD, measured_times=[1.0, 1.0],
                   measured_values=[100.0, 100.0], noise_std=0.0)
    r = client.post("/api/select", json=payload)
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "KERNEL_NOT_PSD"


def test_no_feasible_combination_rejected():
    payload = dict(VALID_PAYLOAD, min_interval=1e6)
    r = client.post("/api/select", json=payload)
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "NO_FEASIBLE_COMBINATION"


def test_mismatched_measured_lengths_rejected():
    payload = dict(VALID_PAYLOAD, measured_values=[1.0, 2.0])
    r = client.post("/api/select", json=payload)
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "INVALID_INPUT"


def test_invalid_parameters_rejected():
    for field, bad in [("length_scale", 0.0), ("noise_std", -1.0),
                       ("signal_variance", -2.0), ("min_interval", -0.5)]:
        r = client.post("/api/select", json=dict(VALID_PAYLOAD, **{field: bad}))
        assert r.status_code == 422, field
        assert r.json()["detail"]["code"] == "INVALID_INPUT"


def test_error_response_has_no_partial_result():
    payload = dict(VALID_PAYLOAD, min_interval=1e6)
    r = client.post("/api/select", json=payload)
    body = r.json()
    assert "candidates" not in body and "selected_times" not in body


# ---------------------------------------------------------------------------
# 超温报警演练 API
# ---------------------------------------------------------------------------

DRILL_BASE = {
    "measured_times": VALID_PAYLOAD["measured_times"],
    "measured_values": VALID_PAYLOAD["measured_values"],
    "length_scale": 2.0,
    "noise_std": 5.0,
    "signal_variance": 100000.0,
    "alarm_temperature": 700.0,
}
DRILL_TIMES = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 9, 10.5, 11.5, 12.5, 14, 15.5]


def _start_drill(seed=20261121):
    r = client.post("/api/drill/start", json=dict(DRILL_BASE, drill_times=DRILL_TIMES, seed=seed))
    assert r.status_code == 200, r.text
    return r.json()


def test_drill_start_ok():
    data = _start_drill()
    assert len(data["trajectories"]) == 3
    for tr in data["trajectories"]:
        assert tr["times"] == DRILL_TIMES
        assert len(tr["values"]) == len(DRILL_TIMES)
        assert tr["alarm"]["exceeded"] in (True, False)
    assert data["n_distinct_times"] == len(DRILL_TIMES)
    assert data["max_distinct_times"] == 200
    assert data["step"] == 0 and data["seed"] == 20261121


def test_drill_start_reproducible():
    a = _start_drill(seed=99)
    b = _start_drill(seed=99)
    for ta, tb in zip(a["trajectories"], b["trajectories"]):
        assert ta["values"] == tb["values"]


def test_drill_extend_keeps_prefix_and_updates_marker():
    start = _start_drill()
    payload = dict(DRILL_BASE,
                   trajectories=[{"times": t["times"], "values": t["values"]}
                                 for t in start["trajectories"]],
                   new_times=[6.2, 12.8], seed=start["seed"], step=start["step"])
    r = client.post("/api/drill/extend", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["step"] == 1
    assert data["n_distinct_times"] == len(DRILL_TIMES) + 2
    for old, new in zip(start["trajectories"], data["trajectories"]):
        assert new["values"][:len(old["values"])] == old["values"]
        assert len(new["values"]) == len(old["values"]) + 2


def test_drill_extend_duplicate_returns_422_without_payload():
    start = _start_drill()
    payload = dict(DRILL_BASE,
                   trajectories=[{"times": t["times"], "values": t["values"]}
                                 for t in start["trajectories"]],
                   new_times=[1.5], seed=start["seed"], step=start["step"])
    r = client.post("/api/drill/extend", json=payload)
    assert r.status_code == 422
    body = r.json()
    assert body["detail"]["code"] == "DUPLICATE_DRILL_TIME"
    assert "trajectories" not in body


def test_drill_extend_limit_422():
    start = _start_drill()
    # 已有 14 个不同时刻，再插 187 个正好 201 个 -> 拒绝
    new_times = [100.0 + k for k in range(187)]
    payload = dict(DRILL_BASE,
                   trajectories=[{"times": t["times"], "values": t["values"]}
                                 for t in start["trajectories"]],
                   new_times=new_times, seed=start["seed"], step=start["step"])
    r = client.post("/api/drill/extend", json=payload)
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "TOO_MANY_DRILL_TIMES"


def test_drill_alarm_only_updates_markers():
    start = _start_drill()
    trajs = [{"times": t["times"], "values": t["values"]} for t in start["trajectories"]]
    r = client.post("/api/drill/alarm", json={"trajectories": trajs, "alarm_temperature": 1e9})
    assert r.status_code == 200, r.text
    data = r.json()
    for old, new in zip(start["trajectories"], data["trajectories"]):
        assert old["values"] == new["values"]
        assert new["alarm"]["exceeded"] is False
        assert new["alarm"]["first_time"] is None


def test_drill_invalid_input_rejected():
    r = client.post("/api/drill/start", json=dict(DRILL_BASE, drill_times=[], seed=1))
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "INVALID_INPUT"
    r = client.post("/api/drill/start", json=dict(DRILL_BASE, drill_times=[1.0], seed=-5))
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "INVALID_INPUT"
    r = client.post("/api/drill/alarm", json={"trajectories": [], "alarm_temperature": 1.0})
    assert r.status_code == 422


def test_drill_page_contains_section():
    r = client.get("/")
    assert r.status_code == 200
    assert "超温报警演练" in r.text
