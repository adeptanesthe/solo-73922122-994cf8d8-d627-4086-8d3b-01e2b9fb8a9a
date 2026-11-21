"""FastAPI 入口：高温炉温度采样选点服务。"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from .drill import (
    DrillError,
    extend_trajectories,
    recheck_alarm,
    sample_new_trajectories,
)
from .gp import GPError, select_pair
from .schemas import (
    DrillAlarmRequest,
    DrillExtendRequest,
    DrillStartRequest,
    SelectRequest,
)

app = FastAPI(
    title="高温炉温度采样选点",
    version="1.1.0",
    summary="基于高斯过程后验协方差下降选择两个新增采样点，并支持超温报警演练",
)

_STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.exception_handler(RequestValidationError)
async def _validation_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    """把 schema 校验失败统一成结构化 422（整批拒绝，不给部分建议）。"""
    errors = [
        {
            "loc": [str(p) for p in err.get("loc", [])],
            "msg": err.get("msg", ""),
            "type": err.get("type", ""),
        }
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "code": "INVALID_INPUT",
                "message": "输入校验失败（字段缺失、非有限值或参数越界），整批拒绝",
                "errors": errors,
            }
        },
    )


def _drill_failure(exc: Exception) -> HTTPException:
    """演练错误 -> 422 结构化响应（前端据此保留上次轨迹与标记）。"""
    if isinstance(exc, DrillError):
        return HTTPException(status_code=422, detail={"code": exc.code, "message": str(exc)})
    if isinstance(exc, GPError):
        return HTTPException(status_code=422, detail={"code": exc.code, "message": str(exc)})
    return HTTPException(
        status_code=422,
        detail={"code": "DRILL_ERROR", "message": f"演练计算失败：{exc}"},
    )


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/select")
def select(req: SelectRequest) -> dict:
    """计算最优新增采样点对，返回选点前后每个候选时刻的方差。

    失败时返回 422，detail.code 取值：
    NON_FINITE_INPUT / KERNEL_NOT_PSD / NO_FEASIBLE_COMBINATION / INVALID_INPUT。
    """
    try:
        res = select_pair(
            req.measured_times,
            req.measured_values,
            req.candidate_times,
            length_scale=req.length_scale,
            noise_std=req.noise_std,
            signal_variance=req.signal_variance,
            min_interval=req.min_interval,
        )
    except GPError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc

    candidates = [
        {
            "index": i,
            "time": float(t),
            "eligible": bool(res.eligible[i]),
            "exclusion_reason": res.exclusion_reasons[i],
            "variance_before": float(res.variance_before[i]),
            "variance_after": float(res.variance_after[i]),
            "posterior_mean": float(res.posterior_mean[i]),
            "selected": i in res.selected_indices,
        }
        for i, t in enumerate(req.candidate_times)
    ]
    return {
        "selected_indices": list(res.selected_indices),
        "selected_times": list(res.selected_times),
        "total_variance_before": res.total_variance_before,
        "total_variance_after": res.total_variance_after,
        "total_reduction": res.total_reduction,
        "n_pairs_total": res.n_pairs_total,
        "n_pairs_feasible": res.n_pairs_feasible,
        "candidates": candidates,
    }


# ---------------------------------------------------------------------------
# 超温报警演练
# ---------------------------------------------------------------------------


def _run_drill(fn, **kwargs) -> dict:
    try:
        return fn(**kwargs)
    except (DrillError, GPError) as exc:
        raise _drill_failure(exc) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail={"code": "INVALID_INPUT", "message": str(exc)}
        ) from exc


@app.post("/api/drill/start", tags=["drill"])
def drill_start(req: DrillStartRequest) -> dict:
    """演练开始：在给定时刻上从后验独立抽出 3 条轨迹，并标出各自的首次超温点。

    失败（422）时不返回任何轨迹，前端保留上次结果。错误码：
    INVALID_INPUT / NON_FINITE_INPUT / KERNEL_NOT_PSD / DUPLICATE_DRILL_TIME /
    TOO_MANY_DRILL_TIMES / DRILL_ERROR。
    """
    result = _run_drill(
        sample_new_trajectories,
        measured_times=req.measured_times,
        measured_values=req.measured_values,
        drill_times=req.drill_times,
        length_scale=req.length_scale,
        noise_std=req.noise_std,
        signal_variance=req.signal_variance,
        alarm_temperature=req.alarm_temperature,
        seed=req.seed,
    )
    result["alarm_temperature"] = req.alarm_temperature
    result["seed"] = req.seed
    result["step"] = 0
    return result


@app.post("/api/drill/extend", tags=["drill"])
def drill_extend(req: DrillExtendRequest) -> dict:
    """插入新时刻：每条轨迹以自身已有取值为条件联合抽样，原有取值不变。

    失败（422：重复时刻 / 超过 200 个不同时刻 / 数值问题）时前端保留上次轨迹与标记。
    """
    result = _run_drill(
        extend_trajectories,
        measured_times=req.measured_times,
        measured_values=req.measured_values,
        trajectories=[t.model_dump() for t in req.trajectories],
        new_times=req.new_times,
        length_scale=req.length_scale,
        noise_std=req.noise_std,
        signal_variance=req.signal_variance,
        alarm_temperature=req.alarm_temperature,
        seed=req.seed,
        step=req.step,
    )
    result["alarm_temperature"] = req.alarm_temperature
    result["seed"] = req.seed
    result["step"] = req.step + 1
    return result


@app.post("/api/drill/alarm", tags=["drill"])
def drill_alarm(req: DrillAlarmRequest) -> dict:
    """只调整报警温度：沿用原轨迹，仅重算每条轨迹的首次超温标记。"""
    result = _run_drill(
        recheck_alarm,
        trajectories=[t.model_dump() for t in req.trajectories],
        alarm_temperature=req.alarm_temperature,
    )
    result["alarm_temperature"] = req.alarm_temperature
    return result


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")
