"""API 请求模型（Pydantic v2）。

allow_inf_nan=False 使 NaN / Inf 在 schema 层即被拒绝；
app.gp / app.drill 内部还有一层非有限值检查作为纵深防御。
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_OBSERVATIONS = 1000
MAX_CANDIDATES = 500
MAX_DRILL_TIMES = 200
MAX_SAFE_SEED = 2 ** 53 - 1  # 前端（JS Number）可精确往返的整数上限


class SelectRequest(BaseModel):
    """选点请求体。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    measured_times: list[float] = Field(
        default_factory=list, max_length=MAX_OBSERVATIONS,
        description="已测时刻",
    )
    measured_values: list[float] = Field(
        default_factory=list, max_length=MAX_OBSERVATIONS,
        description="已测温度，与已测时刻一一对应",
    )
    candidate_times: list[float] = Field(
        min_length=2, max_length=MAX_CANDIDATES,
        description="候选采样时刻",
    )
    length_scale: float = Field(gt=0, description="平方指数核长度尺度 ℓ")
    noise_std: float = Field(ge=0, description="观测噪声标准差 σ_n")
    signal_variance: float = Field(default=1.0, gt=0, description="核信号方差")
    min_interval: float = Field(default=0.0, ge=0, description="两个新采样点之间的最小时间间隔")

    @model_validator(mode="after")
    def _check_aligned(self) -> "SelectRequest":
        if len(self.measured_times) != len(self.measured_values):
            raise ValueError("已测时刻与已测温度数量不一致")
        return self


class DrillBaseRequest(BaseModel):
    """演练的公共字段：沿用表单里的实测读数、观测噪声与核参数。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    measured_times: list[float] = Field(
        default_factory=list, max_length=MAX_OBSERVATIONS,
        description="已测时刻",
    )
    measured_values: list[float] = Field(
        default_factory=list, max_length=MAX_OBSERVATIONS,
        description="已测温度，与已测时刻一一对应",
    )
    length_scale: float = Field(gt=0, description="平方指数核长度尺度 ℓ")
    noise_std: float = Field(ge=0, description="观测噪声标准差 σ_n")
    signal_variance: float = Field(default=1.0, gt=0, description="核信号方差")
    alarm_temperature: float = Field(description="报警温度")

    @model_validator(mode="after")
    def _check_aligned(self) -> "DrillBaseRequest":
        if len(self.measured_times) != len(self.measured_values):
            raise ValueError("已测时刻与已测温度数量不一致")
        return self


class DrillStartRequest(DrillBaseRequest):
    """开始演练：在一批时刻上从后验独立抽出 3 条轨迹。"""

    drill_times: list[float] = Field(
        min_length=1, max_length=MAX_DRILL_TIMES,
        description="演练时刻（每批最多 200 个，累计也不超过 200 个）",
    )
    seed: int = Field(
        default=1, ge=0, le=MAX_SAFE_SEED,
        description="随机种子（便于复现；同一种子 + 同一输入得到同一批轨迹）",
    )


class DrillTrajectory(BaseModel):
    """一条已模拟轨迹（由前端原样带回，用于追加抽样）。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    times: list[float] = Field(default_factory=list, max_length=MAX_DRILL_TIMES)
    values: list[float] = Field(default_factory=list, max_length=MAX_DRILL_TIMES)

    @model_validator(mode="after")
    def _check_aligned(self) -> "DrillTrajectory":
        if len(self.times) != len(self.values):
            raise ValueError("轨迹时刻与取值数量不一致")
        return self


class DrillExtendRequest(DrillBaseRequest):
    """插入新时刻：每条轨迹以自身已有取值为条件，联合抽样新时刻。"""

    trajectories: list[DrillTrajectory] = Field(
        min_length=3, max_length=3,
        description="当前 3 条轨迹（追加后原有点的取值保持不变）",
    )
    new_times: list[float] = Field(
        min_length=1, max_length=MAX_DRILL_TIMES,
        description="本次要插入的新时刻",
    )
    seed: int = Field(default=1, ge=0, le=MAX_SAFE_SEED)
    step: int = Field(default=0, ge=0, description="已完成的加点批次数")


class DrillAlarmRequest(BaseModel):
    """只改报警温度：沿用原轨迹，仅重算首次超温标记。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    trajectories: list[DrillTrajectory] = Field(min_length=3, max_length=3)
    alarm_temperature: float = Field(description="报警温度")
