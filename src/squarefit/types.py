"""几何层中稳定、不随算法变化的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Dimensions:
    """方孔、方轴及赛题判定阈值。

    所有长度均为微米（um）。以任务书给出的极限尺寸为例：
    孔边长可以取 10006 um、轴边长取 10000 um，此时单边名义间隙为 3 um。
    """

    hole_side_um: float
    shaft_side_um: float
    insertion_depth_um: float = 10000.0
    gap_lower_um: float = 2.0
    gap_upper_um: float = 4.0
    residual_translation_limit_um: float = 0.1
    residual_rotation_limit_arcsec: float = 0.2

    def __post_init__(self) -> None:
        if self.hole_side_um <= 0 or self.shaft_side_um <= 0:
            raise ValueError("孔和轴的边长必须为正数。")
        if self.insertion_depth_um <= 0:
            raise ValueError("插入深度必须为正数。")
        if self.gap_lower_um > self.gap_upper_um:
            raise ValueError("间隙下限不能大于上限。")
        if self.residual_translation_limit_um < 0:
            raise ValueError("平移残差阈值不能为负数。")
        if self.residual_rotation_limit_arcsec < 0:
            raise ValueError("角度残差阈值不能为负数。")

    @property
    def nominal_single_side_clearance_um(self) -> float:
        """同轴、无转角时任意一侧的名义间隙。"""
        return (self.hole_side_um - self.shaft_side_um) / 2.0


@dataclass(frozen=True)
class Pose6D:
    """方轴相对方孔坐标系的六维位姿。

    平移 dx/dy/dz 的单位为 um；旋转量是同一个旋转向量的三个分量，单位为 rad。
    约定 R = exp([theta_x, theta_y, theta_z]_x)，不是依次绕 x、y、z
    轴旋转的欧拉角。因此本项目不存在“先 Rx 还是先 Rz”的旋转顺序问题。
    """

    dx_um: float = 0.0
    dy_um: float = 0.0
    dz_um: float = 0.0
    theta_x_rad: float = 0.0
    theta_y_rad: float = 0.0
    theta_z_rad: float = 0.0


@dataclass(frozen=True)
class AssemblyResult:
    """一次横向装配几何计算的结果。"""

    g1_um: float
    g2_um: float
    g3_um: float
    g4_um: float
    minimum_clearance_um: float
    collision: bool
    labelled_gaps_qualified: bool

    @property
    def labelled_gaps_um(self) -> tuple[float, float, float, float]:
        """按赛题标签顺序返回 (g1, g2, g3, g4)。"""
        return self.g1_um, self.g2_um, self.g3_um, self.g4_um


@dataclass(frozen=True)
class WallClearance:
    """一个截面到一面孔壁的最小间隙及其限制角点。

    ``station`` 只会是 ``entry``（孔口）或 ``bottom``（插入底部）；
    ``wall`` 只会是 ``right``、``left``、``top``、``bottom``。
    ``limiting_corners`` 保存达到该最小间隙的方轴局部角点名称。居中时
    可能有两个角点并列，这是正常的，不代表已经发生真实的线接触。
    """

    station: str
    station_um: float
    wall: str
    clearance_um: float
    limiting_corners: tuple[str, ...]


@dataclass(frozen=True)
class ContactState:
    """方形轴孔的几何侧壁接触风险判定结果。

    这是由刚体几何间隙得到的“几何预测风险”，不是包含摩擦、倒角、弹性
    变形和接触力的完整物理接触模型。后续网络和控制器可使用它做安全逻辑、
    分层抽样和结果统计。
    """

    safety_level: str
    topology: str
    axial_location: str
    minimum: WallClearance
    active_constraints: tuple[WallClearance, ...]
    recommendation: str
    contact_tolerance_um: float
    warning_clearance_um: float

    @property
    def minimum_clearance_um(self) -> float:
        """所有八个侧壁间隙中的最小值。"""
        return self.minimum.clearance_um


@dataclass(frozen=True)
class AnalyticCompensationResult:
    """一次理想解析补偿的可核验结果。

    ``adjustment`` 就是报告中的解析首轮补偿 u_geo。它在方孔坐标系
    中左乘到当前刚体位姿；在真实系统中，还需要由后续执行器模型把该理想
    调整量映射为实际压电/运动模组命令。
    """

    measured_pose: Pose6D
    target_pose: Pose6D
    adjustment: Pose6D
    predicted_pose_after: Pose6D
