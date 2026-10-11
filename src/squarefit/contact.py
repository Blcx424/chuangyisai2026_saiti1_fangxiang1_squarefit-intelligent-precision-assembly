"""方形轴孔的可解释侧壁接触风险分类。

为什么不直接做“1060 类接触识别”？因为那样需要大量极其稀少的离散样本，
而且许多类别只是左右、上下或深度位置的镜像。这里采用更适合本赛题的分层
描述：

    八个侧壁间隙 -> 活跃约束集合 -> 少数风险拓扑

本模块只判断理想直方孔、刚性方轴的侧壁几何风险。它不声称识别真实接触力、
面接触或线接触；这些需要倒角、摩擦、柔顺和力传感器等额外模型。
"""

from __future__ import annotations

import math

from .geometry import peg_corners_at_station
from .types import ContactState, Dimensions, Pose6D, WallClearance


_CORNER_NAMES = ("左下角", "左上角", "右下角", "右上角")
_WALL_ORDER = ("right", "left", "top", "bottom")
_WALL_NAMES = {
    "right": "右壁",
    "left": "左壁",
    "top": "上壁",
    "bottom": "下壁",
}
_STATION_NAMES = {
    "entry": "入口",
    "bottom": "底部",
}


def wall_clearance_details_at_station(
    dimensions: Dimensions,
    pose: Pose6D,
    station_um: float,
    *,
    station: str,
) -> tuple[WallClearance, WallClearance, WallClearance, WallClearance]:
    """计算一个截面到四面孔壁的间隙，并保留限制角点。

    四个方轴角点与四面孔壁本来会形成 16 个候选距离。对每面孔壁只保留
    最小距离即可，因为它就是该面是否会干涉的充分判据；相同最小值的角点
    会一起记录，方便人工核对。
    """
    if station not in _STATION_NAMES:
        raise ValueError("station 只能是 'entry' 或 'bottom'。")

    corners = peg_corners_at_station(dimensions, pose, station_um)
    labelled_corners = tuple(zip(_CORNER_NAMES, corners, strict=True))
    half_hole = dimensions.hole_side_um / 2.0

    def make_detail(wall: str) -> WallClearance:
        if wall == "right":
            extreme = max(point[0] for _, point in labelled_corners)
            clearance = half_hole - extreme
            limiting = tuple(
                name
                for name, point in labelled_corners
                if math.isclose(point[0], extreme, rel_tol=0.0, abs_tol=1e-9)
            )
        elif wall == "left":
            extreme = min(point[0] for _, point in labelled_corners)
            clearance = extreme + half_hole
            limiting = tuple(
                name
                for name, point in labelled_corners
                if math.isclose(point[0], extreme, rel_tol=0.0, abs_tol=1e-9)
            )
        elif wall == "top":
            extreme = max(point[1] for _, point in labelled_corners)
            clearance = half_hole - extreme
            limiting = tuple(
                name
                for name, point in labelled_corners
                if math.isclose(point[1], extreme, rel_tol=0.0, abs_tol=1e-9)
            )
        elif wall == "bottom":
            extreme = min(point[1] for _, point in labelled_corners)
            clearance = extreme + half_hole
            limiting = tuple(
                name
                for name, point in labelled_corners
                if math.isclose(point[1], extreme, rel_tol=0.0, abs_tol=1e-9)
            )
        else:
            raise ValueError(f"未知孔壁：{wall}")

        return WallClearance(
            station=station,
            station_um=station_um,
            wall=wall,
            clearance_um=clearance,
            limiting_corners=limiting,
        )

    details = tuple(make_detail(wall) for wall in _WALL_ORDER)
    return details  # type: ignore[return-value]


def wall_clearance_profile_um(
    dimensions: Dimensions,
    pose: Pose6D,
) -> tuple[WallClearance, ...]:
    """返回入口和底部的八个侧壁间隙。

    对直壁方孔和刚性方轴，每条棱到某孔壁的距离随插入深度线性变化，因此
    最危险位置必在入口或底部。中间截面可在未来加入孔形误差时再额外采样。
    """
    entry = wall_clearance_details_at_station(
        dimensions,
        pose,
        station_um=0.0,
        station="entry",
    )
    bottom = wall_clearance_details_at_station(
        dimensions,
        pose,
        station_um=dimensions.insertion_depth_um,
        station="bottom",
    )
    return (*entry, *bottom)


def _validate_thresholds(contact_tolerance_um: float, warning_clearance_um: float) -> None:
    if contact_tolerance_um < 0.0:
        raise ValueError("接触数值容差不能为负数。")
    if warning_clearance_um < contact_tolerance_um:
        raise ValueError("预警间隙必须不小于接触数值容差。")


def _safety_level(minimum_clearance_um: float, contact_tolerance_um: float, warning_clearance_um: float) -> str:
    if minimum_clearance_um < -contact_tolerance_um:
        return "几何干涉"
    if abs(minimum_clearance_um) <= contact_tolerance_um:
        return "几何接触"
    if minimum_clearance_um <= warning_clearance_um:
        return "接近接触"
    return "安全"


def _topology(active_constraints: tuple[WallClearance, ...]) -> str:
    """按活跃孔壁的几何关系压缩为少数风险类别。"""
    if not active_constraints:
        return "无接触风险"

    active_walls = {item.wall for item in active_constraints}
    if len(active_constraints) == 1:
        return "单侧局部风险"
    if len(active_walls) == 1:
        return "同侧双端风险（近似线接触）"
    if len(active_walls) >= 3:
        return "多壁复杂风险"

    walls = frozenset(active_walls)
    opposite_pairs = (frozenset(("right", "left")), frozenset(("top", "bottom")))
    if walls in opposite_pairs:
        return "对向双壁楔紧风险"
    return "相邻双壁角部风险"


def _axial_location(active_constraints: tuple[WallClearance, ...]) -> str:
    if not active_constraints:
        return "无"

    stations = {item.station for item in active_constraints}
    if stations == {"entry"}:
        return "入口"
    if stations == {"bottom"}:
        return "底部"

    walls = {item.wall for item in active_constraints}
    if len(walls) == 1:
        return "入口与底部同侧"
    if frozenset(walls) in (frozenset(("right", "left")), frozenset(("top", "bottom"))):
        return "入口与底部对侧"
    return "入口与底部"


def _recommendation(safety_level: str, topology: str) -> str:
    if safety_level == "安全":
        return "可执行常规解析补偿，并在下一次测量后验证。"
    if topology == "单侧局部风险":
        return "暂停继续插入，优先沿主导孔壁法向微调平移后复测。"
    if topology == "同侧双端风险（近似线接触）":
        return "暂停继续插入，联合修正对应平移与倾斜角后复测。"
    if topology == "相邻双壁角部风险":
        return "回退至安全位，联合修正 dx、dy 和 θz 后重新测量。"
    if topology == "对向双壁楔紧风险":
        return "回退至安全位，优先修正 θx/θy 与对应平移后重新测量。"
    return "回退至安全位并重新测量；记录该复杂样本用于后续模型改进。"


def classify_lateral_contact_state(
    dimensions: Dimensions,
    pose: Pose6D,
    *,
    contact_tolerance_um: float = 1e-9,
    warning_clearance_um: float = 0.5,
) -> ContactState:
    """对八个侧壁间隙做“安全等级 + 风险拓扑 + 位置”的分层判定。

    ``contact_tolerance_um`` 仅处理浮点数意义上的“约等于零”；
    ``warning_clearance_um`` 是项目可配置的保守预警裕量，而不是赛题强制值。
    等后续测量噪声模型确定后，应由噪声标准差和安全系数重新设定它。
    """
    _validate_thresholds(contact_tolerance_um, warning_clearance_um)
    profile = wall_clearance_profile_um(dimensions, pose)
    minimum = min(profile, key=lambda item: item.clearance_um)
    safety_level = _safety_level(
        minimum.clearance_um,
        contact_tolerance_um,
        warning_clearance_um,
    )

    if safety_level == "安全":
        active_constraints: tuple[WallClearance, ...] = ()
    else:
        active_constraints = tuple(
            item for item in profile if item.clearance_um <= warning_clearance_um
        )

    topology = _topology(active_constraints)
    return ContactState(
        safety_level=safety_level,
        topology=topology,
        axial_location=_axial_location(active_constraints),
        minimum=minimum,
        active_constraints=active_constraints,
        recommendation=_recommendation(safety_level, topology),
        contact_tolerance_um=contact_tolerance_um,
        warning_clearance_um=warning_clearance_um,
    )


def format_contact_state(state: ContactState) -> str:
    """将分类结果格式化为可直接放进控制台的中文说明。"""
    minimum = state.minimum
    limiting_corners = "、".join(minimum.limiting_corners)
    lines = [
        f"  几何安全等级：{state.safety_level}",
        f"  风险拓扑：{state.topology}",
        f"  风险轴向位置：{state.axial_location}",
        (
            "  最小间隙："
            f"{minimum.clearance_um:.6f} 微米（{_STATION_NAMES[minimum.station]}"
            f" {_WALL_NAMES[minimum.wall]}，限制角点：{limiting_corners}）"
        ),
    ]
    if state.active_constraints:
        active_text = "；".join(
            (
                f"{_STATION_NAMES[item.station]}{_WALL_NAMES[item.wall]} "
                f"{item.clearance_um:.6f} 微米"
            )
            for item in state.active_constraints
        )
        lines.append(f"  活跃约束：{active_text}")
    lines.append(f"  建议动作：{state.recommendation}")
    return "\n".join(lines)
