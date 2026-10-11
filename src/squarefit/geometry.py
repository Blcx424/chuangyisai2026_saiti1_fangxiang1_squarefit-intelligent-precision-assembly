"""方形轴孔的可解释几何模型。

坐标约定：
* 方孔中心线为 z 轴，入口截面为 z=0，底部截面为 z=L；
* +x 指向孔的右壁，+y 指向孔的上壁；
* 方轴局部坐标的入口端为 z=0，沿 +z 插入；
* g1=入口上壁间隙，g2=入口右壁间隙；
  g3=底部上壁间隙，g4=底部右壁间隙。

这里的 g1--g4 是赛题图中标注的四个间隙。计算碰撞时仍检查四面墙，
而不是只检查这四个被标注的边。
"""

from __future__ import annotations

from collections.abc import Iterable

from .transforms import Vector3, transform_point
from .types import AssemblyResult, Dimensions, Pose6D
from .units import arcsec_to_rad


def peg_corners_at_station(
    dimensions: Dimensions,
    pose: Pose6D,
    station_um: float,
) -> tuple[Vector3, Vector3, Vector3, Vector3]:
    """返回方轴在给定局部 z 截面处的四个角点世界坐标"""
    half_side = dimensions.shaft_side_um / 2.0
    local_corners = (
        (-half_side, -half_side, station_um),
        (-half_side, half_side, station_um),
        (half_side, -half_side, station_um),
        (half_side, half_side, station_um),
    )
    return tuple(transform_point(pose, point) for point in local_corners)  # type: ignore[return-value]
"""建立方轴某个截面的四个角点  接着它调用三维旋转和位移，把“理想状态下的角点”变成“有六维误差后的实际角点"""

def wall_clearances_at_station(
    dimensions: Dimensions,
    pose: Pose6D,
    station_um: float,
) -> dict[str, float]:
    """计算该截面至方孔四面侧壁的最小法向间隙。 它计算该截面到四面孔壁的距离
    右壁间隙 = D/2 - 所有角点中最大的 x
    左壁间隙 = 所有角点中最小的 x + D/2
    上壁间隙 = D/2 - 所有角点中最大的 y
    下壁间隙 = 所有角点中最小的 y + D/2

    返回 right、left、top、bottom。任一值小于等于零意味着与该壁相交或接触。
    结果大于 0：没有碰撞。
    结果等于 0：刚好接触。
    结果小于 0：已经穿进孔壁，发生碰撞。
    """
    corners = peg_corners_at_station(dimensions, pose, station_um)
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    half_hole = dimensions.hole_side_um / 2.0
    return {
        "right": half_hole - max(xs),
        "left": min(xs) + half_hole,
        "top": half_hole - max(ys),
        "bottom": min(ys) + half_hole,
    }


def labelled_gaps_um(dimensions: Dimensions, pose: Pose6D) -> tuple[float, float, float, float]:
    """按 (g1, g2, g3, g4) 计算赛题标注的四个间隙。
       入口：取上侧和右侧 → g1、g2
       底部：取上侧和右侧 → g3、g4
    """
    entry = wall_clearances_at_station(dimensions, pose, station_um=0.0)
    bottom = wall_clearances_at_station(
        dimensions,
        pose,
        station_um=dimensions.insertion_depth_um,
    )
    return entry["top"], entry["right"], bottom["top"], bottom["right"]


def minimum_clearance_um(
    dimensions: Dimensions,
    pose: Pose6D,
    stations_um: Iterable[float] | None = None,
) -> float:
    """返回被检查截面中四面侧壁的最小间隙。

    对刚性直方轴，横向极值会出现在端部；默认同时保留中间截面，便于后续
    扩展为非直孔或引入形状误差时复用接口。

    检查入口、中间、底部三个截面，以及每个截面的四面孔壁，取最小值作为安全指标
    """
    if stations_um is None:
        stations_um = (0.0, dimensions.insertion_depth_um / 2.0, dimensions.insertion_depth_um)
    values: list[float] = []
    for station_um in stations_um:
        values.extend(wall_clearances_at_station(dimensions, pose, station_um).values())
    if not values:
        raise ValueError("至少需要检查一个截面。")
    return min(values)


def evaluate_lateral_fit(dimensions: Dimensions, pose: Pose6D) -> AssemblyResult:
    """计算四个标注间隙、横向碰撞及其是否都位于目标区间。
       判定：四个 g 是否都在 2～4 微米？
            全部位置是否都没有碰撞？
         全部满足再进行下一步内容
    """
    g1, g2, g3, g4 = labelled_gaps_um(dimensions, pose)
    minimum = minimum_clearance_um(dimensions, pose)
    collision = minimum <= 0.0
    qualified = (
        not collision
        and all(
            dimensions.gap_lower_um <= gap <= dimensions.gap_upper_um
            for gap in (g1, g2, g3, g4)
        )
    )
    return AssemblyResult(
        g1_um=g1,
        g2_um=g2,
        g3_um=g3,
        g4_um=g4,
        minimum_clearance_um=minimum,
        collision=collision,
        labelled_gaps_qualified=qualified,
    )


def residual_pose_within_requirement(dimensions: Dimensions, pose: Pose6D) -> bool:
    """按任务书残余位姿阈值检查六维姿态。

    这是对补偿后残余误差的独立判定。理想直孔的横向间隙本身不受 dz 平移
    影响，但 dz 仍必须保留在六维状态和残余精度检查中。

    它检查补偿后的六维残余误差是否满足任务书阈值：
    dx、dy、dz：都不超过 0.1 微米
    θx、θy、θz：都不超过 0.2 角秒
    """
    translation_ok = all(
        abs(value) <= dimensions.residual_translation_limit_um
        for value in (pose.dx_um, pose.dy_um, pose.dz_um)
    )
    rotation_limit_rad = arcsec_to_rad(dimensions.residual_rotation_limit_arcsec)
    rotation_ok = all(
        abs(value) <= rotation_limit_rad
        for value in (pose.theta_x_rad, pose.theta_y_rad, pose.theta_z_rad)
    )
    return translation_ok and rotation_ok
