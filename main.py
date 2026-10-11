"""项目统一运行入口。

直接在 PyCharm 运行本文件，默认展示三个几何计算示例。
若要运行自动检查，在 Run Configuration 的 Parameters 中填写：
    --mode test
"""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from squarefit import (
    Dimensions,
    Pose6D,
    arcsec_to_rad,
    analytic_compensation,
    classify_lateral_contact_state,
    evaluate_lateral_fit,
    format_compensation_result,
    format_contact_state,
    make_source_domain,
    make_safety_auxiliary_domain,
    make_size_boundary_domain,
    make_stress_domain,
    make_target_domain,
    residual_pose_within_requirement,
    residual_summary,
    save_dataset_bundle,
    save_incoming_pair_screening_csv,
    simulate_many,
    thermal_clearance_sensitivity,
    TrainingConfig,
    run_target_adaptation_experiment,
    run_target_adaptation_replicates,
    train_source_model,
    evaluate_saved_source_model,
    evaluate_frozen_target_models_on_holdout,
    run_closed_loop_experiment,
    run_size_boundary_closed_loop_evaluation,
    run_closed_loop_monte_carlo,
    run_disturbance_robustness_scan,
    run_convergence_stability_analysis,
    evaluate_closed_loop_on_final_holdout,
    run_prediction_horizon_ablation,
)
from squarefit.release_audit import verify_release_package
'''
Dimensions：孔、轴、深度、公差等尺寸；
Pose6D：六维误差；
arcsec_to_rad：角秒转弧度；
evaluate_lateral_fit：计算 g1～g4、最小间隙、碰撞和是否合格
'''


def print_case(name: str, dimensions: Dimensions, pose: Pose6D) -> None:
    """打印一个便于人工核对的几何案例。"""
    result = evaluate_lateral_fit(dimensions, pose)
    """"调用几何模型"""
    gaps = ", ".join(f"{value:.6f}" for value in result.labelled_gaps_um)
    print(name)
    print(f"  g1～g4 间隙（微米）：[{gaps}]")
    print(f"  全部位置中的最小间隙（微米）：{result.minimum_clearance_um:.6f}")
    print(f"  是否发生碰撞：{'是' if result.collision else '否'}")
    print(f"  四个标注间隙是否合格：{'是' if result.labelled_gaps_qualified else '否'}")


def run_demo() -> None:
    """运行第一阶段的三个可解释示例。"""
    dimensions = Dimensions(hole_side_um=10006.0, shaft_side_um=10000.0)
    print("方形轴孔几何模型演示")
    print(f"名义单边间隙：{dimensions.nominal_single_side_clearance_um:.3f} 微米")
    print_case("案例 1：理想居中", dimensions, Pose6D())
    print_case("案例 2：沿 x 方向平移 +0.8 微米", dimensions, Pose6D(dx_um=0.8))
    print_case(
        "案例 3：绕 x 轴旋转 +1 角秒",
        dimensions,
        Pose6D(theta_x_rad=arcsec_to_rad(1.0)),
    )


def run_contact_demo() -> None:
    """展示为什么用少数风险类即可覆盖复杂的方孔侧壁风险。"""
    dimensions = Dimensions(hole_side_um=10006.0, shaft_side_um=10000.0)
    cases = (
        ("案例 A：理想居中", Pose6D()),
        ("案例 B：右壁同侧双端接近", Pose6D(dx_um=2.9)),
        ("案例 C：右上角双壁接近", Pose6D(dx_um=2.9, dy_um=2.9)),
        (
            "案例 D：入口右壁与底部左壁对向楔紧",
            Pose6D(dx_um=2.9, theta_y_rad=arcsec_to_rad(-120.0)),
        ),
    )
    print("方形轴孔侧壁接触风险分类演示")
    print("说明：本演示使用较大扰动来显示各种风险；默认预警间隙为 0.5 微米。")
    for name, pose in cases:
        print(name)
        print(format_contact_state(classify_lateral_contact_state(dimensions, pose)))


def run_compensation_demo() -> None:
    """展示“测得误差 -> 解析补偿 -> 理想补偿后”的完整可核对过程。"""
    dimensions = Dimensions(hole_side_um=10006.0, shaft_side_um=10000.0)
    measured_pose = Pose6D(
        dx_um=0.8,
        dy_um=-0.4,
        dz_um=0.2,
        theta_x_rad=arcsec_to_rad(1.0),
        theta_y_rad=arcsec_to_rad(-0.8),
        theta_z_rad=arcsec_to_rad(0.6),
    )
    before = evaluate_lateral_fit(dimensions, measured_pose)
    compensation = analytic_compensation(measured_pose)
    after = evaluate_lateral_fit(dimensions, compensation.predicted_pose_after)

    print("方形轴孔解析首轮补偿演示")
    print(format_compensation_result(compensation))
    print(
        "  补偿前 g1～g4（微米）："
        f"[{', '.join(f'{value:.6f}' for value in before.labelled_gaps_um)}]"
    )
    print(
        "  补偿后 g1～g4（微米）："
        f"[{', '.join(f'{value:.6f}' for value in after.labelled_gaps_um)}]"
    )
    print(f"  补偿后四个标注间隙是否合格：{'是' if after.labelled_gaps_qualified else '否'}")
    print(
        "  补偿后六维残余位姿是否满足任务书要求："
        f"{'是' if residual_pose_within_requirement(dimensions, compensation.predicted_pose_after) else '否'}"
    )


def _format_pose(name: str, pose: Pose6D) -> str:
    """用赛题使用的微米、角秒单位简洁展示一个六维位姿。"""
    from squarefit import rad_to_arcsec

    return (
        f"  {name}平移 [dx, dy, dz]（微米）："
        f"[{pose.dx_um:.4f}, {pose.dy_um:.4f}, {pose.dz_um:.4f}]\n"
        f"  {name}转角 [θx, θy, θz]（角秒）："
        f"[{rad_to_arcsec(pose.theta_x_rad):.4f}, {rad_to_arcsec(pose.theta_y_rad):.4f}, "
        f"{rad_to_arcsec(pose.theta_z_rad):.4f}]"
    )


def _print_summary(title: str, summary: dict[str, float | int]) -> None:
    print(title)
    print(f"  样本数：{summary['sample_count']}")
    print(
        "  仅使用解析补偿后，残余六维精度达标率："
        f"{float(summary['residual_requirement_pass_rate']) * 100:.2f}%"
    )
    print(f"  解析补偿后碰撞数：{summary['after_analytic_collision_count']}")
    print(
        "  解析补偿后四侧间隙合格率："
        f"{float(summary['after_analytic_gap_qualified_rate']) * 100:.2f}%"
    )
    print(
        "  解析补偿后六维与间隙同时合格率："
        f"{float(summary['after_analytic_fully_qualified_rate']) * 100:.2f}%"
    )
    print(
        "  解析补偿后平均最小间隙裕度（微米）："
        f"{float(summary['mean_after_analytic_minimum_gap_margin_um']):.5f}"
    )
    print(
        "  平均残余平移范数（微米）："
        f"{float(summary['mean_residual_translation_norm_um']):.5f}"
    )
    print(
        "  平均残余转角范数（角秒）："
        f"{float(summary['mean_residual_rotation_norm_arcsec']):.5f}"
    )


def run_simulation_demo(seed: int) -> None:
    """演示一条完整虚拟装配链，并统计少量源域样本。"""
    samples = simulate_many(20, make_source_domain(), seed=seed)
    first = samples[0]
    print("虚拟装配器演示（尚未使用神经网络）")
    print("数据链：真实误差 → 测量误差 → 解析命令 → 虚拟执行器 → 残余误差")
    print(_format_pose("真实初始误差", first.true_pose))
    print(_format_pose("传感器测得误差", first.measured_pose))
    print(_format_pose("解析补偿命令 u_geo", first.analytic_command))
    print(_format_pose("执行后的真实残余", first.residual_pose_after_analytic))
    print(_format_pose("主方案网络的最优完整标签 u*", first.optimal_command))
    print(_format_pose("残差网络对照组标签 Δu", first.residual_command_label))
    print(
        "  这条样本解析补偿后是否满足六维残余精度要求："
        f"{'是' if first.after_analytic_residual_qualified else '否'}"
    )
    _print_summary("\n20 条源域样本的解析补偿基线：", residual_summary(samples))


def run_thermal_sensitivity_demo() -> None:
    """输出材料线膨胀系数敏感性，供报告假设核验使用。"""
    print("材料线膨胀系数敏感性（代表性 10 006 / 10 000 微米轴孔对）")
    print("说明：这是解析量级比较，不替代赛场零件材料标定，也不改变 M0/M1 主数据集。")
    for row in thermal_clearance_sensitivity():
        print(
            f"  {row['material_assumption']}；{row['condition']}\n"
            f"    参考单边间隙：{float(row['reference_clearance_um']):.4f} 微米；"
            f"运行单边间隙：{float(row['effective_clearance_um']):.4f} 微米；"
            f"变化量：{float(row['clearance_shift_um']):+.4f} 微米"
        )


def run_source_training(
    *,
    data_directory: Path,
    output_directory: Path,
    feature_set_name: str,
    epochs: int,
    seed: int,
) -> None:
    """训练第一版源域单逆网络，并输出最重要的验证指标。"""
    print("开始源域预训练：单逆网络 + 精确可导几何物理层")
    print(f"  输入特征组：{feature_set_name}；训练轮数：{epochs}；随机种子：{seed}")
    result = train_source_model(
        data_directory=data_directory,
        output_directory=output_directory,
        config=TrainingConfig(feature_set_name=feature_set_name, epochs=epochs, seed=seed),
    )
    metrics = result["final_validation"]
    assert isinstance(metrics, dict)
    print("源域验证集结果（仅用于检查训练链是否正常，尚未进行目标域小样本适配）：")
    print(f"  联合 PASS 率：{float(metrics['joint_pass_rate']) * 100:.2f}%")
    print(f"  碰撞率：{float(metrics['collision_rate']) * 100:.2f}%")
    print(f"  平均间隙裕度：{float(metrics['mean_gap_margin_um']):.4f} 微米")
    print(f"  总损失：{float(metrics['loss_total']):.6f}")
    print(f"  已保存模型、归一化器与训练记录：{output_directory}")


def run_source_evaluation(*, data_directory: Path, model_directory: Path) -> None:
    """加载已训练源域模型并打印连续补偿精度与物理质量。"""
    metrics = evaluate_saved_source_model(data_directory=data_directory, model_directory=model_directory)
    print("已保存源域模型的独立验证评价：")
    print(f"  联合 PASS 率：{float(metrics['joint_pass_rate']) * 100:.2f}%")
    print(f"  碰撞率：{float(metrics['collision_rate']) * 100:.2f}%")
    print(f"  平均间隙裕度：{float(metrics['mean_gap_margin_um']):.4f} 微米")
    print(f"  补偿平移 MAE / RMSE：{float(metrics['command_translation_mae_um']):.5f} / {float(metrics['command_translation_rmse_um']):.5f} 微米")
    print(f"  补偿转角 MAE / RMSE：{float(metrics['command_rotation_mae_arcsec']):.5f} / {float(metrics['command_rotation_rmse_arcsec']):.5f} 角秒")


def run_target_adaptation(
    *,
    data_directory: Path,
    source_model_directory: Path,
    output_directory: Path,
    adaptation_epochs: int,
) -> None:
    """正式运行固定协议的 M1 zero-shot 与 10/30/50 条小样本适配实验。"""
    print("开始 M1 小样本适配：固定 M0 检查点，分别使用 10 / 30 / 50 条标定数据")
    print("说明：M1 运行集、蒙特卡洛集和压力集全程不参与参数更新。")
    result = run_target_adaptation_experiment(
        data_directory=data_directory,
        source_model_directory=source_model_directory,
        output_directory=output_directory,
        adaptation_epochs=adaptation_epochs,
    )
    zero = result["zero_shot"]
    assert isinstance(zero, dict)
    operational = zero["target_operational_1000"]
    assert isinstance(operational, dict)
    print("  M1 未适配（zero-shot）运行集 PASS："
          f"{float(operational['joint_pass_rate']) * 100:.2f}%")
    adapted = result["adapted"]
    assert isinstance(adapted, dict)
    for size in (10, 30, 50):
        item = adapted[str(size)]
        assert isinstance(item, dict)
        evaluations = item["evaluations"]
        assert isinstance(evaluations, dict)
        metrics = evaluations["target_operational_1000"]
        assert isinstance(metrics, dict)
        print(f"  用 {size} 条标定适配后的 M1 运行集 PASS：{float(metrics['joint_pass_rate']) * 100:.2f}%")
    print(f"  已保存全部 zero-shot、10/30/50、蒙特卡洛、压力集结果：{output_directory}")


def run_target_adaptation_replicate_evaluation(
    *,
    data_directory: Path,
    source_model_directory: Path,
    output_directory: Path,
    adaptation_epochs: int,
) -> None:
    """五组分层重抽样复验，避免把一次标定件抽样当成普遍结论。"""
    print("开始 M1 小样本五随机种子复验：每次从同一 50 条池中分层重抽 10/30/50 条。")
    print("说明：M1 运行集、10000 次蒙特卡洛集和压力集均仅评价，不参与标定更新。")
    result = run_target_adaptation_replicates(
        data_directory=data_directory,
        source_model_directory=source_model_directory,
        output_directory=output_directory,
        adaptation_epochs=adaptation_epochs,
    )
    aggregate = result["aggregate"]
    assert isinstance(aggregate, dict)
    for size in (10, 30, 50):
        metrics = aggregate[str(size)]
        assert isinstance(metrics, dict)
        operational = metrics["target_operational_1000"]
        assert isinstance(operational, dict)
        print(
            f"  {size} 条：M1 运行集 PASS "
            f"{float(operational['joint_pass_rate_mean']) * 100:.2f}% ± "
            f"{float(operational['joint_pass_rate_std']) * 100:.2f}pt"
        )
    print(f"  已保存五次复验明细与汇总：{output_directory}")


def run_closed_loop_monte_carlo_evaluation(
    *,
    data_directory: Path,
    source_model_directory: Path,
    adaptation_directory: Path,
    output_directory: Path,
    seed: int,
) -> None:
    """运行任务书要求的主闭环 N=10000 蒙特卡洛，不与单次推理混淆。"""
    print("开始任务书闭环蒙特卡洛：N=10000，控制器为 C15 MLP 首轮 + 双复测确认多轮几何安全反馈。")
    result = run_closed_loop_monte_carlo(
        data_directory=data_directory,
        source_model_directory=source_model_directory,
        adaptation_directory=adaptation_directory,
        output_directory=output_directory,
        seed=seed,
    )
    metrics = result["method"]
    assert isinstance(metrics, dict)
    print(
        f"  联合 PASS：{float(metrics['joint_pass_rate']) * 100:.2f}%；"
        f"碰撞率：{(1.0 - float(metrics['no_collision_rate'])) * 100:.2f}%；"
        f"平均补偿动作：{float(metrics['mean_actions']):.2f}"
    )
    print(f"  已保存逐件间隙、分布汇总和最坏样本：{output_directory}")


def run_disturbance_scan_evaluation(
    *,
    data_directory: Path,
    source_model_directory: Path,
    adaptation_directory: Path,
    output_directory: Path,
    seed: int,
    operational_limit: int,
) -> None:
    """运行 1x/1.5x/2x 随机测量与执行扰动的四方法鲁棒性扫描。"""
    print("开始 M1 抗扰扫描：固定同一真实初始状态，只放大测量与执行随机噪声至 1x / 1.5x / 2x。")
    result = run_disturbance_robustness_scan(
        data_directory=data_directory,
        source_model_directory=source_model_directory,
        adaptation_directory=adaptation_directory,
        output_directory=output_directory,
        operational_limit=operational_limit,
        seed=seed,
    )
    summaries = result["summaries"]
    assert isinstance(summaries, dict)
    for scenario, summary in summaries.items():
        assert isinstance(summary, list)
        main_method = next(item for item in summary if item["method"] == "mlp_geometry_closed_loop")
        print(f"  {scenario}：主方法 PASS {float(main_method['joint_pass_rate']) * 100:.2f}%；"
              f"过程碰撞 {float(main_method['post_action_collision_rate']) * 100:.2f}%")
    print(f"  已保存四方法鲁棒性记录与汇总：{output_directory}")


def run_final_holdout_generation(output_dir: Path, seed: int) -> None:
    """生成一次性封存的 M1 独立测试集；不覆盖协议 v3 主数据包。"""
    domain = make_target_domain()
    samples = simulate_many(1000, domain, seed=seed)
    metadata = save_dataset_bundle(
        output_dir,
        (("target_final_holdout_1000.csv", samples, domain, True),),
        base_seed=seed,
    )
    quality = metadata["files"]["target_final_holdout_1000.csv"]["quality_gate"]
    print("已生成独立 M1 封存测试集：")
    print(f"  样本数：{quality['sample_count']}；随机种子：{seed}")
    print(f"  教师联合 PASS：{float(quality['teacher_joint_pass_rate']) * 100:.1f}%")
    print("  此文件只允许用于冻结模型的最终验收，不参与任何后续训练或调参。")


def run_final_holdout_evaluation(
    *,
    holdout_directory: Path,
    source_model_directory: Path,
    adaptation_directory: Path,
) -> None:
    """在封存集上对 zero-shot、10/30/50 条冻结模型进行一次性最终验收。"""
    result = evaluate_frozen_target_models_on_holdout(
        holdout_csv=holdout_directory / "target_final_holdout_1000.csv",
        source_model_directory=source_model_directory,
        adaptation_directory=adaptation_directory,
    )
    methods = result["methods"]
    assert isinstance(methods, dict)
    print("冻结模型在独立 M1 封存测试集上的最终验收：")
    for name in ("zero_shot", "adapt_10", "adapt_30", "adapt_50"):
        metrics = methods[name]
        assert isinstance(metrics, dict)
        print(f"  {name}：联合 PASS {float(metrics['joint_pass_rate']) * 100:.2f}%；"
              f"碰撞率 {float(metrics['collision_rate']) * 100:.2f}%")
    print(f"  已保存最终结果：{adaptation_directory / 'final_m1_holdout_evaluation.json'}")
    closed_loop = evaluate_closed_loop_on_final_holdout(
        holdout_csv=holdout_directory / "target_final_holdout_1000.csv",
        source_model_directory=source_model_directory,
        adaptation_directory=adaptation_directory,
        output_directory=holdout_directory / "closed_loop_final_evaluation",
    )
    methods = closed_loop["methods"]
    assert isinstance(methods, list)
    print("冻结闭环控制器在同一封存集上的最终验收：")
    names = {
        "analytic_open": "解析开环",
        "mlp_open": "C15 MLP 开环",
        "mlp_open_fused_initial": "C15 MLP 开环（双初测融合）",
        "pid_feedback": "传统六维 PID",
        "mlp_geometry_closed_loop": "C15 MLP + 双复测确认多轮几何安全闭环",
    }
    for item in methods:
        assert isinstance(item, dict)
        print(f"  {names[str(item['method'])]}：联合 PASS {float(item['joint_pass_rate']) * 100:.2f}%；"
              f"平均动作 {float(item['mean_actions']):.2f} 次")
    print(f"  已保存闭环逐件与汇总结果：{holdout_directory / 'closed_loop_final_evaluation'}")


def run_closed_loop(
    *,
    data_directory: Path,
    source_model_directory: Path,
    adaptation_directory: Path,
    output_directory: Path,
    seed: int,
    operational_limit: int,
    stress_limit: int,
) -> None:
    """运行闭环开发比较，明确不使用已封存的最终 M1 测试集。"""
    print("开始在线闭环开发评估：测量 → MLP 推理 → 补偿 → 复测验证")
    print("主方法：C15 MLP 首轮补偿 + 双复测确认的多轮几何安全反馈；每件最多 3 次动作。")
    result = run_closed_loop_experiment(
        data_directory=data_directory,
        source_model_directory=source_model_directory,
        adaptation_directory=adaptation_directory,
        output_directory=output_directory,
        operational_limit=operational_limit,
        stress_limit=stress_limit,
        seed=seed,
    )
    operational = result["operational"]
    assert isinstance(operational, list)
    print("M1 开发运行集结果（用于固定控制器，不能当封存最终成绩）：")
    names = {
        "analytic_open": "解析开环",
        "mlp_open": "C15 MLP 开环",
        "mlp_open_fused_initial": "C15 MLP 开环（双初测融合，仅测量策略消融）",
        "pid_feedback": "传统六维 PID 反馈",
        "mlp_geometry_closed_loop": "MLP + 几何安全闭环（主方法）",
    }
    for item in operational:
        assert isinstance(item, dict)
        print(
            f"  {names[str(item['method'])]}：联合 PASS {float(item['joint_pass_rate']) * 100:.2f}%；"
            f"平均动作 {float(item['mean_actions']):.2f} 次；"
            f"平均间隙裕度 {float(item['mean_gap_margin_um']):.3f} 微米"
        )
    print(f"已保存逐件记录、汇总结果：{output_directory}")


def run_horizon_ablation(
    *,
    data_directory: Path,
    source_model_directory: Path,
    adaptation_directory: Path,
    output_directory: Path,
    seed: int,
    operational_limit: int,
) -> None:
    """检验 H=1 单步预测是否足够，不把消融结果混入最终测试。"""
    print("开始 H=1/2/3 几何预测时域消融：相同 MLP 首轮、相同三次动作上限与噪声轨迹。")
    result = run_prediction_horizon_ablation(
        data_directory=data_directory,
        source_model_directory=source_model_directory,
        adaptation_directory=adaptation_directory,
        output_directory=output_directory,
        operational_limit=operational_limit,
        seed=seed,
    )
    summary = result["summary"]
    assert isinstance(summary, list)
    for item in summary:
        assert isinstance(item, dict)
        print(
            f"  {item['method']}：联合 PASS {float(item['joint_pass_rate']) * 100:.2f}%；"
            f"平均动作 {float(item['mean_actions']):.2f} 次；"
            f"平均决策 {float(item['mean_decision_time_ms']):.2f} ms"
        )
    print(f"已保存消融逐件记录与汇总：{output_directory}")


def run_convergence_stability_evaluation(
    *,
    data_directory: Path,
    source_model_directory: Path,
    adaptation_directory: Path,
    output_directory: Path,
    seed: int,
    operational_limit: int,
) -> None:
    """输出主闭环逐动作收敛与安全稳定性证据。"""
    print("开始主闭环收敛/稳定性分析：记录每次真实动作后的违规量、PASS 与安全状态。")
    result = run_convergence_stability_analysis(
        data_directory=data_directory,
        source_model_directory=source_model_directory,
        adaptation_directory=adaptation_directory,
        output_directory=output_directory,
        operational_limit=operational_limit,
        seed=seed,
    )
    transition = result["transition_summary"]
    assert isinstance(transition, dict)
    summary = result["per_action_summary"]
    assert isinstance(summary, list)
    for item in summary:
        assert isinstance(item, dict)
        print(
            f"  动作后 {int(item['action_index'])}：样本 {int(item['sample_count'])}；"
            f"平均违规量 {float(item['mean_violation_score']):.4f}；"
            f"联合 PASS {float(item['joint_pass_rate']) * 100:.2f}%"
        )
    print(
        f"  相邻动作违规量非增比例：{float(transition['nonincreasing_transition_rate']) * 100:.2f}%；"
        f"过程碰撞：{int(transition['process_collision_count'])} 件"
    )
    print(f"已保存逐动作轨迹与收敛汇总：{output_directory}")


def run_dataset_generation(
    output_dir: Path,
    seed: int,
    source_samples: int,
    source_validation_samples: int,
    source_safety_samples: int,
    calibration_samples: int,
    target_test_samples: int,
    stress_samples: int,
    monte_carlo_samples: int,
    size_boundary_samples: int,
    incoming_screening_samples: int,
) -> None:
    """按 v5 完整可装配域协议生成数据；这里不训练任何网络。"""
    counts = (
        source_samples,
        source_validation_samples,
        source_safety_samples,
        calibration_samples,
        target_test_samples,
        stress_samples,
        monte_carlo_samples,
        size_boundary_samples,
        incoming_screening_samples,
    )
    if min(counts) <= 0:
        raise ValueError("所有数据集与来料筛查样本数都必须为正整数。")
    if calibration_samples < 50:
        raise ValueError("正式小样本标定集至少生成 50 条，供嵌套的 10/30/50 实验使用。")
    if monte_carlo_samples < 10_000:
        raise ValueError("正式蒙特卡洛数据集必须不少于 10000 条。")

    source_domain = make_source_domain()
    safety_auxiliary_domain = make_safety_auxiliary_domain()
    target_domain = make_target_domain()
    size_boundary_domain = make_size_boundary_domain()
    stress_domain = make_stress_domain()
    source = simulate_many(source_samples, source_domain, seed=seed, sampling_strategy="full_feasible_coverage")
    source_validation = simulate_many(source_validation_samples, source_domain, seed=seed + 1, sampling_strategy="full_feasible_coverage")
    source_safety = simulate_many(
        source_safety_samples,
        safety_auxiliary_domain,
        seed=seed + 5,
        sampling_strategy="safety_stress",
    )
    calibration = simulate_many(
        calibration_samples,
        target_domain,
        seed=seed + 2,
        sampling_strategy="calibration",
    )
    target_test = simulate_many(target_test_samples, target_domain, seed=seed + 3)
    stress_test = simulate_many(
        stress_samples,
        stress_domain,
        seed=seed + 4,
        sampling_strategy="safety_stress",
    )
    monte_carlo = simulate_many(monte_carlo_samples, target_domain, seed=seed + 100)
    size_boundary = simulate_many(size_boundary_samples, size_boundary_domain, seed=seed + 6, sampling_strategy="size_boundary")
    splits = (
        ("source_pretrain.csv", source, source_domain, True),
        ("source_validation.csv", source_validation, source_domain, True),
        ("source_safety_auxiliary.csv", source_safety, safety_auxiliary_domain, False),
        ("target_calibration_50.csv", calibration, target_domain, True),
        ("target_operational_test.csv", target_test, target_domain, True),
        ("target_size_boundary_test.csv", size_boundary, size_boundary_domain, True),
        ("target_stress_test.csv", stress_test, stress_domain, False),
        ("monte_carlo_operational_10000.csv", monte_carlo, target_domain, True),
    )
    print("开始生成 v5 完整可装配域虚拟装配数据包（本步骤不训练神经网络）")
    incoming_quality = save_incoming_pair_screening_csv(output_dir / "incoming_pair_screening.csv", count=incoming_screening_samples, seed=seed + 200)
    metadata = save_dataset_bundle(output_dir, splits, base_seed=seed, incoming_screening_quality=incoming_quality)
    for filename, samples, _, _ in splits:
        print(f"  已保存 {filename}：{len(samples)} 条 → {output_dir / filename}")
        _print_summary(f"  {filename} 的解析补偿基线：", residual_summary(samples))
        if filename == "target_stress_test.csv":
            print("  注意：Mstress 仅用于安全/鲁棒性压力检查；其单次残余达标率受高噪声地板限制，不得作为神经网络主性能对比。")
    print(f"  已保存 metadata.json，记录参数、种子、SHA-256 与质量门禁。文件数：{len(metadata['files'])}")
    print(f"  来料尺寸筛查：可由位姿补偿的选配对 {incoming_quality['pose_compensation_feasible_count']} 条；需重新选配 {incoming_quality['requires_rematching_count']} 条。")
    print("\n网络输入仅可使用六维 measured_*、实测轴孔尺寸、选配间隙和可选 analytic_command_*；")
    print("label_optimal_command_* 是主方案六维教师标签；true_* 严禁进入网络。")


def run_tests() -> int:
    """运行 tests 文件夹中的全部自动检查。"""
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(PROJECT_ROOT / "tests"),
        pattern="test_*.py",
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="方形轴孔智能精密装配项目入口")
    parser.add_argument(
        "--mode",
        choices=("demo", "contact", "compensate", "simulate", "thermal-sensitivity", "dataset", "train-source", "evaluate-source", "adapt-target", "adapt-target-replicates", "closed-loop", "closed-loop-size-boundary", "closed-loop-monte-carlo", "disturbance-scan", "convergence-stability", "horizon-ablation", "make-final-holdout", "evaluate-final-holdout", "verify-release", "test"),
        default="demo",
        help=(
            "demo: 查看三个计算示例；contact: 查看风险分类；"
            "compensate: 查看解析补偿；simulate: 查看虚拟装配链；"
            "thermal-sensitivity: 查看材料热膨胀敏感性；dataset: 生成 CSV 数据集；"
            "train-source: 源域预训练单逆网络；evaluate-source: 评价已保存源域模型；"
            "adapt-target: 运行 M1 的 zero-shot 与 10/30/50 条小样本适配；"
            "adapt-target-replicates: 五次分层重抽样复验小样本适配；"
            "closed-loop: 运行 M1 开发闭环对照；"
            "closed-loop-size-boundary: 运行 Msize 完整可行域边界闭环验收；"
            "closed-loop-monte-carlo: 运行任务书要求的 N=10000 主闭环蒙特卡洛；"
            "disturbance-scan: 运行 1x/1.5x/2x 测量和执行随机扰动鲁棒性扫描；"
            "convergence-stability: 输出主闭环逐动作收敛与过程安全证据；"
            "horizon-ablation: 比较 H=1/2/3 几何预测时域；"
            "make-final-holdout: 生成封存的 M1 最终测试集；"
            "evaluate-final-holdout: 一次性验收冻结模型；"
            "verify-release: 校验随包 CSV 哈希、C15 白名单和适配汇总；"
            "test: 运行全部自动检查。"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260912, help="仿真随机种子，默认可复现。")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "generated_task_v5_full_feasible",
        help="--mode dataset 时 CSV 的保存目录。",
    )
    parser.add_argument(
        "--final-holdout-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "final_holdout_task_v5_final",
        help="最终 M1 封存测试集的目录；不得与训练或开发数据混用。",
    )
    parser.add_argument(
        "--final-holdout-seed",
        type=int,
        default=20261013,
        help="生成封存测试集的独立随机种子。",
    )
    parser.add_argument(
        "--target-adaptation-output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "task_v5" / "target_adaptation_c15",
        help="--mode adapt-target 时保存 M1 小样本适配检查点与评价结果的位置。",
    )
    parser.add_argument(
        "--target-adaptation-replicates-output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "task_v5" / "target_adaptation_replicates_c15",
        help="--mode adapt-target-replicates 时保存五随机种子复验结果的位置。",
    )
    parser.add_argument(
        "--closed-loop-output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "task_v5" / "closed_loop_c15",
        help="--mode closed-loop 时保存逐件闭环记录和汇总的位置。",
    )
    parser.add_argument(
        "--closed-loop-size-boundary-output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "task_v5" / "closed_loop_size_boundary_c15_contact_guard",
        help="--mode closed-loop-size-boundary 时保存 Msize 边界闭环结果的位置。",
    )
    parser.add_argument(
        "--closed-loop-monte-carlo-output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "task_v5" / "closed_loop_monte_carlo_c15",
        help="--mode closed-loop-monte-carlo 时保存 N=10000 主闭环蒙特卡洛结果的位置。",
    )
    parser.add_argument(
        "--disturbance-scan-output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "task_v5" / "disturbance_scan_c15",
        help="--mode disturbance-scan 时保存四方法随机扰动鲁棒性扫描结果的位置。",
    )
    parser.add_argument(
        "--horizon-ablation-output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "task_v5" / "horizon_ablation_c15",
        help="--mode horizon-ablation 时保存 H=1/2/3 消融结果的位置。",
    )
    parser.add_argument(
        "--convergence-stability-output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "task_v5" / "convergence_stability_c15",
        help="--mode convergence-stability 时保存主闭环逐动作收敛证据的位置。",
    )
    parser.add_argument(
        "--closed-loop-operational-limit",
        type=int,
        default=1000,
        help="--mode closed-loop 使用的 M1 开发运行集样本数，默认 1000。",
    )
    parser.add_argument(
        "--closed-loop-stress-limit",
        type=int,
        default=1000,
        help="--mode closed-loop 使用的 Mstress 安全集样本数，默认 1000。",
    )
    parser.add_argument(
        "--adaptation-epochs",
        type=int,
        default=80,
        help="--mode adapt-target 时每种标定规模固定采用的微调轮数。",
    )
    parser.add_argument("--source-samples", type=int, default=5000, help="源域预训练数据量。")
    parser.add_argument("--source-validation-samples", type=int, default=1000, help="源域验证数据量。")
    parser.add_argument("--source-safety-samples", type=int, default=2000, help="源域越界安全辅助数据量，只用于物理约束课程训练。")
    parser.add_argument("--calibration-samples", type=int, default=50, help="目标域嵌套分层空间覆盖标定数据量，正式值至少为 50。")
    parser.add_argument("--target-test-samples", type=int, default=1000, help="目标域生产分布独立测试数据量。")
    parser.add_argument("--stress-samples", type=int, default=1000, help="Mstress 压力测试数据量。")
    parser.add_argument("--size-boundary-samples", type=int, default=1000, help="Msize 尺寸边界测试集数量，完整可行域两端各占一半。")
    parser.add_argument("--incoming-screening-samples", type=int, default=10000, help="独立来料尺寸筛查数；不可补偿件仅标记重选配，不进入网络。")
    parser.add_argument("--monte-carlo-samples", type=int, default=10000, help="最终生产分布蒙特卡洛数据量，正式值至少为 10000。")
    parser.add_argument(
        "--feature-set",
        default="C_解析先验BP_15维",
        help="--mode train-source 时使用的特征白名单名称；C15 是 C_解析先验BP_15维 的简称。",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=120,
        help="--mode train-source 时的源域训练轮数。",
    )
    parser.add_argument(
        "--training-output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "task_v5" / "source_pretrain_c15",
        help="--mode train-source 时保存模型、归一化器和训练记录的位置。",
    )
    arguments = parser.parse_args()
    if arguments.mode == "test":
        return run_tests()
    if arguments.mode == "verify-release":
        result = verify_release_package(PROJECT_ROOT)
        print("最终提交包完整性校验通过。")
        for key, value in result.items():
            print(f"  {key}: {value}")
        return 0
    if arguments.mode == "contact":
        run_contact_demo()
        return 0
    if arguments.mode == "compensate":
        run_compensation_demo()
        return 0
    if arguments.mode == "thermal-sensitivity":
        run_thermal_sensitivity_demo()
        return 0
    if arguments.mode == "simulate":
        run_simulation_demo(arguments.seed)
        return 0
    if arguments.mode == "dataset":
        run_dataset_generation(
            output_dir=arguments.output_dir,
            seed=arguments.seed,
            source_samples=arguments.source_samples,
            source_validation_samples=arguments.source_validation_samples,
            source_safety_samples=arguments.source_safety_samples,
            calibration_samples=arguments.calibration_samples,
            target_test_samples=arguments.target_test_samples,
            stress_samples=arguments.stress_samples,
            monte_carlo_samples=arguments.monte_carlo_samples,
            size_boundary_samples=arguments.size_boundary_samples,
            incoming_screening_samples=arguments.incoming_screening_samples,
        )
        return 0
    if arguments.mode == "train-source":
        if arguments.epochs <= 0:
            raise ValueError("训练轮数 epochs 必须为正整数。")
        run_source_training(
            data_directory=arguments.output_dir,
            output_directory=arguments.training_output_dir,
            feature_set_name=arguments.feature_set,
            epochs=arguments.epochs,
            seed=arguments.seed,
        )
        return 0
    if arguments.mode == "evaluate-source":
        run_source_evaluation(
            data_directory=arguments.output_dir,
            model_directory=arguments.training_output_dir,
        )
        return 0
    if arguments.mode == "adapt-target":
        if arguments.adaptation_epochs <= 0:
            raise ValueError("适配轮数 adaptation-epochs 必须为正整数。")
        run_target_adaptation(
            data_directory=arguments.output_dir,
            source_model_directory=arguments.training_output_dir,
            output_directory=arguments.target_adaptation_output_dir,
            adaptation_epochs=arguments.adaptation_epochs,
        )
        return 0
    if arguments.mode == "adapt-target-replicates":
        if arguments.adaptation_epochs <= 0:
            raise ValueError("适配轮数 adaptation-epochs 必须为正整数。")
        run_target_adaptation_replicate_evaluation(
            data_directory=arguments.output_dir,
            source_model_directory=arguments.training_output_dir,
            output_directory=arguments.target_adaptation_replicates_output_dir,
            adaptation_epochs=arguments.adaptation_epochs,
        )
        return 0
    if arguments.mode == "closed-loop":
        run_closed_loop(
            data_directory=arguments.output_dir,
            source_model_directory=arguments.training_output_dir,
            adaptation_directory=arguments.target_adaptation_output_dir,
            output_directory=arguments.closed_loop_output_dir,
            seed=arguments.seed,
            operational_limit=arguments.closed_loop_operational_limit,
            stress_limit=arguments.closed_loop_stress_limit,
        )
        return 0
    if arguments.mode == "closed-loop-size-boundary":
        result = run_size_boundary_closed_loop_evaluation(
            data_directory=arguments.output_dir,
            source_model_directory=arguments.training_output_dir,
            adaptation_directory=arguments.target_adaptation_output_dir,
            output_directory=arguments.closed_loop_size_boundary_output_dir,
            seed=arguments.seed,
        )
        print(f"Msize 边界闭环验收完成：{result['sample_count']} 件。")
        return 0
    if arguments.mode == "closed-loop-monte-carlo":
        run_closed_loop_monte_carlo_evaluation(
            data_directory=arguments.output_dir,
            source_model_directory=arguments.training_output_dir,
            adaptation_directory=arguments.target_adaptation_output_dir,
            output_directory=arguments.closed_loop_monte_carlo_output_dir,
            seed=arguments.seed,
        )
        return 0
    if arguments.mode == "disturbance-scan":
        run_disturbance_scan_evaluation(
            data_directory=arguments.output_dir,
            source_model_directory=arguments.training_output_dir,
            adaptation_directory=arguments.target_adaptation_output_dir,
            output_directory=arguments.disturbance_scan_output_dir,
            seed=arguments.seed,
            operational_limit=arguments.closed_loop_operational_limit,
        )
        return 0
    if arguments.mode == "horizon-ablation":
        run_horizon_ablation(
            data_directory=arguments.output_dir,
            source_model_directory=arguments.training_output_dir,
            adaptation_directory=arguments.target_adaptation_output_dir,
            output_directory=arguments.horizon_ablation_output_dir,
            seed=arguments.seed,
            operational_limit=arguments.closed_loop_operational_limit,
        )
        return 0
    if arguments.mode == "convergence-stability":
        run_convergence_stability_evaluation(
            data_directory=arguments.output_dir,
            source_model_directory=arguments.training_output_dir,
            adaptation_directory=arguments.target_adaptation_output_dir,
            output_directory=arguments.convergence_stability_output_dir,
            seed=arguments.seed,
            operational_limit=arguments.closed_loop_operational_limit,
        )
        return 0
    if arguments.mode == "make-final-holdout":
        run_final_holdout_generation(arguments.final_holdout_dir, arguments.final_holdout_seed)
        return 0
    if arguments.mode == "evaluate-final-holdout":
        run_final_holdout_evaluation(
            holdout_directory=arguments.final_holdout_dir,
            source_model_directory=arguments.training_output_dir,
            adaptation_directory=arguments.target_adaptation_output_dir,
        )
        return 0
    run_demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
