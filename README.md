# SquareFit：方形轴孔智能精密装配仿真与补偿

本仓库提供方形轴孔微小间隙装配的仿真、几何复核、神经网络补偿与安全闭环控制代码。项目面向六维位姿误差下的精密装配场景，以在线可测信息为输入，输出六维补偿指令，并通过几何约束复核间隙与干涉风险。

> 本仓库用于算法研究、仿真验证和教学交流，不代表已部署的实体装配系统，也不包含任何真实生产数据。

## 主要内容

- 方形轴孔四角点几何建模、间隙计算与侧壁干涉判定；
- 六维位姿扰动、测量误差和执行误差的虚拟装配仿真；
- 基于 BP 训练的轻量级 MLP 逆解补偿网络；
- 物理约束损失、解析补偿和安全复核；
- MLP 残余再推理与 DLS 几何修正的双候选闭环控制；
- 小样本目标域适配，以及自动化单元测试。

## 快速开始

建议使用 Python 3.10 或更高版本。

```powershell
git clone https://github.com/<your-account>/squarefit-intelligent-precision-assembly.git
cd squarefit-intelligent-precision-assembly
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

运行几何与解析补偿示例：

```powershell
python main.py --mode demo
python main.py --mode compensate
```

运行自动化测试：

```powershell
python main.py --mode test
```

## 生成、训练与评价

仓库默认不提供完整训练数据、封存集、模型权重和批量评价记录。使用者可按需要在本地生成数据和产物：

```powershell
# 生成仿真数据（默认保存到 data/generated_task_v5_full_feasible）
python main.py --mode dataset

# 训练源域模型
python main.py --mode train-source

# 进行目标域小样本适配
python main.py --mode adapt-target
```

训练和评价产物会写入本地 `data/` 与 `artifacts/` 目录；这两个目录已被 Git 忽略，避免将大规模数据、模型检查点或机器相关路径误提交到公开仓库。

## 项目结构

```text
.
├── src/squarefit/       # 几何、仿真、网络、损失函数与闭环控制核心
├── tests/               # 自动化单元测试
├── main.py              # 命令行入口
├── requirements.txt     # Python 依赖
└── pyproject.toml       # 项目与测试配置
```

## 方法概述

模型以在线可获得的六维测量位姿、轴孔实测尺寸、名义间隙和解析补偿先验为输入，输出平移与转角补偿量。方轴方孔的间隙和干涉关系由几何模型统一复核；当首轮补偿后仍未达标时，控制器基于复测残余生成学习型与几何型候选动作，并只执行满足安全条件、改善更明显的候选。所有闭环动作均受有限步数约束。

## 发布范围与复现实验说明

本公开版仅包含可阅读、可测试、可本地生成实验数据的核心实现。为保护独立封存验收流程并保持仓库轻量，以下内容不随仓库发布：

- 完整训练数据、批量蒙特卡洛样本和独立封存集；
- 训练检查点、目标域适配权重及逐件评价记录；
- 竞赛文档、答辩材料和本机开发环境配置。

如将本项目用于论文、报告或二次开发，请说明该实现来自 SquareFit 项目，并自行完成参数设置、数据验证与安全评估。

## 许可证

本项目采用 [MIT License](LICENSE)。
