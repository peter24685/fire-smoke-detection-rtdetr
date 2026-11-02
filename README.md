# RT-DETR 火焰与烟雾检测

基于 Ultralytics RT-DETR-L 的课程项目与工程整理实践，提供数据配置、训练、验证和图片预测入口。作者：peter24685。

**当前状态：已有两次 1 epoch 流程实验，尚未完成充分训练或实际消防场景验证。** 本项目的贡献是数据与配置整理、模型应用、运行调试和实验记录，不包含原创检测算法。代码整理使用 AI 辅助，历史实验数据来自保留的本地运行记录。

## 项目内容

- 将原先固定路径的三个脚本统一为参数化命令行入口。
- 固定 `0=smoke、1=fire` 类别顺序，生成适配本机的 YAML 配置。
- 保留历史训练 CSV，明确区分验证集指标和测试集结果。
- 支持图片或本地目录预测，保存带检测框的结果。

## 环境

本机验证环境：Windows、Python 3.10.20、PyTorch 2.7.1+cu118、Ultralytics 8.4.55。CUDA 可用。先按 [PyTorch 官方说明](https://pytorch.org/get-started/locally/)安装适配本机的 PyTorch，再安装应用依赖：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python main.py --help
```

没有 NVIDIA GPU 时可使用 PyTorch CPU 安装源，运行时传入 `--device cpu`。新建环境的安装命令尚未在干净机器上验证；实际推理检查使用上述已有环境。

## 数据准备

数据来源线索为 [D-Fire](https://github.com/gaia-solutions-on-demand/DFireDataset) 及其预划分版本。仓库不重分发原始图片或标注；请从数据作者提供的入口获取并遵守相应使用条件。当前本地版本的精确下载包及原始校验值未保留，不能将它视为已锁定版本的公开基准。

将本地数据组织为：

```text
dataset-root/
  train/images/   train/labels/
  val/images/     val/labels/
  test/images/    test/labels/
```

```powershell
python main.py prepare --root "D:/datasets/fire-smoke"
```

生成的 `configs/local.yaml` 不会进入 Git。新数据集必须先核对标签 ID，不能只改类别名称来迁就标签。现有配置顺序来自实际使用的训练配置，并做了样例抽查，尚未逐图核验所有类别。

## 训练与验证

```powershell
# 1 epoch 用于检查流程；历史结果也是 1 epoch，并非收敛结果。
python main.py train --weights rtdetr-l.pt --epochs 1 --device 0 --name baseline

# 实际保存目录以终端输出为准；框架在目录重名时会自动递增名称。
python main.py val --weights "runs/baseline/weights/best.pt" --split val --device 0
```

`rtdetr-l.pt` 为框架提供的通用预训练模型。火焰烟雾预测需要本项目数据训练后的权重。本仓库暂不提供可下载的自训练权重；须自行训练或使用已保存的本地权重。需要测试集评估时显式使用 `--split test`，不要使用测试集反复调参。

## 预测

```powershell
python main.py predict --weights "runs/baseline/weights/best.pt" --source "examples/input.jpg" --device cpu --name demo
```

`examples/input.jpg` 是待替换的示例路径，并非仓库附带图片。可以换成自己的图片或图片目录。检测结果保存在终端报告的 `save_dir`。

## 历史实验结果

以下为原训练过程中 **验证集** CSV 最后一行；没有在本次整理中重新训练，也不是独立测试集成绩。两个实验均为 1 epoch、640 输入、batch=2。

| 运行 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| test_run-3 | 0.51105 | 0.47722 | 0.43786 | 0.18581 |
| test_run-4 | 0.51555 | 0.45115 | 0.46118 | 0.20082 |

原始记录见 [实验资料](docs/experiments.md) 和 `reports/`。模型存在漏检、误检的可能，不能用于实际火灾报警决策。

![test_run-3 验证集混淆矩阵](reports/confusion_matrix_run3.png)

该图来自原训练目录，展示该次评估阈值下的分类混淆和背景误检情况，不等同于上表各指标的计算阈值。

## 检查与后续工作

```powershell
python -m unittest discover -s tests -v
```

检查覆盖数据目录验证、类别顺序与默认验证集选择。实际推理验证见 [运行验证记录](docs/verification.md)。

下一阶段：锁定数据版本和划分，核验标签语义，开展更充分训练，统计误检与漏检，最后在固定测试集评估。尚未实现实时报警、视频监控服务、Web 页面或生产部署。

## 来源与许可

- [Ultralytics RT-DETR 文档](https://docs.ultralytics.com/models/rtdetr/)
- [原始 RT-DETR 项目](https://github.com/lyuwenyu/RT-DETR)
- [D-Fire 数据集](https://github.com/gaia-solutions-on-demand/DFireDataset)
- 项目代码使用 AGPL-3.0，见 `LICENSE`；第三方模型、框架、数据仍遵循各自许可。数据和模型不因本仓库代码许可而自动获得再分发授权。
