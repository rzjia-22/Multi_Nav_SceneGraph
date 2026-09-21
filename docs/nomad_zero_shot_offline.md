# NoMaD Zero-shot 离线基线

## 目的与结论边界

这条链路在 Dataset V0 全部 70 条专家轨迹上运行官方 NoMaD，不训练、
不微调，也不使用 Dataset V0 估计模型参数。它回答的是“官方 NoMaD 的局部轨迹
分布在当前森林日志上是否具有安全性、进展性和专家走廊一致性”，不是闭环任务成功率。

离线日志的后续图像由专家动作产生。NoMaD 的动作没有改变后续观测，因此离线评测
无法暴露视觉分布偏移、卡死、恢复失败和控制误差。正式闭环成功率仍须在 Isaac 中
复现 NoMaD 策略、控制与安全边界后测量。

## 冻结的输入适配

- 使用官方 NoMaD 源码 revision `dca79815b704e5aa9c6bdc3082351f9e3b2848c2`
  和公开 `nomad.pth`，加载时严格核对源码 revision、文件大小、SHA256、参数量和
  全部权重键。
- Dataset V0 的 10 Hz RGB 按最近时间戳重采样到 NoMaD 的 4 Hz；输入 4 帧上下文。
- 原始 640×360 图像先做 4:3 中心裁剪，再缩放到 96×96，并使用官方 ImageNet
  均值和标准差。深度不输入模型。
- 主口径把锚点后 2 秒的专家 RGB 作为局部视觉子目标，使目标距离与 NoMaD 的
  8 点、4 Hz（2 秒）预测时域一致。
- 同时记录“整条专家轨迹终点图像”和“目标遮蔽探索”两个敏感性模式；它们不替代
  主口径。
- 按官方部署做法将归一化动作乘以 `最大前进速度 / 模型频率`。当前 DIABLO 配置为
  `0.65 / 4 = 0.1625 m`。
- 每个锚点固定生成 8 条扩散样本。样本 0 是不看真值的部署口径；best-of-8 和
  any-of-8 只能作为特权上界。

所有阈值和适配参数集中在
[`config/navigation/nomad_zero_shot_offline.yaml`](../config/navigation/nomad_zero_shot_offline.yaml)，
应在查看正式结果前冻结。原始第三方源码和 checkpoint 保持在 Git 忽略的
`models/NoMaD`，不复制进本仓库。

## 指标

ADE/FDE 作为轨迹模仿误差保留，但不是首要判断。正式报告同时计算：

- 连续折线最小净空、无碰撞锚点比例和保守净空比例；
- 对局部视觉目标的有效进展、方向误差；
- 将预测折线加密到至多 0.05 m 间隔后，计算其相对专家未来连续折线的横向偏差
  与 0.75 m 走廊保持率；
- 同时满足安全、进展、方向和走廊约束的综合动作接受率；
- 最长连续不安全锚点数；
- 8 样本安全概率、末端多样性、any-of-8 与 best-of-8 特权上界；
- 距离头误差与 GPU 推理时间。

逐任务“路线可行性”采用配置中预先冻结的宽松、标称、严格三档阈值。它是用于决定
是否值得进入闭环测试的离线代理，报告中不会将其冒充为真实成功率。

## 复现

```bash
make nomad-assets
make nomad-zero-shot-smoke
make nomad-zero-shot-eval
```

`nomad-assets` 获取并校验官方源码与权重；smoke 只跑 1 条轨迹的 2 个锚点；正式命令
遍历 70 条轨迹并写入 `artifacts/baselines/nomad_zero_shot_offline/`。带逐锚点数据的
运行目录位于 Git 忽略的 `runs/nomad_zero_shot_offline/`，以免污染版本库。

## 与 NavDiffusion V0 的比较规则

NoMaD 对全部 70 条轨迹都没有训练，因此全量结果是 zero-shot 诊断。NavDiffusion V0
使用了其中 50 条 train 轨迹；两者正式、同数据治理口径的比较只能使用 validation +
test 共 20 条。Dataset V0 test 已在 V0 阶段打开，任何根据这些结果设计的新模型都需要
另建未触碰的最终测试场景。

## 正式结果

正式运行在 70/70 条任务上得到 1,197 个有效决策锚点。预先冻结的宽松、标称、严格
路线可行性代理分别通过 37/70、20/70、11/70；validation+test 20 条切片分别为
7/20、3/20、1/20。主样本逐任务等权平均的无碰撞比例为 88.3%，保守净空比例
76.6%，有效进展比例 91.1%，专家走廊保持率 90.9%，综合动作接受率 76.0%。

8 条样本中只要有一条满足综合条件的特权上界为 90.9%，明显高于固定样本 0 的
76.0%。这表明 NoMaD 的多模态分布中经常存在更合适的候选，但原模型没有针对本森林
几何的在线选样器。未来 2 秒局部视觉目标也优于终点视觉目标和目标遮蔽探索，后两者
的综合接受率分别只有 59.5% 和 66.6%。

当前状态为 `OFFLINE_ONLY`：原始 NoMaD 不应直接替换 V0 或进入实机。若继续推进，
优先顺序应是先在 Isaac 中接入闭环安全筛选/候选打分，再决定是否做输入适配微调或
网络层重设计。完整报告见
[NoMaD 离线基线正式报告](../artifacts/baselines/nomad_zero_shot_offline/summary.md)。

## 上游依据

- 论文：[NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration](https://arxiv.org/abs/2310.07896)
- 官方代码与部署示例：[robodhruv/visualnav-transformer](https://github.com/robodhruv/visualnav-transformer)
