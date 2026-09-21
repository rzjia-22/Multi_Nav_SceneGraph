"""NoMaD 离线零样本结果聚合与中文报告生成。"""

from __future__ import annotations

from collections import defaultdict
import csv
import io
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.inprogress")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def mean(items: list[dict], key: str) -> float | None:
    values = [float(item[key]) for item in items if item.get(key) is not None]
    return float(np.mean(values)) if values else None


def summarize_episodes(items: list[dict]) -> dict[str, Any]:
    if not items:
        return {"episode_count": 0}
    levels = tuple(items[0]["route_viability_proxy"].keys())
    return {
        "episode_count": len(items),
        "route_viability_proxy": {
            level: {
                "success_count": sum(
                    bool(item["route_viability_proxy"][level]) for item in items
                ),
                "success_rate": sum(
                    bool(item["route_viability_proxy"][level]) for item in items
                ) / len(items),
            }
            for level in levels
        },
        "mean_collision_free_fraction": mean(items, "collision_free_fraction"),
        "mean_conservative_safe_fraction": mean(items, "conservative_safe_fraction"),
        "mean_goal_progress_fraction": mean(items, "goal_progress_fraction"),
        "mean_corridor_adherence_fraction": mean(items, "corridor_adherence_fraction"),
        "mean_accepted_action_fraction": mean(items, "accepted_action_fraction"),
        "mean_any_sample_accepted_fraction": mean(items, "any_sample_accepted_fraction"),
        "mean_sample_safe_probability": mean(items, "sample_safe_probability"),
        "mean_ADE_m": mean(items, "ADE_m"),
        "mean_FDE_m": mean(items, "FDE_m"),
        "mean_best_of_n_ADE_m": mean(items, "best_of_n_ADE_m"),
        "mean_best_of_n_FDE_m": mean(items, "best_of_n_FDE_m"),
        "mean_endpoint_diversity_m": mean(items, "endpoint_diversity_m"),
        "mean_distance_MAE_steps": mean(items, "distance_MAE_steps"),
        "mean_inference_ms": mean(items, "inference_ms_mean"),
        "mean_full_anchor_wall_ms": mean(items, "full_anchor_wall_ms_mean"),
    }


def group_summary(
    episodes: list[dict], key_function: Callable[[dict], str]
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for episode in episodes:
        groups[str(key_function(episode))].append(episode)
    return {key: summarize_episodes(values) for key, values in sorted(groups.items())}


def build_report(metadata: dict, episodes: list[dict]) -> dict[str, Any]:
    return {
        "report_version": 1,
        "evaluation_id": "nomad_zero_shot_offline_dataset_v0",
        "interpretation": {
            "closed_loop_success_rate": None,
            "offline_route_viability_is_proxy": True,
            "expert_future_images_used_as_local_visual_goals": True,
            "ground_truth_used_to_select_primary_diffusion_sample": False,
            "primary_policy_sample": "deterministic_sample_0",
            "best_of_n_metrics_are_privileged_upper_bounds": True,
        },
        **metadata,
        "episode_count": len(episodes),
        "summary": summarize_episodes(episodes),
        "by_split": group_summary(episodes, lambda item: item["split"]),
        "by_scene": group_summary(episodes, lambda item: item["scene_id"]),
        "by_route_bucket": group_summary(episodes, lambda item: item["route_bucket"]),
        "by_terrain": group_summary(episodes, lambda item: item["environment"]["terrain"]),
        "by_tree_density": group_summary(
            episodes, lambda item: item["environment"]["tree_density"]
        ),
        "by_lighting": group_summary(episodes, lambda item: item["environment"]["lighting"]),
        "episodes": episodes,
    }


def episode_csv(episodes: list[dict]) -> str:
    fields = [
        "episode_id", "split", "scene_id", "route_bucket", "terrain", "tree_density",
        "lighting", "anchor_count", "collision_free_fraction",
        "conservative_safe_fraction", "goal_progress_fraction",
        "corridor_adherence_fraction", "accepted_action_fraction",
        "any_sample_accepted_fraction", "sample_safe_probability", "ADE_m", "FDE_m",
        "minimum_clearance_m", "maximum_consecutive_unsafe", "nominal_proxy_success",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for item in episodes:
        writer.writerow({
            "episode_id": item["episode_id"],
            "split": item["split"],
            "scene_id": item["scene_id"],
            "route_bucket": item["route_bucket"],
            "terrain": item["environment"]["terrain"],
            "tree_density": item["environment"]["tree_density"],
            "lighting": item["environment"]["lighting"],
            "anchor_count": item["anchor_count"],
            "collision_free_fraction": item["collision_free_fraction"],
            "conservative_safe_fraction": item["conservative_safe_fraction"],
            "goal_progress_fraction": item["goal_progress_fraction"],
            "corridor_adherence_fraction": item["corridor_adherence_fraction"],
            "accepted_action_fraction": item["accepted_action_fraction"],
            "any_sample_accepted_fraction": item["any_sample_accepted_fraction"],
            "sample_safe_probability": item["sample_safe_probability"],
            "ADE_m": item["ADE_m"],
            "FDE_m": item["FDE_m"],
            "minimum_clearance_m": item["minimum_clearance_m"],
            "maximum_consecutive_unsafe": item["maximum_consecutive_unsafe"],
            "nominal_proxy_success": item["route_viability_proxy"]["nominal"],
        })
    return buffer.getvalue()


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


def _metres(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f} m"


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    viability = summary["route_viability_proxy"]
    lines = [
        "# NoMaD 在 Dataset V0 上的离线 Zero-shot 报告",
        "",
        f"- 状态：**{report['status']}**",
        f"- 轨迹任务：**{report['episode_count']}/70**",
        f"- 有效离线决策锚点：**{report['anchor_count']}**",
        f"- 官方 NoMaD checkpoint SHA256：`{report['nomad']['checkpoint_sha256']}`",
        f"- 官方源码 revision：`{report['nomad']['source_revision']}`",
        "",
        "## 首要结论",
        "",
        "本报告不能给出真实闭环任务成功率：日志中的后续 RGB 来自专家，而不是",
        "NoMaD 动作造成的新状态。以下“路线可行性成功率”是预先冻结阈值下的离线代理，",
        "用于筛选模型是否值得进入 Isaac 闭环，不能替代闭环成功/碰撞统计。",
        "",
        "| 代理口径 | 成功任务 | 成功率 |",
        "|---|---:|---:|",
    ]
    for level in ("lenient", "nominal", "strict"):
        item = viability[level]
        lines.append(f"| {level} | {item['success_count']}/{report['episode_count']} | {_percent(item['success_rate'])} |")
    lines.extend([
        "",
        "主策略口径固定使用每个锚点的第 1 条确定性扩散样本，不使用真值选样。",
        f"其预测折线无碰撞比例为 **{_percent(summary['mean_collision_free_fraction'])}**，",
        f"保守净空比例为 **{_percent(summary['mean_conservative_safe_fraction'])}**，",
        f"朝局部视觉目标取得有效进展的比例为 **{_percent(summary['mean_goal_progress_fraction'])}**，",
        f"保持在 0.75 m 专家走廊内的比例为 **{_percent(summary['mean_corridor_adherence_fraction'])}**。",
        "",
        f"综合局部动作接受率为 **{_percent(summary['mean_accepted_action_fraction'])}**。",
        f"允许在 8 条样本中事后选择时，上界提高到 **{_percent(summary['mean_any_sample_accepted_fraction'])}**；",
        "该数值使用了特权评估信息，仅表示多模态分布容量。",
        "",
        "## 辅助误差与效率",
        "",
        f"- ADE：{_metres(summary['mean_ADE_m'])}",
        f"- FDE：{_metres(summary['mean_FDE_m'])}",
        f"- 特权 best-of-8 ADE / FDE：{_metres(summary['mean_best_of_n_ADE_m'])} / {_metres(summary['mean_best_of_n_FDE_m'])}",
        f"- 8 样本末端多样性：{_metres(summary['mean_endpoint_diversity_m'])}",
        f"- 距离头 MAE：{summary['mean_distance_MAE_steps']:.3f} 个 4 Hz 步长",
        f"- 单模式 8 样本平均扩散解码时间：{summary['mean_inference_ms']:.1f} ms",
        f"- 单锚点三模式平均完整墙钟时间：{summary['mean_full_anchor_wall_ms']:.1f} ms",
        f"- 8 样本平均安全概率：{_percent(summary['mean_sample_safe_probability'])}",
        "",
        "ADE/FDE 只作为辅助：森林中不同于专家的轨迹仍可能安全；反过来，低误差也不保证闭环稳定。",
        "",
        "## 输入模式敏感性",
        "",
        "| 模式 | 无碰撞比例 | 有效进展比例 | 专家走廊比例 | 综合接受率 | ADE | FDE |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| 未来 2 秒局部视觉目标（主口径） | {_percent(summary['mean_collision_free_fraction'])} | "
        f"{_percent(summary['mean_goal_progress_fraction'])} | {_percent(summary['mean_corridor_adherence_fraction'])} | "
        f"{_percent(summary['mean_accepted_action_fraction'])} | {_metres(summary['mean_ADE_m'])} | {_metres(summary['mean_FDE_m'])} |",
    ])
    for mode, label in (
        ("final_visual_goal", "轨迹终点视觉目标"),
        ("goal_masked_exploration", "目标遮蔽探索"),
    ):
        item = report["secondary_mode_summary"][mode]
        lines.append(
            f"| {label} | {_percent(item['mean_collision_free_fraction'])} | "
            f"{_percent(item['mean_goal_progress_fraction'])} | "
            f"{_percent(item['mean_corridor_adherence_fraction'])} | "
            f"{_percent(item['mean_accepted_action_fraction'])} | "
            f"{_metres(item['mean_ADE_m'])} | {_metres(item['mean_FDE_m'])} |"
        )
    lines.extend([
        "",
        "## 分组结果（标称路线可行性代理）",
        "",
        "| 分组 | 任务数 | 标称成功率 | 无碰撞比例 | 综合接受率 |",
        "|---|---:|---:|---:|---:|",
    ])
    for title, groups in (
        ("split", report["by_split"]),
        ("terrain", report["by_terrain"]),
        ("tree_density", report["by_tree_density"]),
        ("route", report["by_route_bucket"]),
    ):
        for key, value in groups.items():
            lines.append(
                f"| {title}:{key} | {value['episode_count']} | "
                f"{_percent(value['route_viability_proxy']['nominal']['success_rate'])} | "
                f"{_percent(value['mean_collision_free_fraction'])} | "
                f"{_percent(value['mean_accepted_action_fraction'])} |"
            )
    lines.extend([
        "",
        "## 评测口径",
        "",
        "- 4 帧 RGB 上下文，按 4 Hz 从 10 Hz 日志重采样；4:3 中心裁剪后缩放到 96×96。",
        "- 主模式使用未来 2 秒的专家 RGB 作为局部视觉子目标；深度不输入 NoMaD。",
        "- 每个锚点固定生成 8 条样本；样本 0 是部署口径，其余样本只用于分布统计。",
        "- 动作按 DIABLO 最大前进速度 0.65 m/s 和 4 Hz 模型频率换算为米。",
        "- 碰撞净空使用连续预测折线、树干代理半径、0.42 m 机器人半径和场景边界计算。",
        "- 综合动作接受条件：不碰撞、至少前进 0.05 m、方向误差不超过 45°、",
        "  且完整预测保持在 0.75 m 专家走廊内。",
        "",
        "## 限制",
        "",
        "1. 未来 RGB 子目标和专家轨迹只用于离线诊断，属于特权信息。",
        "2. 专家日志无法暴露模型动作导致的视觉分布偏移、卡死或恢复行为。",
        "3. Dataset V0 的 50 条 train 轨迹对 NoMaD 是零样本数据，但对本项目 V0 不是 held-out；",
        "   因此与 V0 的正式头对头比较应限制在 validation+test 20 条。",
        "4. 路线可行性阈值没有在结果上回调；宽松/标称/严格三档用于展示敏感性。",
        "",
        "机器可读明细见 `results.json`，逐任务表见 `episode_metrics.csv`。",
        "",
    ])
    return "\n".join(lines)
