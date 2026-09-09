#!/usr/bin/env python3
"""Summarize a saved Hydra graph, mesh, and trajectory without a GUI."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import statistics


LABEL_NAMES = {
    0: "unknown",
    1: "ground",
    2: "tree_trunk",
    3: "foliage",
    4: "rock",
    5: "building",
    6: "robot",
    7: "other_object",
}


def quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return round(ordered[index], 4)


def inspect_graph(path: Path) -> dict:
    graph = json.loads(path.read_text(encoding="utf-8"))
    node_types = Counter()
    labels = Counter()
    for node in graph.get("nodes", []):
        attributes = node.get("attributes", {})
        node_types[str(attributes.get("type", "unknown"))] += 1
        if "semantic_label" in attributes:
            label = int(attributes["semantic_label"])
            labels[LABEL_NAMES.get(label, str(label))] += 1
    return {
        "nodes": len(graph.get("nodes", [])),
        "edges": len(graph.get("edges", [])),
        "node_types": dict(sorted(node_types.items())),
        "semantic_nodes": dict(sorted(labels.items())),
    }


def inspect_mesh(path: Path) -> dict:
    with path.open("r", encoding="ascii") as stream:
        properties: list[str] = []
        vertex_count = 0
        in_vertex = False
        for raw in stream:
            line = raw.strip()
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
                in_vertex = True
            elif line.startswith("element "):
                in_vertex = False
            elif in_vertex and line.startswith("property "):
                properties.append(line.split()[-1])
            elif line == "end_header":
                break
        required = {"x", "y", "z", "label"}
        if not required.issubset(properties):
            raise ValueError(f"mesh is missing required properties: {required - set(properties)}")
        indices = {name: properties.index(name) for name in required}
        bounds = {axis: [float("inf"), float("-inf")] for axis in "xyz"}
        labels = Counter()
        label_z: dict[int, list[float]] = defaultdict(list)
        for _ in range(vertex_count):
            values = stream.readline().split()
            if len(values) < len(properties):
                raise ValueError("mesh ended before all declared vertices were read")
            coordinates = {axis: float(values[indices[axis]]) for axis in "xyz"}
            label = int(values[indices["label"]])
            labels[label] += 1
            label_z[label].append(coordinates["z"])
            for axis in "xyz":
                bounds[axis][0] = min(bounds[axis][0], coordinates[axis])
                bounds[axis][1] = max(bounds[axis][1], coordinates[axis])
    return {
        "vertices": vertex_count,
        "bounds_m": {axis: [round(value, 4) for value in pair] for axis, pair in bounds.items()},
        "semantic_vertices": {
            LABEL_NAMES.get(label, str(label)): count for label, count in sorted(labels.items())
        },
        "z_quantiles_m": {
            LABEL_NAMES.get(label, str(label)): {
                "min": quantile(values, 0.0),
                "median": quantile(values, 0.5),
                "p99": quantile(values, 0.99),
                "max": quantile(values, 1.0),
            }
            for label, values in sorted(label_z.items())
        },
    }


def inspect_trajectory(path: Path) -> dict:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return {"poses": 0}
    coordinates = {axis: [float(row[axis]) for row in rows] for axis in "xyz"}
    return {
        "poses": len(rows),
        "duration_s": round((int(rows[-1]["timestamp[ns]"]) - int(rows[0]["timestamp[ns]"])) / 1e9, 3),
        "bounds_m": {
            axis: [round(min(values), 4), round(max(values), 4)]
            for axis, values in coordinates.items()
        },
        "median_altitude_m": round(statistics.median(coordinates["z"]), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("hydra_dir", type=Path, help="runs/<id>/<robot>/hydra directory")
    args = parser.parse_args()
    backend = args.hydra_dir / "backend"
    required = [backend / "dsg.json", backend / "mesh.ply", backend / "trajectory.csv"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("missing finalized artifacts: " + ", ".join(missing))
    summary = {
        "hydra_dir": str(args.hydra_dir.resolve()),
        "graph": inspect_graph(required[0]),
        "mesh": inspect_mesh(required[1]),
        "trajectory": inspect_trajectory(required[2]),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
