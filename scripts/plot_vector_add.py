"""从 Vector Add CSV 生成可提交到 GitHub 的 SVG 性能曲线。"""

import csv
import math
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = (
    PROJECT_ROOT
    / "benchmarks"
    / "results"
    / "vector_add_block_size_comparison_rtx3080ti_fp32.csv"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "assets" / "benchmark" / "vector_add_block_size_comparison.svg"
)

WIDTH = 1200
HEIGHT = 560
COLORS = {128: "#2563EB", 256: "#EA580C", 512: "#16A34A"}


def load_rows() -> dict[int, list[dict[str, float]]]:
    """读取并按 Block Size 分组，保留数值类型。"""
    grouped: dict[int, list[dict[str, float]]] = defaultdict(list)
    with INPUT_PATH.open(encoding="utf-8", newline="") as file:
        for raw in csv.DictReader(file):
            block_size = int(raw["block_size"])
            grouped[block_size].append(
                {
                    "N": int(raw["N"]),
                    "p50_us": float(raw["p50_us"]),
                    "p20_us": float(raw["p20_us"]),
                    "p80_us": float(raw["p80_us"]),
                    "bandwidth": float(raw["effective_bandwidth_GBps"]),
                }
            )
    for rows in grouped.values():
        rows.sort(key=lambda row: row["N"])
    return dict(sorted(grouped.items()))


def esc(text: object) -> str:
    """转义 SVG 文本。"""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def line(x1: float, y1: float, x2: float, y2: float, **attrs: object) -> str:
    properties = " ".join(f'{key.replace("_", "-")}="{esc(value)}"' for key, value in attrs.items())
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" {properties}/>'


def text(x: float, y: float, value: object, **attrs: object) -> str:
    properties = " ".join(f'{key.replace("_", "-")}="{esc(item)}"' for key, item in attrs.items())
    return f'<text x="{x:.2f}" y="{y:.2f}" {properties}>{esc(value)}</text>'


def polyline(points: list[tuple[float, float]], **attrs: object) -> str:
    point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    properties = " ".join(f'{key.replace("_", "-")}="{esc(value)}"' for key, value in attrs.items())
    return f'<polyline points="{point_text}" {properties}/>'


def polygon(points: list[tuple[float, float]], **attrs: object) -> str:
    point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    properties = " ".join(f'{key.replace("_", "-")}="{esc(value)}"' for key, value in attrs.items())
    return f'<polygon points="{point_text}" {properties}/>'


def draw_panel(
    svg: list[str],
    grouped: dict[int, list[dict[str, float]]],
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    title_value: str,
    y_label: str,
    y_ticks: list[float],
    value_key: str,
    log_y: bool,
    show_band: bool,
) -> None:
    """绘制一个共享对数 X 轴的折线图面板。"""
    all_rows = [row for rows in grouped.values() for row in rows]
    x_min = min(math.log2(row["N"]) for row in all_rows)
    x_max = max(math.log2(row["N"]) for row in all_rows)
    transform_y = math.log10 if log_y else lambda value: value
    y_min = transform_y(min(y_ticks))
    y_max = transform_y(max(y_ticks))

    def sx(n_value: float) -> float:
        return left + (math.log2(n_value) - x_min) / (x_max - x_min) * width

    def sy(y_value: float) -> float:
        normalized = (transform_y(y_value) - y_min) / (y_max - y_min)
        return top + height - normalized * height

    svg.append(text(left, top - 24, title_value, font_size=18, font_weight="600", fill="#111827"))
    for tick in y_ticks:
        y = sy(tick)
        svg.append(line(left, y, left + width, y, stroke="#E5E7EB", stroke_width=1))
        svg.append(text(left - 10, y + 4, f"{tick:g}", text_anchor="end", font_size=12, fill="#4B5563"))

    x_ticks = sorted({int(math.log2(row["N"])) for row in all_rows})
    for power in x_ticks:
        x = sx(2**power)
        svg.append(line(x, top, x, top + height, stroke="#F3F4F6", stroke_width=1))
        svg.append(text(x, top + height + 22, f"2^{power}", text_anchor="middle", font_size=11, fill="#4B5563"))

    svg.append(line(left, top + height, left + width, top + height, stroke="#6B7280", stroke_width=1.2))
    svg.append(line(left, top, left, top + height, stroke="#6B7280", stroke_width=1.2))
    svg.append(text(left + width / 2, top + height + 48, "元素数量 N（对数刻度）", text_anchor="middle", font_size=13, fill="#374151"))
    svg.append(text(left - 54, top + height / 2, y_label, text_anchor="middle", font_size=13, fill="#374151", transform=f"rotate(-90 {left - 54:.2f} {top + height / 2:.2f})"))

    for block_size, rows in grouped.items():
        color = COLORS[block_size]
        if show_band:
            upper = [(sx(row["N"]), sy(row["p80_us"])) for row in rows]
            lower = [(sx(row["N"]), sy(row["p20_us"])) for row in reversed(rows)]
            svg.append(polygon(upper + lower, fill=color, fill_opacity=0.10, stroke="none"))
        points = [(sx(row["N"]), sy(row[value_key])) for row in rows]
        svg.append(polyline(points, fill="none", stroke=color, stroke_width=2.5, stroke_linejoin="round"))
        for x, y in points:
            svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.2" fill="{color}"/>')


def main() -> None:
    grouped = load_rows()
    expected = {128, 256, 512}
    if set(grouped) != expected:
        raise ValueError(f"期望 Block Size {sorted(expected)}，实际得到 {sorted(grouped)}")
    if any(len(rows) != 9 for rows in grouped.values()):
        raise ValueError("每个 Block Size 必须包含 9 个相同尺寸的测量点")

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<g font-family="Microsoft YaHei, Noto Sans CJK SC, Arial, sans-serif">',
        text(60, 38, "Vector Add：不同 Triton Block Size 的性能比较", font_size=22, font_weight="700", fill="#111827"),
        text(60, 62, "RTX 3080 Ti · FP32 · 阴影表示 P20–P80 延迟范围", font_size=13, fill="#6B7280"),
    ]

    legend_x = 770
    for index, block_size in enumerate(grouped):
        x = legend_x + index * 125
        svg.append(line(x, 54, x + 24, 54, stroke=COLORS[block_size], stroke_width=3))
        svg.append(text(x + 31, 59, f"Block {block_size}", font_size=12, fill="#374151"))

    draw_panel(
        svg,
        grouped,
        left=80,
        top=105,
        width=455,
        height=350,
        title_value="P50 延迟",
        y_label="延迟（μs，对数刻度）",
        y_ticks=[4, 16, 64, 256, 1024],
        value_key="p50_us",
        log_y=True,
        show_band=True,
    )
    draw_panel(
        svg,
        grouped,
        left=680,
        top=105,
        width=455,
        height=350,
        title_value="有效显存带宽",
        y_label="有效带宽（GB/s）",
        y_ticks=[0, 200, 400, 600, 800, 900],
        value_key="bandwidth",
        log_y=False,
        show_band=False,
    )
    svg.extend(["</g>", "</svg>"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(f"已生成：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()

