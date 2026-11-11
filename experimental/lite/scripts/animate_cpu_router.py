#!/usr/bin/env python3
"""Render the 8xL4 CPU-router collective packet flows as animated GIFs."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets"
PREVIEW = ROOT / "build/figures"
WIDTH, HEIGHT = 1200, 760
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
INK = "#17324D"
MUTED = "#607486"
BLUE = "#0068B5"
TEAL = "#148D91"
ORANGE = "#D65A00"
PAPER = "#F7FAFC"
WHITE = "#FFFFFF"
PALE_BLUE = "#E7F2FA"
PALE_TEAL = "#E6F5F2"
PALE_ORANGE = "#FFF1E8"
GRAY = "#D7E1E9"
GROUPS = ((62, 568), (632, 1138))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(BOLD if bold else FONT, size)


def text_center(draw: ImageDraw.ImageDraw, x: int, y: int, value: str,
                size: int = 20, color: str = INK, bold: bool = False) -> None:
    f = font(size, bold)
    box = draw.textbbox((0, 0), value, font=f)
    draw.text((x - (box[2] - box[0]) / 2, y - (box[3] - box[1]) / 2 - box[1]),
              value, font=f, fill=color)


def card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int],
         label: str, *, active: bool = False, accent: str = BLUE,
         fill: str = WHITE, size: int = 20) -> None:
    draw.rounded_rectangle(box, radius=14, fill=fill,
                           outline=accent if active else GRAY,
                           width=4 if active else 2)
    x0, y0, x1, y1 = box
    lines = label.split("\n")
    spacing = 26 if size >= 19 else 23
    top = (y0 + y1) / 2 - spacing * (len(lines) - 1) / 2
    for i, line in enumerate(lines):
        text_center(draw, (x0 + x1) // 2, round(top + i * spacing), line,
                    size=size, color=accent if active else INK,
                    bold=active or i == 0)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int],
          end: tuple[int, int], color: str, progress: float) -> None:
    draw.line((start, end), fill=color, width=6)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    tip = end
    wing = 14
    points = [tip,
              (tip[0] - 18 * math.cos(angle) + wing * math.sin(angle),
               tip[1] - 18 * math.sin(angle) - wing * math.cos(angle)),
              (tip[0] - 18 * math.cos(angle) - wing * math.sin(angle),
               tip[1] - 18 * math.sin(angle) + wing * math.cos(angle))]
    draw.polygon(points, fill=color)
    x = start[0] + (end[0] - start[0]) * progress
    y = start[1] + (end[1] - start[1]) * progress
    draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=WHITE,
                 outline=color, width=4)


def routed_arrow(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]],
                 color: str, progress: float) -> None:
    draw.line(points, fill=color, width=6, joint="curve")
    start, end = points[-2:]
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    draw.polygon([end,
                  (end[0] - 18 * math.cos(angle) + 14 * math.sin(angle),
                   end[1] - 18 * math.sin(angle) - 14 * math.cos(angle)),
                  (end[0] - 18 * math.cos(angle) - 14 * math.sin(angle),
                   end[1] - 18 * math.sin(angle) + 14 * math.cos(angle))],
                 fill=color)
    lengths = [math.dist(a, b) for a, b in zip(points, points[1:])]
    remaining = progress * sum(lengths)
    for (a, b), length in zip(zip(points, points[1:]), lengths):
        if remaining <= length:
            x = a[0] + (b[0] - a[0]) * remaining / length
            y = a[1] + (b[1] - a[1]) * remaining / length
            draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=WHITE,
                         outline=color, width=4)
            break
        remaining -= length


def frame_base(title: str, subtitle: str, step: int,
               steps: list[str]) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), PAPER)
    draw = ImageDraw.Draw(image)
    text_center(draw, WIDTH // 2, 38, title, 30, INK, True)
    text_center(draw, WIDTH // 2, 78, subtitle, 17, MUTED)
    for i, label in enumerate(steps):
        x = 70 + i * 214
        draw.rounded_rectangle((x, 109, x + 202, 152), radius=13,
                               fill=BLUE if i == step else WHITE,
                               outline=BLUE if i == step else GRAY, width=2)
        text_center(draw, x + 101, 130, f"{i + 1}. {label}", 16,
                    WHITE if i == step else MUTED, i == step)
    return image, draw


def gpu_rows(draw: ImageDraw.ImageDraw, active: set[int] = frozenset()) -> None:
    for node, (left, right) in enumerate(GROUPS):
        text_center(draw, (left + right) // 2, 184,
                    f"NUMA {node} · GPUs {node * 4}–{node * 4 + 3}",
                    19, INK, True)
        for offset in range(4):
            rank = node * 4 + offset
            x = left + 22 + offset * 120
            card(draw, (x, 207, x + 110, 269), f"GPU {rank}",
                 active=rank in active, fill=PALE_BLUE, size=17)


def allreduce(step: int, progress: float) -> Image.Image:
    steps = ["Upload", "Reduce 4", "Merge 2", "Publish", "Read result"]
    image, d = frame_base("AllReduce · 8xL4 CPU router",
                          "One packet per GPU; each NUMA group holds four inputs.",
                          step, steps)
    gpu_rows(d, set(range(8)) if step in (0, 4) else set())
    for node, (left, right) in enumerate(GROUPS):
        mid = (left + right) // 2
        card(d, (left + 27, 316, right - 27, 388),
             f"Host packets · GPU {node * 4}–{node * 4 + 3}",
             active=step == 0, accent=BLUE, fill=PALE_BLUE)
        card(d, (left + 27, 448, right - 27, 516),
             f"CPU: reduce 4 → partial P{node}",
             active=step == 1, accent=TEAL, fill=PALE_TEAL)
        card(d, (left + 27, 577, right - 27, 645),
             "P0 + P1 → shared result", active=step in (2, 3),
             accent=TEAL, fill=PALE_TEAL)
        if step == 0:
            arrow(d, (mid, 273), (mid, 309), BLUE, progress)
        elif step == 1:
            arrow(d, (mid, 392), (mid, 441), TEAL, progress)
        elif step == 2:
            arrow(d, (mid, 521), (mid, 569), TEAL, progress)
        elif step == 4:
            arrow(d, (mid, 649), (mid, 687), BLUE, progress)
        card(d, (left + 27, 692, right - 27, 744),
             f"Result on GPUs {node * 4}–{node * 4 + 3}",
             active=step == 4, accent=BLUE, fill=PALE_BLUE, size=18)
    if step == 2:
        arrow(d, (557, 538), (641, 538), ORANGE, progress)
        arrow(d, (643, 554), (559, 554), ORANGE, progress)
    if step == 3:
        text_center(d, WIDTH // 2, 664,
                    "CPU publishes one immutable result per NUMA group",
                    18, TEAL, True)
    return image


def allgather(step: int, progress: float) -> Image.Image:
    steps = ["Upload once", "Publish ready", "Keep own", "Read seven", "Assemble"]
    image, d = frame_base("AllGather · 8xL4 CPU router",
                          "GPU 0 is the example destination; every GPU does the same.",
                          step, steps)
    gpu_rows(d, {0} if step in (2, 3, 4) else set(range(8)) if step == 0 else set())
    for node, (left, right) in enumerate(GROUPS):
        mid = (left + right) // 2
        card(d, (left + 27, 315, right - 27, 388),
             f"Host shards: {node * 4} | {node * 4 + 1} | "
             f"{node * 4 + 2} | {node * 4 + 3}",
             active=step in (0, 3), accent=BLUE, fill=PALE_BLUE, size=19)
        card(d, (left + 27, 449, right - 27, 513),
             "CPU publishes ready flag", active=step == 1,
             accent=TEAL, fill=PALE_TEAL, size=19)
        if step == 0:
            arrow(d, (mid, 273), (mid, 309), BLUE, progress)
        elif step == 1:
            arrow(d, (mid, 393), (mid, 442), TEAL, progress)
    if step == 2:
        routed_arrow(d, [(140, 273), (60, 286), (60, 614), (222, 626)],
                     ORANGE, progress)
    elif step == 3:
        routed_arrow(d, [(318, 394), (589, 417), (589, 612), (548, 626)],
                     BLUE, progress)
        routed_arrow(d, [(885, 394), (1144, 417), (1144, 609), (648, 626)],
                     BLUE, progress)
    colors = [ORANGE if i == 0 else BLUE for i in range(8)]
    for rank in range(8):
        x = 176 + rank * 106
        active = (rank == 0 and step >= 2) or (rank > 0 and step >= 4)
        card(d, (x, 634, x + 96, 701), f"shard {rank}",
             active=active, accent=colors[rank],
             fill=PALE_ORANGE if rank == 0 else PALE_BLUE, size=16)
    text_center(d, WIDTH // 2, 729, "GPU 0 output · rank order", 18, INK, True)
    return image


def reducescatter(step: int, progress: float) -> Image.Image:
    steps = ["Keep own", "Upload seven", "Reduce groups", "Merge remote", "Add own"]
    image, d = frame_base("ReduceScatter · 8xL4 CPU router",
                          "Output shard 0 on GPU 0; the other shards run in parallel.",
                          step, steps)
    gpu_rows(d, {0} if step in (0, 4) else set(range(1, 8)) if step == 1 else set())
    card(d, (89, 316, 541, 390), "Host: shard 0 from GPUs 1–3",
         active=step == 1, accent=BLUE, fill=PALE_BLUE, size=19)
    card(d, (659, 316, 1111, 390), "Host: shard 0 from GPUs 4–7",
         active=step == 1, accent=BLUE, fill=PALE_BLUE, size=19)
    card(d, (89, 452, 541, 520), "CPU partial: 1 + 2 + 3",
         active=step == 2, accent=TEAL, fill=PALE_TEAL, size=19)
    card(d, (659, 452, 1111, 520), "CPU partial: 4 + 5 + 6 + 7",
         active=step == 2, accent=TEAL, fill=PALE_TEAL, size=19)
    card(d, (300, 584, 900, 650), "CPU: merge partials → remote sum",
         active=step == 3, accent=TEAL, fill=PALE_TEAL, size=19)
    card(d, (300, 692, 900, 744), "GPU 0: remote sum + retained shard 0",
         active=step == 4, accent=ORANGE, fill=PALE_ORANGE, size=18)
    if step == 0:
        text_center(d, WIDTH // 2, 553,
                    "GPU 0 keeps its own input shard in device memory",
                    18, ORANGE, True)
    elif step == 1:
        arrow(d, (315, 273), (315, 309), BLUE, progress)
        arrow(d, (885, 273), (885, 309), BLUE, progress)
    elif step == 2:
        arrow(d, (315, 395), (315, 445), TEAL, progress)
        arrow(d, (885, 395), (885, 445), TEAL, progress)
    elif step == 3:
        arrow(d, (350, 525), (512, 578), TEAL, progress)
        arrow(d, (850, 525), (688, 578), TEAL, progress)
    elif step == 4:
        arrow(d, (600, 654), (600, 685), TEAL, progress)
        routed_arrow(d, [(140, 273), (60, 287), (60, 674), (425, 686)],
                     ORANGE, progress)
    return image


def render(name: str, draw_scene) -> None:
    frames = []
    durations = []
    for step in range(5):
        for phase in (0.18, 0.55, 0.90):
            rgb = draw_scene(step, phase)
            frames.append(rgb.quantize(colors=96, method=Image.Quantize.MEDIANCUT))
            durations.append(330 if phase == 0.90 else 180)
    durations[-1] = 900
    output = ASSETS / f"cpu-router-8l4-{name}.gif"
    frames[0].save(output, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, disposal=2, optimize=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    for step in (0, 2, 4):
        draw_scene(step, 0.75).save(PREVIEW / f"cpu-router-8l4-{name}-step{step}.png")
    print(output.relative_to(ROOT), output.stat().st_size, "bytes")


def main() -> None:
    for name, scene in (("allreduce", allreduce),
                        ("allgather", allgather),
                        ("reducescatter", reducescatter)):
        render(name, scene)


if __name__ == "__main__":
    main()
