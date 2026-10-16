"""CropWhisperAI - colour system block diagram (Figure 1) and method workflow (Figure 5), 600 dpi."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family": "Liberation Sans", "savefig.dpi": 600, "savefig.bbox": "tight"})

INK, MUTED = "#1f2328", "#57606a"
BUS = {"Single-wire": "#eb6834", "1-Wire": "#2a78d6", "I²C": "#1baf7a", "ADC": "#8a63d2", "Wi-Fi": "#e34948"}
LAYER = {"sense": ("#fdf1e8", "#eb6834"), "edge": ("#eaf2fc", "#2a78d6"), "out": ("#e9f7f1", "#1baf7a"), "pwr": ("#fff6dc", "#c98500")}


def box(ax, x, y, w, h, fc, ec, lw=1.2, r=0.02, z=2):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z)
    ax.add_patch(p); return p


def txt(ax, x, y, s, size=7.5, w="normal", c=INK, ha="center", va="center", z=5, style="normal"):
    ax.text(x, y, s, fontsize=size, fontweight=w, color=c, ha=ha, va=va, zorder=z, fontstyle=style, linespacing=1.25)


def arrow(ax, p, q, c=INK, lw=1.3, style="-|>", ms=9, ls="-", z=4, cs="arc3"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=c, lw=lw, linestyle=ls, zorder=z,
                                 connectionstyle=cs, shrinkA=0, shrinkB=0))


def block_diagram():
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.4); ax.axis("off")
    # ---- layers
    for (x, w, key, title) in ((0.05, 2.75, "sense", "SENSING LAYER"), (3.25, 3.45, "edge", "EDGE LAYER – ESP32"),
                               (7.15, 2.8, "out", "OUTPUT LAYER")):
        fc, ec = LAYER[key]
        box(ax, x, 0.35, w, 5.25, fc, ec, lw=1.4, r=0.08, z=1)
        txt(ax, x + 0.15 if key == "edge" else x + w / 2, 5.38, title, 8, "bold", ec, ha="left" if key == "edge" else "center")
    # ---- sensors
    sensors = [("DHT22", "Air temperature, humidity", "GPIO 4", "Single-wire"),
               ("DS18B20", "Root-zone temperature", "GPIO 5", "1-Wire"),
               ("Soil probe", "Capacitive soil moisture", "GPIO 34", "ADC"),
               ("BH1750", "Illuminance (lx)", "SDA 21 / SCL 22", "I²C"),
               ("Water level", "Reservoir level strip", "GPIO 35", "ADC")]
    ys = [4.55, 3.7, 2.85, 2.0, 1.15]
    for (name, q, pin, bus), y in zip(sensors, ys):
        c = BUS[bus]
        box(ax, 0.25, y - 0.33, 2.35, 0.66, "white", c, lw=1.4, r=0.05)
        box(ax, 0.25, y - 0.33, 0.12, 0.66, c, c, lw=0, r=0.02, z=3)
        txt(ax, 0.5, y + 0.12, name, 7.6, "bold", INK, ha="left")
        txt(ax, 0.5, y - 0.14, q, 6.4, c=MUTED, ha="left")
        txt(ax, 2.52, y + 0.12, pin, 5.6, c=c, ha="right", w="bold")
        arrow(ax, (2.6, y), (3.55, max(min(y, 4.55), 1.15) if False else y), c=c, lw=1.5)
    # ---- ESP32 pipeline
    box(ax, 3.55, 0.6, 2.85, 4.55, "white", "#2a78d6", lw=1.6, r=0.06)
    txt(ax, 3.72, 4.93, "Xtensa dual-core · 520 KB SRAM", 6.4, "bold", "#1c5cab", ha="left")
    steps = [("Read sensors every 5 min", "#cde2fb"),
             ("Validate each sensor + flag;\ncarry forward (max. 15 min)", "#b7d3f6"),
             ("24-h ring buffer\n(6 channels × 288, 6.9 KB)", "#9ec5f4"),
             ("8 streaming features\n(max, mean, thermal load, VPD)", "#86b6ef"),
             ("Logistic model (9 parameters)\nheat-risk in next 30 min", "#5598e7")]
    sy = [4.5, 3.8, 3.05, 2.25, 1.45]
    for (s, c), y in zip(steps, sy):
        box(ax, 3.8, y - 0.3, 2.35, 0.6, c, "#2a78d6", lw=0.8, r=0.05)
        txt(ax, 4.975, y, s, 6.4, c=INK if c != "#5598e7" else "white", w="bold" if c == "#5598e7" else "normal")
    for a, b in zip(sy[:-1], sy[1:]):
        arrow(ax, (4.975, a - 0.3), (4.975, b + 0.3), c="#1c5cab", lw=1.1, ms=7)
    # alarm decision
    box(ax, 3.8, 0.68, 2.35, 0.36, "#e34948", "#e34948", lw=0, r=0.05)
    txt(ax, 4.975, 0.86, "HEAT WARNING if p ≥ τ", 6.4, "bold", "white")
    arrow(ax, (4.975, sy[-1] - 0.3), (4.975, 1.04), c="#e34948", lw=1.1, ms=6)
    # ---- power
    box(ax, 5.05, 5.72, 2.3, 0.55, LAYER["pwr"][0], LAYER["pwr"][1], lw=1.3, r=0.06)
    txt(ax, 6.2, 6.0, "Power: 5 V USB / DC jack", 6.6, "bold", "#8a5a00")
    arrow(ax, (6.2, 5.72), (6.2, 5.15), c="#c98500", lw=1.5)
    # ---- outputs
    outs = [("0.96″ OLED (SSD1306)", "Live readings, risk %,\nwarning banner", 4.3, BUS["I²C"], "I²C"),
            ("Cloud endpoint", "ThingSpeak / REST API\n(JSON every 5 min)", 2.95, BUS["Wi-Fi"], "Wi-Fi"),
            ("Web dashboard", "Time series, alarms,\nhistorical records", 1.6, "#1baf7a", "")]
    for name, sub, y, c, lab in outs:
        box(ax, 7.35, y - 0.45, 2.4, 0.9, "white", c, lw=1.4, r=0.05)
        box(ax, 7.35, y - 0.45, 0.12, 0.9, c, c, lw=0, r=0.02, z=3)
        txt(ax, 7.6, y + 0.18, name, 7.4, "bold", ha="left")
        txt(ax, 7.6, y - 0.16, sub, 6.2, c=MUTED, ha="left")
    arrow(ax, (6.4, 4.3), (7.35, 4.3), c=BUS["I²C"], lw=1.5); txt(ax, 6.925, 4.45, "I²C", 5.6, "bold", BUS["I²C"])
    arrow(ax, (6.4, 2.95), (7.35, 2.95), c=BUS["Wi-Fi"], lw=1.5); txt(ax, 6.925, 3.1, "Wi-Fi", 5.6, "bold", BUS["Wi-Fi"])
    arrow(ax, (8.55, 2.5), (8.55, 2.05), c="#1baf7a", lw=1.5)
    # ---- legend (buses)
    lx = 0.25
    for i, (k, c) in enumerate(BUS.items()):
        ax.plot([lx + i * 1.95, lx + i * 1.95 + 0.35], [0.12, 0.12], color=c, lw=2.2)
        txt(ax, lx + i * 1.95 + 0.43, 0.12, k, 6.4, ha="left", c=MUTED)
    fig.savefig("figures/fig1_block_diagram_colour.png"); fig.savefig("figures/fig1_block_diagram_colour.pdf"); plt.close(fig)


def workflow():
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis("off")
    stages = [
        ("1  Acquire", "30,816 records\n5-min interval\n1 Jun–15 Sep 2026", "#eb6834", "#fdf1e8"),
        ("2  Quality control", "24 flagged records\ncarry-forward\nnot used as samples", "#c98500", "#fff6dc"),
        ("3  Episodes", "air > 35 °C or\nsoil > 32 °C, debounced\n→ 42 episodes", "#e34948", "#fdecec"),
        ("4  Labels", "onset within H\n(15, 30, 60 min)\n26,042 scored steps", "#8a63d2", "#f1ecfb"),
        ("5  Features", "64 causal features\n(lags, 60-min stats,\nVPD, thermal load)", "#1baf7a", "#e9f7f1"),
    ]
    stages2 = [
        ("6  Models", "2 rules, time of day,\nLR, RF,\nLightGBM, MLP", "#2a78d6", "#eaf2fc"),
        ("7  Blocked CV", "10 day-blocks\n1-day embargo\nbootstrap 95% CI", "#2a78d6", "#eaf2fc"),
        ("8  Analyses", "event metrics\nseasonal hold-out\nablation, SHAP", "#2a78d6", "#eaf2fc"),
        ("9  On-device", "8-feature LR\n→ C header\nhost-verified", "#1c5cab", "#cde2fb"),
        ("10  ESP32", "streaming inference\nOLED warning\nlatency: micros()", "#e34948", "#fdecec"),
    ]
    w, h, gap = 1.72, 1.45, 0.3
    def row(items, y0):
        xs = []
        for i, (t, s, ec, fc) in enumerate(items):
            x = 0.1 + i * (w + gap)
            box(ax, x, y0, w, h, fc, ec, lw=1.4, r=0.07)
            box(ax, x, y0 + h - 0.36, w, 0.36, ec, ec, lw=0, r=0.07, z=3)
            txt(ax, x + w / 2, y0 + h - 0.18, t, 7.2, "bold", "white")
            txt(ax, x + w / 2, y0 + (h - 0.36) / 2, s, 6.0, c=INK)
            xs.append(x)
        for x in xs[:-1]:
            arrow(ax, (x + w, y0 + h / 2 - 0.18), (x + w + gap, y0 + h / 2 - 0.18), c=MUTED, lw=1.2, ms=8)
        return xs
    x1 = row(stages, 2.9)
    x2 = row(stages2, 0.35)
    # connector from stage 5 down to stage 6
    xa, xb, ym = x1[-1] + w / 2, x2[0] + w / 2, 2.35
    ax.plot([xa, xa, xb], [2.9, ym, ym], color=MUTED, lw=1.2, zorder=1, solid_joinstyle="round")
    arrow(ax, (xb, ym), (xb, 0.35 + h), c=MUTED, lw=1.2, ms=8)
    txt(ax, 5.0, 2.52, "Python pipeline (scripts 01–05)", 6.5, c=MUTED, style="italic")
    fig.savefig("figures/fig5_workflow.png"); fig.savefig("figures/fig5_workflow.pdf"); plt.close(fig)


if __name__ == "__main__":
    block_diagram(); workflow(); print("ok")
