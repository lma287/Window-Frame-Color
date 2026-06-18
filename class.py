from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt


# ======================================================
# 1. File settings
# ======================================================

INPUT_CSV = Path("ai_rgb_cielab_results.csv")

OUTPUT_CSV = Path("ai_rgb_cielab_results_refined_classified.csv")
OUTPUT_FINE_BAR = Path("rgb_class_barplot.png")

OUTPUT_PER_FINE_DIR = Path("rgb_class_barplots")
OUTPUT_PER_FINE_DIR.mkdir(exist_ok=True)


# ======================================================
# 2. Fine class order
# ======================================================

FINE_CLASS_ORDER = [
    "white",
    "light grey",
    "medium grey",
    "dark grey",
    "black",

    "cream",
    "beige",
    "taupe / greige",

    "tan",
    "light brown",
    "medium brown",
    "dark brown",
    "reddish brown",

    "olive / yellow green",
    "grey green",
    "dark green",
    "green",
    "light green",

    "dark teal",
    "teal",
    "grey teal",

    "blue grey",
    "blue",
    "navy / slate blue",

    "red",
    "orange",
    "yellow / gold",

    "purple / pink",

    "invalid_rgb",
]


# ======================================================
# 3. Class colours for main bar plot
# ======================================================

FINE_CLASS_COLOURS = {
    "white": "#ffffff",
    "light grey": "#d8d8d8",
    "medium grey": "#808080",
    "dark grey": "#404040",
    "black": "#000000",

    "cream": "#ead8aa",
    "beige": "#d2b48c",
    "taupe / greige": "#a99a86",

    "tan": "#c49a6c",
    "light brown": "#b88755",
    "medium brown": "#7a4a24",
    "dark brown": "#3b2414",
    "reddish brown": "#7a2e1f",

    "olive / yellow green": "#8a9a28",
    "grey green": "#6f8064",
    "dark green": "#0b4d20",
    "green": "#2ca02c",
    "light green": "#8bcf6a",

    "dark teal": "#014d4e",
    "teal": "#008080",
    "grey teal": "#6f9a9a",

    "blue grey": "#8fa3b5",
    "blue": "#1f77ff",
    "navy / slate blue": "#263850",

    "red": "#d62728",
    "orange": "#ff7f0e",
    "yellow / gold": "#d8b000",

    "purple / pink": "#b05aa0",

    "invalid_rgb": "#bbbbbb",
}


# ======================================================
# 4. Helper functions
# ======================================================

def clamp_rgb_value(x):
    return max(0, min(255, int(round(float(x)))))


def rgb_to_hex(r, g, b):
    return "#{:02x}{:02x}{:02x}".format(
        clamp_rgb_value(r),
        clamp_rgb_value(g),
        clamp_rgb_value(b),
    )


def readable_text_colour(hex_colour):
    hex_colour = hex_colour.lstrip("#")
    r = int(hex_colour[0:2], 16)
    g = int(hex_colour[2:4], 16)
    b = int(hex_colour[4:6], 16)

    brightness = 0.299 * r + 0.587 * g + 0.114 * b

    if brightness >= 160:
        return "black"
    else:
        return "white"


def safe_filename(text):
    text = str(text).lower()
    text = text.replace("/", "_")
    text = text.replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]+", "", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


# ======================================================
# 5. Refined RGB classification
# ======================================================

def classify_rgb_refined(r, g, b):
    """
    Refined RGB classification.

    This function directly assigns RGB values to final classes.
    No post-classification merge is used.
    """

    if pd.isna(r) or pd.isna(g) or pd.isna(b):
        return "invalid_rgb"

    r = clamp_rgb_value(r)
    g = clamp_rgb_value(g)
    b = clamp_rgb_value(b)

    max_rgb = max(r, g, b)
    min_rgb = min(r, g, b)
    diff = max_rgb - min_rgb
    brightness = (r + g + b) / 3.0

    # ==================================================
    # 0. Strong primary / secondary colours
    # ==================================================

    # [255, 255, 0] and strong yellow
    if (
        r >= 220
        and g >= 220
        and b <= 90
    ):
        return "yellow / gold"

    # [0, 0, 255] and strong blue
    if (
        b >= 200
        and r <= 90
        and g <= 140
        and b >= r + 80
        and b >= g + 80
    ):
        return "blue"

    # Strong red
    if (
        r >= 200
        and g <= 95
        and b <= 110
        and r >= g + 80
        and r >= b + 70
    ):
        return "red"

    # ==================================================
    # 1. White / grey / black
    # ==================================================

    if brightness >= 235 and min_rgb >= 220:
        return "white"

    if brightness >= 225 and min_rgb >= 205 and diff <= 45:
        return "white"

    if diff <= 25:
        if brightness >= 210:
            return "light grey"
        elif brightness >= 90:
            return "medium grey"
        elif brightness >= 45:
            return "dark grey"
        else:
            return "black"

    if max_rgb <= 35:
        return "black"

    if brightness < 45 and diff <= 45:
        return "black"

    # Cool muted grey / blue-grey
    if (
        diff <= 55
        and b >= r
        and b >= g
        and brightness >= 85
    ):
        return "blue grey"

    # Warm muted grey / taupe
    if (
        90 <= brightness <= 190
        and diff <= 55
        and r >= b
        and r >= g - 12
    ):
        return "taupe / greige"

    # ==================================================
    # 2. Cream / beige
    # ==================================================

    if (
        r >= 225
        and g >= 210
        and b >= 170
        and r >= g
        and g >= b
    ):
        return "cream"

    if (
        r >= 200
        and g >= 175
        and b >= 125
        and r >= g
        and g >= b
        and brightness >= 165
    ):
        return "beige"

    # ==================================================
    # 3. Yellow / olive / yellow-green
    # ==================================================

    # Strong yellow / gold
    if (
        r >= 185
        and g >= 165
        and b <= 110
        and abs(r - g) <= 80
    ):
        return "yellow / gold"

    # Muted olive / yellow-green, not pure yellow
    if (
        g >= r
        and r >= 0.50 * g
        and b <= 0.75 * r
        and brightness >= 70
    ):
        return "olive / yellow green"

    if (
        r >= g
        and g >= 0.68 * r
        and b <= 0.60 * g
        and r < 185
    ):
        return "olive / yellow green"

    # ==================================================
    # 4. Brown family
    #    Dark brown is restored by RGB brightness.
    # ==================================================

    # Reddish brown / mahogany
    if (
        r >= g
        and r >= b
        and brightness < 145
        and g >= 0.18 * r
        and b >= 0.12 * r
        and b <= 0.72 * r
        and g <= 0.72 * r
    ):
        # If G and B are very low, it is visually closer to red
        if g < 0.38 * r and b < 0.38 * r:
            return "red"
        return "reddish brown"

    # Standard brown axis: R >= G >= B
    if (
        r >= g
        and g >= b
        and brightness < 200
        and g >= 0.35 * r
        and b >= 0.14 * r
    ):
        if brightness >= 165:
            return "tan"
        elif brightness >= 125:
            return "light brown"
        elif brightness >= 80:
            return "medium brown"
        else:
            return "dark brown"

    # Very dark warm brown-like colours
    if (
        r >= g
        and g >= b
        and brightness < 80
        and g >= 0.25 * r
        and b >= 0.08 * r
    ):
        return "dark brown"

    # ==================================================
    # 5. Teal family
    #    No cyan output.
    # ==================================================

    # Grey teal: muted green-blue
    if (
        g >= 80
        and b >= 80
        and abs(g - b) <= 45
        and r <= min(g, b) + 35
        and diff <= 90
        and brightness >= 80
    ):
        return "grey teal"

    # Strong teal / cyan-like colours go to teal
    if (
        g >= 0.70 * max_rgb
        and b >= 0.70 * max_rgb
        and r <= 0.80 * max_rgb
    ):
        if brightness < 90:
            return "dark teal"
        else:
            return "teal"

    # ==================================================
    # 6. Blue family
    # ==================================================

    if b >= r and b >= g:
        if (
            b >= 160
            and b >= r + 55
            and b >= g + 45
        ):
            return "blue"

        if brightness < 105:
            return "navy / slate blue"

        if diff <= 65:
            return "blue grey"

        if g >= 0.60 * b and r <= 0.85 * b:
            if brightness < 100:
                return "dark teal"
            else:
                return "teal"

        if r >= 0.50 * b:
            return "purple / pink"

        return "blue"

    # ==================================================
    # 7. Green family
    #    Green is classified by RGB values.
    # ==================================================

    if g >= r and g >= b:

        # Light green:
        # includes [100, 180, 80] and [100, 220, 100].
        # G is clearly dominant, but R and B are still present.
        if (
            g >= 170
            and brightness >= 115
            and r >= 0.40 * g
            and b >= 0.35 * g
            and diff <= 130
        ):
            return "light green"

        # Dark green:
        # green is dominant and overall dark.
        if (
            brightness < 85
            and g >= r + 15
            and g >= b + 15
        ):
            return "dark green"

        # Grey green:
        # muted green with enough red and blue.
        # Example: [100, 150, 80]
        if (
            diff <= 90
            and r >= 0.40 * g
            and b >= 0.35 * g
            and brightness >= 70
        ):
            return "grey green"

        # Olive / yellow-green:
        # green with high red and low blue.
        if (
            r >= 0.55 * g
            and b <= 0.70 * r
        ):
            return "olive / yellow green"

        if brightness < 95:
            return "dark green"

        if brightness >= 150:
            return "light green"

        return "green"

    # ==================================================
    # 8. Red / orange / yellow / purple-pink
    # ==================================================

    if r >= g and r >= b:

        # Purple / pink:
        # R and B both visible.
        if (
            b >= 0.45 * r
            and r >= 120
            and brightness >= 80
        ):
            return "purple / pink"

        # Yellow / gold
        if (
            r >= 180
            and g >= 0.72 * r
            and b <= 0.65 * g
        ):
            return "yellow / gold"

        # Orange
        if (
            g >= 0.35 * r
            and b <= 0.80 * g
            and brightness >= 80
        ):
            return "orange"

        # No brick red output.
        return "red"

    return "medium grey"


# ======================================================
# 6. Load CSV
# ======================================================

df = pd.read_csv(INPUT_CSV)

required_cols = ["ai_r", "ai_g", "ai_b"]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")


# ======================================================
# 7. Apply refined classification
# ======================================================

df["rgb_fine_class"] = df.apply(
    lambda row: classify_rgb_refined(
        row["ai_r"],
        row["ai_g"],
        row["ai_b"],
    ),
    axis=1
)

df["ai_r_int"] = df["ai_r"].round().astype("Int64").clip(0, 255)
df["ai_g_int"] = df["ai_g"].round().astype("Int64").clip(0, 255)
df["ai_b_int"] = df["ai_b"].round().astype("Int64").clip(0, 255)

df["rgb_hex"] = df.apply(
    lambda row: rgb_to_hex(row["ai_r"], row["ai_g"], row["ai_b"])
    if not pd.isna(row["ai_r"])
    and not pd.isna(row["ai_g"])
    and not pd.isna(row["ai_b"])
    else "#bbbbbb",
    axis=1
)

df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

print(f"Saved classified CSV to: {OUTPUT_CSV}")

# ======================================================
# Region-based fine colour class bar plots
# ======================================================

OUTPUT_REGION_CLASS_CSV = Path("region_rgb_fine_class_counts.csv")
OUTPUT_REGION_CLASS_DIR = Path("region_rgb_fine_class_barplots")

OUTPUT_REGION_CLASS_DIR.mkdir(exist_ok=True)


# ======================================================
# 1. Region list and aliases
# ======================================================

REGION_ALIASES = {
    "Northland": [
        "northland",
    ],
    "Auckland": [
        "auckland",
    ],
    "Waikato": [
        "waikato",
    ],
    "Bay Of Plenty": [
        "bay of plenty",
        "bay_of_plenty",
    ],
    "Gisborne": [
        "gisborne",
    ],
    "Hawke's Bay": [
        "hawke's bay",
        "hawkes bay",
        "hawke s bay",
        "hawke_bay",
        "hawkes_bay",
    ],
    "Taranaki": [
        "taranaki",
    ],
    "Manawatu / Whanganui": [
        "manawatu",
        "whanganui",
        "wanganui",
        "manawatu whanganui",
        "manawatu_wanganui",
    ],
    "Wellington": [
        "wellington",
    ],
    "Nelson / Tasman": [
        "nelson",
        "tasman",
        "nelson tasman",
    ],
    "Marlborough": [
        "marlborough",
    ],
    "West Coast": [
        "west coast",
        "west_coast",
    ],
    "Canterbury": [
        "canterbury",
    ],
    "Otago": [
        "otago",
    ],
    "Southland": [
        "southland",
    ],
}

REGION_ORDER = list(REGION_ALIASES.keys())


# ======================================================
# 2. Detect region from crop_path
# ======================================================

def normalise_region_text(text):
    text = str(text).lower()

    text = text.replace("\\", " ")
    text = text.replace("/", " ")
    text = text.replace("_", " ")

    text = text.replace("’", "'")
    text = text.replace("`", "'")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def detect_region_from_path(path_text):
    text = normalise_region_text(path_text)

    for region, aliases in REGION_ALIASES.items():
        for alias in aliases:
            alias_norm = normalise_region_text(alias)

            pattern = r"(?<![a-z])" + re.escape(alias_norm) + r"(?![a-z])"

            if re.search(pattern, text):
                return region

    return "Unknown"


if "crop_path" not in df.columns:
    raise ValueError("Missing required column: crop_path")

df["region"] = df["crop_path"].apply(detect_region_from_path)


# ======================================================
# 3. Save region x colour-class count table
# ======================================================

region_class_counts = (
    df
    .groupby(["region", "rgb_fine_class"])
    .size()
    .reset_index(name="count")
)

region_class_pivot = (
    region_class_counts
    .pivot(index="region", columns="rgb_fine_class", values="count")
    .fillna(0)
    .astype(int)
)

region_class_pivot = region_class_pivot.reindex(
    REGION_ORDER + ["Unknown"],
    fill_value=0
)

region_class_pivot = region_class_pivot.reindex(
    columns=FINE_CLASS_ORDER,
    fill_value=0
)

region_class_pivot.to_csv(
    OUTPUT_REGION_CLASS_CSV,
    encoding="utf-8-sig"
)

print(f"Saved region colour-class counts to: {OUTPUT_REGION_CLASS_CSV}")


# ======================================================
# 4. Draw one fine-class bar plot for each region
# ======================================================

for region in REGION_ORDER + ["Unknown"]:

    df_region = df[df["region"] == region].copy()

    if len(df_region) == 0:
        continue

    counts = (
        df_region["rgb_fine_class"]
        .value_counts()
        .reindex(FINE_CLASS_ORDER, fill_value=0)
    )

    counts = counts[counts > 0]

    if len(counts) == 0:
        continue

    labels = counts.index.tolist()
    values = counts.values

    bar_colours = [
        FINE_CLASS_COLOURS.get(label, "#bbbbbb")
        for label in labels
    ]

    plt.figure(figsize=(max(14, len(labels) * 0.65), 7))

    plt.bar(
        labels,
        values,
        color=bar_colours,
        edgecolor="black",
        linewidth=0.7
    )

    plt.xlabel("Fine colour class")
    plt.ylabel("Count")
    plt.title(f"RGB colour classes by region: {region}")

    plt.xticks(rotation=45, ha="right")

    for i, value in enumerate(values):
        plt.text(
            i,
            value,
            str(int(value)),
            ha="center",
            va="bottom",
            fontsize=9
        )

    # x-axis label background colour = class colour
    ax = plt.gca()
    tick_labels = ax.get_xticklabels()

    for tick_label, label in zip(tick_labels, labels):
        hex_colour = FINE_CLASS_COLOURS.get(label, "#bbbbbb")

        tick_label.set_bbox(
            dict(
                facecolor=hex_colour,
                edgecolor="black",
                linewidth=0.4,
                boxstyle="round,pad=0.20"
            )
        )
        tick_label.set_color(readable_text_colour(hex_colour))
        tick_label.set_fontsize(9)

    ymax = max(values) if len(values) > 0 else 1
    plt.ylim(0, ymax * 1.15 + 1)

    plt.tight_layout()

    out_path = OUTPUT_REGION_CLASS_DIR / f"{safe_filename(region)}_fine_class_barplot.png"

    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved region fine-class bar plot to: {out_path}")


# ======================================================
# 8. Main fine class bar plot
#    Label background colour = class colour
# ======================================================

fine_counts = (
    df["rgb_fine_class"]
    .value_counts()
    .reindex(FINE_CLASS_ORDER, fill_value=0)
)

fine_counts = fine_counts[fine_counts > 0]

print("\nFine class counts:")
print(fine_counts)

labels = fine_counts.index.tolist()
values = fine_counts.values

bar_colours = [
    FINE_CLASS_COLOURS.get(label, "#bbbbbb")
    for label in labels
]

plt.figure(figsize=(max(14, len(labels) * 0.65), 7))

plt.bar(
    labels,
    values,
    color=bar_colours,
    edgecolor="black",
    linewidth=0.7
)

plt.xlabel("Fine colour class")
plt.ylabel("Count")
plt.title("RGB colour classes")
plt.xticks(rotation=45, ha="right")

for i, value in enumerate(values):
    plt.text(
        i,
        value,
        str(int(value)),
        ha="center",
        va="bottom",
        fontsize=9
    )

ax = plt.gca()
tick_labels = ax.get_xticklabels()

for tick_label, label in zip(tick_labels, labels):
    hex_colour = FINE_CLASS_COLOURS.get(label, "#bbbbbb")

    tick_label.set_bbox(
        dict(
            facecolor=hex_colour,
            edgecolor="black",
            linewidth=0.4,
            boxstyle="round,pad=0.20"
        )
    )
    tick_label.set_color(readable_text_colour(hex_colour))
    tick_label.set_fontsize(9)

ymax = max(values) if len(values) > 0 else 1
plt.ylim(0, ymax * 1.15 + 1)

plt.tight_layout()
plt.savefig(OUTPUT_FINE_BAR, dpi=300)
plt.close()

print(f"Saved fine class bar plot to: {OUTPUT_FINE_BAR}")


# ======================================================
# 9. Per-fine-class RGB bar plots
#    RGB label background colour = actual RGB
# ======================================================

MAX_RGB_PER_FINE_CLASS = None
COUNT_LABEL_LIMIT = 200

for fine_class in fine_counts.index:
    if fine_class == "invalid_rgb":
        continue

    df_class = df[df["rgb_fine_class"] == fine_class].copy()

    if len(df_class) == 0:
        continue

    df_class["rgb_label"] = (
        "[" +
        df_class["ai_r_int"].astype(str) + "," +
        df_class["ai_g_int"].astype(str) + "," +
        df_class["ai_b_int"].astype(str) +
        "]"
    )

    # Sort by perceived brightness: dark -> light
    df_class["rgb_brightness"] = (
        0.299 * df_class["ai_r_int"].astype(int)
        + 0.587 * df_class["ai_g_int"].astype(int)
        + 0.114 * df_class["ai_b_int"].astype(int)
    )

    rgb_counts = (
        df_class
        .groupby(
            [
                "rgb_label",
                "rgb_hex",
                "ai_r_int",
                "ai_g_int",
                "ai_b_int",
                "rgb_brightness",
            ]
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            ["rgb_brightness", "ai_r_int", "ai_g_int", "ai_b_int"],
            ascending=[True, True, True, True]
        )
    )

    if MAX_RGB_PER_FINE_CLASS is not None and len(rgb_counts) > MAX_RGB_PER_FINE_CLASS:
        rgb_counts = rgb_counts.head(MAX_RGB_PER_FINE_CLASS)

    if len(rgb_counts) == 0:
        continue

    fig_width = max(12, len(rgb_counts) * 0.42)

    plt.figure(figsize=(fig_width, 7))

    plt.bar(
        rgb_counts["rgb_label"],
        rgb_counts["count"],
        color=rgb_counts["rgb_hex"],
        edgecolor="black",
        linewidth=0.6,
        width=0.85
    )

    plt.xlabel("RGB value, sorted by RGB")
    plt.ylabel("Count")
    plt.title(f"RGB distribution inside fine class: {fine_class}")

    plt.xticks(rotation=60, ha="right")

    if len(rgb_counts) <= COUNT_LABEL_LIMIT:
        for i, value in enumerate(rgb_counts["count"]):
            plt.text(
                i,
                value,
                str(int(value)),
                ha="center",
                va="bottom",
                fontsize=8
            )

    ax = plt.gca()
    tick_labels = ax.get_xticklabels()

    for tick_label, hex_colour in zip(tick_labels, rgb_counts["rgb_hex"]):
        tick_label.set_bbox(
            dict(
                facecolor=hex_colour,
                edgecolor="black",
                linewidth=0.4,
                boxstyle="round,pad=0.18"
            )
        )
        tick_label.set_color(readable_text_colour(hex_colour))
        tick_label.set_fontsize(8)

    ymax = rgb_counts["count"].max()
    plt.ylim(0, ymax * 1.15 + 1)

    plt.tight_layout()

    out_path = OUTPUT_PER_FINE_DIR / f"{safe_filename(fine_class)}_rgb_barplot.png"

    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved per-fine-class RGB bar plot to: {out_path}")

    # ======================================================
# 10. Per-region, per-fine-class RGB bar plots
#     Region x Colour class x RGB value
# ======================================================

OUTPUT_REGION_CLASS_RGB_DIR = Path("region_rgb_fine_class_rgb_barplots")
OUTPUT_REGION_CLASS_RGB_CSV = Path("region_rgb_fine_class_rgb_counts.csv")

OUTPUT_REGION_CLASS_RGB_DIR.mkdir(exist_ok=True)


# Save all detailed counts into one CSV
all_region_class_rgb_counts = []

# None = plot all RGB values
# If one image becomes too wide, set for example 100
MAX_RGB_PER_REGION_CLASS = None

# If too many RGB bars, do not print count text above every bar
COUNT_LABEL_LIMIT = 200


for region in REGION_ORDER + ["Unknown"]:

    df_region = df[df["region"] == region].copy()

    if len(df_region) == 0:
        continue

    region_dir = OUTPUT_REGION_CLASS_RGB_DIR / safe_filename(region)
    region_dir.mkdir(exist_ok=True)

    for fine_class in FINE_CLASS_ORDER:

        if fine_class == "invalid_rgb":
            continue

        df_sub = df_region[df_region["rgb_fine_class"] == fine_class].copy()

        if len(df_sub) == 0:
            continue

        # RGB label
        df_sub["rgb_label"] = (
            "[" +
            df_sub["ai_r_int"].astype(str) + "," +
            df_sub["ai_g_int"].astype(str) + "," +
            df_sub["ai_b_int"].astype(str) +
            "]"
        )

        # Sort by perceived brightness: dark -> light
        df_sub["rgb_brightness"] = (
            0.299 * df_sub["ai_r_int"].astype(int)
            + 0.587 * df_sub["ai_g_int"].astype(int)
            + 0.114 * df_sub["ai_b_int"].astype(int)
        )

        rgb_counts = (
            df_sub
            .groupby(
                [
                    "region",
                    "rgb_fine_class",
                    "rgb_label",
                    "rgb_hex",
                    "ai_r_int",
                    "ai_g_int",
                    "ai_b_int",
                    "rgb_brightness",
                ]
            )
            .size()
            .reset_index(name="count")
            .sort_values(
                ["rgb_brightness", "ai_r_int", "ai_g_int", "ai_b_int"],
                ascending=[True, True, True, True]
            )
        )

        if MAX_RGB_PER_REGION_CLASS is not None and len(rgb_counts) > MAX_RGB_PER_REGION_CLASS:
            rgb_counts = rgb_counts.head(MAX_RGB_PER_REGION_CLASS)

        if len(rgb_counts) == 0:
            continue

        all_region_class_rgb_counts.append(rgb_counts)

        # ==================================================
        # Draw plot
        # ==================================================

        fig_width = max(12, len(rgb_counts) * 0.42)

        plt.figure(figsize=(fig_width, 7))

        plt.bar(
            rgb_counts["rgb_label"],
            rgb_counts["count"],
            color=rgb_counts["rgb_hex"],
            edgecolor="black",
            linewidth=0.6,
            width=0.85
        )

        plt.xlabel("RGB value, sorted by RGB")
        plt.ylabel("Count")
        plt.title(f"RGB distribution: {region} - {fine_class}")

        plt.xticks(rotation=60, ha="right")

        # Count labels
        if len(rgb_counts) <= COUNT_LABEL_LIMIT:
            for i, value in enumerate(rgb_counts["count"]):
                plt.text(
                    i,
                    value,
                    str(int(value)),
                    ha="center",
                    va="bottom",
                    fontsize=8
                )

        # RGB label background = actual RGB colour
        ax = plt.gca()
        tick_labels = ax.get_xticklabels()

        for tick_label, hex_colour in zip(tick_labels, rgb_counts["rgb_hex"]):
            tick_label.set_bbox(
                dict(
                    facecolor=hex_colour,
                    edgecolor="black",
                    linewidth=0.4,
                    boxstyle="round,pad=0.18"
                )
            )
            tick_label.set_color(readable_text_colour(hex_colour))
            tick_label.set_fontsize(8)

        ymax = rgb_counts["count"].max()
        plt.ylim(0, ymax * 1.15 + 1)

        plt.tight_layout()

        out_path = (
            region_dir
            / f"{safe_filename(fine_class)}_rgb_barplot.png"
        )

        plt.savefig(out_path, dpi=300)
        plt.close()

        print(f"Saved region-class RGB bar plot to: {out_path}")


# ======================================================
# 11. Save detailed region x class x RGB counts
# ======================================================

if len(all_region_class_rgb_counts) > 0:
    detailed_counts_df = pd.concat(
        all_region_class_rgb_counts,
        ignore_index=True
    )

    detailed_counts_df.to_csv(
        OUTPUT_REGION_CLASS_RGB_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Saved detailed region-class RGB counts to: {OUTPUT_REGION_CLASS_RGB_CSV}")

else:
    print("No region-class RGB counts were generated.")
