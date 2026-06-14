from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ======================================================
# 1. Plot settings
# ======================================================

INPUT_CSV = Path("ai_rgb_cielab_results.csv")

# None = plot all points
MAX_PLOT_POINTS = None
RANDOM_SEED = 42

OUTPUT_L_CHROMA_TRUE = Path("ai_plot_L_chroma_true_colours.png")
OUTPUT_AB_TRUE = Path("ai_plot_ab_true_colours.png")

OUTPUT_REGION_DIR = Path("region_cielab_plots")
OUTPUT_REGION_DIR.mkdir(exist_ok=True)

OUTPUT_REGION_COUNTS_CSV = Path("region_cielab_sample_counts.csv")


# ======================================================
# 2. RGB -> CIELAB conversion
#    Used only if L/a/b/chroma/colour_hex are missing
# ======================================================

def srgb_to_linear(rgb):
    rgb = np.asarray(rgb, dtype=float)

    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )


def rgb01_to_xyz(rgb):
    rgb_linear = srgb_to_linear(rgb)

    matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])

    return rgb_linear @ matrix.T


def xyz_to_lab(xyz):
    white = np.array([0.95047, 1.00000, 1.08883])
    xyz_scaled = xyz / white

    epsilon = 216 / 24389
    kappa = 24389 / 27

    f = np.where(
        xyz_scaled > epsilon,
        np.cbrt(xyz_scaled),
        (kappa * xyz_scaled + 16) / 116,
    )

    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])

    return np.stack([L, a, b], axis=-1)


def rgb255_to_lab(rgb255):
    rgb01 = np.asarray(rgb255, dtype=float).reshape(1, 3) / 255.0
    xyz = rgb01_to_xyz(rgb01)
    lab = xyz_to_lab(xyz)[0]
    return lab


def rgb_to_hex(rgb255):
    r, g, b = np.clip(np.round(rgb255), 0, 255).astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"


# ======================================================
# 3. Region detection
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
        "bay-of-plenty",
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
        "hawke-bay",
        "hawkes-bay",
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
        "manawatu-whanganui",
    ],
    "Wellington": [
        "wellington",
    ],
    "Nelson / Tasman": [
        "nelson",
        "tasman",
        "nelson tasman",
        "nelson_tasman",
        "nelson-tasman",
    ],
    "Marlborough": [
        "marlborough",
    ],
    "West Coast": [
        "west coast",
        "west_coast",
        "west-coast",
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


def normalise_region_text(text):
    text = str(text).lower()

    text = text.replace("\\", " ")
    text = text.replace("/", " ")
    text = text.replace("_", " ")
    text = text.replace("-", " ")

    text = text.replace("’", "'")
    text = text.replace("`", "'")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def detect_region_from_text(path_text):
    text = normalise_region_text(path_text)

    for region, aliases in REGION_ALIASES.items():
        for alias in aliases:
            alias_norm = normalise_region_text(alias)

            pattern = r"(?<![a-z])" + re.escape(alias_norm) + r"(?![a-z])"

            if re.search(pattern, text):
                return region

    return "Unknown"


def safe_filename(text):
    text = str(text).lower()
    text = text.replace("/", "_")
    text = text.replace(" ", "_")
    text = text.replace("'", "")
    text = re.sub(r"[^a-z0-9_]+", "", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


# ======================================================
# 4. Load and prepare CSV
# ======================================================

def prepare_plot_dataframe(csv_path):
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}. Run the RGB extraction script first."
        )

    df = pd.read_csv(csv_path)

    required_rgb_cols = ["ai_r", "ai_g", "ai_b"]
    missing_rgb_cols = [col for col in required_rgb_cols if col not in df.columns]

    if missing_rgb_cols:
        raise ValueError(f"Missing RGB columns in CSV: {missing_rgb_cols}")

    for col in required_rgb_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid_rgb = (
        df["ai_r"].notna()
        & df["ai_g"].notna()
        & df["ai_b"].notna()
    )

    df = df[valid_rgb].copy()

    if len(df) == 0:
        raise RuntimeError("No valid RGB rows found in the CSV.")

    rgb_values = df[["ai_r", "ai_g", "ai_b"]].to_numpy(dtype=float)

    need_lab = any(col not in df.columns for col in ["L", "a", "b", "chroma"])
    need_hex = "colour_hex" not in df.columns

    if need_lab:
        lab_values = np.vstack([
            rgb255_to_lab(rgb)
            for rgb in rgb_values
        ])

        df["L"] = lab_values[:, 0]
        df["a"] = lab_values[:, 1]
        df["b"] = lab_values[:, 2]
        df["chroma"] = np.sqrt(df["a"] ** 2 + df["b"] ** 2)

    if need_hex:
        df["colour_hex"] = [
            rgb_to_hex(rgb)
            for rgb in rgb_values
        ]

    for col in ["L", "a", "b", "chroma"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["L", "a", "b", "chroma", "colour_hex"]).copy()

    if len(df) == 0:
        raise RuntimeError("No valid rows left after preparing plot columns.")

    # Detect region
    if "region" not in df.columns:
        if "crop_path" in df.columns:
            region_source_col = "crop_path"
        elif "relative_folder" in df.columns:
            region_source_col = "relative_folder"
        elif "crop_folder" in df.columns:
            region_source_col = "crop_folder"
        else:
            raise ValueError(
                "Cannot detect region. CSV needs one of these columns: "
                "crop_path, relative_folder, crop_folder."
            )

        df["region"] = df[region_source_col].apply(detect_region_from_text)

    return df


def sample_for_plot(df):
    if MAX_PLOT_POINTS is not None and len(df) > MAX_PLOT_POINTS:
        return df.sample(
            n=MAX_PLOT_POINTS,
            random_state=RANDOM_SEED,
        ).copy()

    return df.copy()


# ======================================================
# 5. Plot 1: L* vs Chroma, true AI RGB colours
# ======================================================

def plot_l_chroma(df_plot, output_path, title_suffix=""):
    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    ax.set_facecolor("#222222")

    # Halo layer, helps black/dark points remain visible
    plt.scatter(
        df_plot["chroma"],
        df_plot["L"],
        s=34,
        c="lightgrey",
        alpha=0.30,
        edgecolors="none",
    )

    # True AI-estimated RGB colours
    plt.scatter(
        df_plot["chroma"],
        df_plot["L"],
        c=df_plot["colour_hex"],
        s=18,
        alpha=0.95,
        edgecolors="white",
        linewidths=0.25,
    )

    plt.axhline(30, linestyle="--", linewidth=1, color="white", alpha=0.7)
    plt.axhline(75, linestyle="--", linewidth=1, color="white", alpha=0.7)
    plt.axvline(15, linestyle="--", linewidth=1, color="white", alpha=0.7)

    plt.text(2, 12, "black / dark grey", fontsize=10, color="white")
    plt.text(2, 50, "grey", fontsize=10, color="white")
    plt.text(2, 88, "white / near-white", fontsize=10, color="white")
    plt.text(25, 82, "light coloured", fontsize=10, color="white")
    plt.text(25, 45, "coloured frames", fontsize=10, color="white")
    plt.text(25, 18, "dark coloured frames", fontsize=10, color="white")

    plt.xlabel("Chroma: colour strength")
    plt.ylabel("L*: lightness")
    plt.title(f"AI-estimated window-frame RGB colours in CIELAB space{title_suffix}")

    plt.xlim(left=0)
    plt.ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ======================================================
# 6. Plot 2: a* vs b*, true AI RGB colours
# ======================================================

def plot_ab(df_plot, output_path, title_suffix=""):
    plt.figure(figsize=(9, 7))
    ax = plt.gca()
    ax.set_facecolor("#222222")

    plt.scatter(
        df_plot["a"],
        df_plot["b"],
        c=df_plot["colour_hex"],
        s=20,
        alpha=0.95,
        edgecolors="white",
        linewidths=0.2,
    )

    plt.axhline(0, linestyle="--", linewidth=1, color="white", alpha=0.7)
    plt.axvline(0, linestyle="--", linewidth=1, color="white", alpha=0.7)

    plt.xlabel("a*: green-red axis")
    plt.ylabel("b*: blue-yellow axis")
    plt.title(f"AI-estimated a*-b* distribution using true RGB point colours{title_suffix}")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ======================================================
# 7. Main
# ======================================================

if __name__ == "__main__":
    df_valid = prepare_plot_dataframe(INPUT_CSV)
    df_plot = sample_for_plot(df_valid)

    print(f"Loaded valid RGB records: {len(df_valid)}")
    print(f"Plotting points in overall plots: {len(df_plot)}")

    # ==================================================
    # Overall plots
    # ==================================================

    plot_l_chroma(df_plot, OUTPUT_L_CHROMA_TRUE)
    print(f"Saved overall Plot 1 to: {OUTPUT_L_CHROMA_TRUE}")

    plot_ab(df_plot, OUTPUT_AB_TRUE)
    print(f"Saved overall Plot 2 to: {OUTPUT_AB_TRUE}")

    # ==================================================
    # Region counts
    # ==================================================

    region_counts = (
        df_valid["region"]
        .value_counts()
        .reindex(REGION_ORDER + ["Unknown"], fill_value=0)
        .reset_index()
    )

    region_counts.columns = ["region", "sample_count"]

    region_counts.to_csv(
        OUTPUT_REGION_COUNTS_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Saved region sample counts to: {OUTPUT_REGION_COUNTS_CSV}")
    print("\nRegion counts:")
    print(region_counts)

    # ==================================================
    # Region plots
    # ==================================================

    for region in REGION_ORDER + ["Unknown"]:
        df_region = df_valid[df_valid["region"] == region].copy()

        if len(df_region) == 0:
            continue

        df_region_plot = sample_for_plot(df_region)

        region_safe = safe_filename(region)

        output_l_chroma_region = OUTPUT_REGION_DIR / f"{region_safe}_L_chroma.png"
        output_ab_region = OUTPUT_REGION_DIR / f"{region_safe}_ab.png"

        title_suffix = f" - {region}"

        plot_l_chroma(
            df_region_plot,
            output_l_chroma_region,
            title_suffix=title_suffix,
        )

        plot_ab(
            df_region_plot,
            output_ab_region,
            title_suffix=title_suffix,
        )

        print(
            f"Saved region plots for {region}: "
            f"{output_l_chroma_region}, {output_ab_region}"
        )