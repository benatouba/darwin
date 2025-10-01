"""Default values for DARWIN."""

from __future__ import annotations

from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap
from pint import Unit
from datetime import datetime
import pandas as pd

basepath: Path = Path.home() / "data"
gar_path: Path = basepath / "GAR"
mm_path: Path = gar_path / "MM"

measured_path: Path = basepath / "darwin_measured"
start_year: int = 1980
default_start = datetime(1980, 1, 1)  # noqa: DTZ001
default_end = datetime(2023, 12, 31)  # noqa: DTZ001

# constants
pvalue_threshold: float = 0.05
ndims: dict[str, int] = {"2d": 2, "3d_press": 3, "3d": 3}
nlevels: dict[str, int] = {"1": 1, "2": 2, "3": 3}
bar_vars: list[str] = [
    "prcp",
    "et",
    "wateravailability",
    "netprcp",
]  # wateravailability is net precipitation

# Coordinates
coordinates: dict[str, tuple[float, float]] = {
    "Cerro Crocker": (-0.642398, -90.326),
    "Cueva de Sucre": (-0.843216, -91.0284),
    "El Junco": (-0.893768, -89.4804),
    "La Galapaguera": (-0.91197, -89.4387),
    "Militar": (-0.489962, -90.2808),
    "Minas Rojas": (-0.618625, -90.3673),
    "Puerto Ayora": (-0.743708, -90.3027),
    "Puerto Baquerizo Moreno": (-0.89515, -89.6068),
    "Puerto Villamil": (-0.946400, -90.9741),
    "Santa Rosa": (-0.65453, -90.4035),
    "Sierra Negra": (-0.848344, -91.1312),
    # "met-e_bellavista": (-0.692384, -90.3282),
    "Bellavista": (-0.696292, -90.327),
    # "met-e_puerto_ayora": (-0.743708, -90.3027)
}

# Variables
measured_vars: dict[str, list[str] | str] = {
    "prcp": [
        "PCP_tot_bucket",
        "PCP_diff_radar",
        "PCP_diffmin_radar",
        "PCP_acoustic",
    ],
    "t2": "T",
    "ws10": ["WS", "WSmax"],
    "wd10": ["WD"],
    "v10": ["WD"],
    "u10": ["WD"],
    "rh2": ["RH"],
    "swdown": ["SLR"],
    "psfc": ["Pabs"],
    "sh2": ["SH"],
    "q2": ["Q"],
}

# Units
measured_units: dict[str, Unit] = {
    "PCP_tot_bucket": Unit("mm"),
    "PCP_diff_radar": Unit("mm"),
    "PCP_diffmin_radar": Unit("mm/min"),
    "PCP_acoustic": Unit("mm"),
    "T": Unit("K"),
    "RH": Unit("%"),
    "Pabs": Unit("hPa"),
    "WS": Unit("m/s"),
    "WSmax": Unit("m/s"),
    "WD": Unit("degree"),
    "FOG": Unit("g"),
    "SLR": Unit("W/m**2"),
    "ST": Unit("K"),
    "Vwc": Unit("count"),
    "SH": Unit("g/kg"),
    "Q": Unit("kg/kg"),
}

fixed_units: dict[str, str] = {
    "k": "K",
    "w m-2": "W/m**2",
    "pa": "Pa",
}

# Seasons
season_names: list[str] = ["JFM", "AMJ", "JAS", "OND"]
season_map: dict[int, tuple[int, str]] = {
    1: (1, season_names[0]),
    2: (1, season_names[0]),
    3: (1, season_names[0]),
    4: (2, season_names[1]),
    5: (2, season_names[1]),
    6: (2, season_names[1]),
    7: (3, season_names[2]),
    8: (3, season_names[2]),
    9: (3, season_names[2]),
    10: (4, season_names[3]),
    11: (4, season_names[3]),
    12: (4, season_names[3]),
}
dry_wet_names: list[str] = ["Wet", "Dry"]
dry_wet_map: dict[int, tuple[int, str]] = {
    1: (1, dry_wet_names[0]),
    2: (1, dry_wet_names[0]),
    3: (1, dry_wet_names[0]),
    4: (1, dry_wet_names[0]),
    5: (1, dry_wet_names[0]),
    6: (2, dry_wet_names[0]),
    7: (2, dry_wet_names[1]),
    8: (2, dry_wet_names[1]),
    9: (2, dry_wet_names[1]),
    10: (2, dry_wet_names[1]),
    11: (2, dry_wet_names[1]),
    12: (1, dry_wet_names[1]),
}


# Colors
color_map: LinearSegmentedColormap = LinearSegmentedColormap.from_list(
    "temperature",
    ["white", "steelblue", "c", "khaki", "orange", "orangered", "r", "darkred"],
)
