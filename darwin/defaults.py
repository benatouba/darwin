"""Default values for DARWIN."""

from __future__ import annotations

from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap
from pint import Unit

basepath: Path = Path("/home/ben/data/GAR/")
coordinates = {
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

measured_vars = {
    "prcp": [
        "PCP_tot_bucket",
        "PCP_diff_radar",
        "PCP_diffmin_radar",
        "PCP_acoustic",
    ],
    "t2": "T",
    "ws10": ["WS", "WSmax"],
    "wd10": ["WD"],
    # "v10": ["WD"],
    # "u10": ["WD"],
    "rh2": ["RH"],
    "swdown": ["SLR"],
    "psfc": ["Pabs"],
}

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
}

color_map: LinearSegmentedColormap = LinearSegmentedColormap.from_list(
    "temperature",
    ["white", "steelblue", "c", "khaki", "orange", "orangered", "r", "darkred"],
)