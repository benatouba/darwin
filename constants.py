from matplotlib.colors import LinearSegmentedColormap
from pint import Unit
import cf_xarray.units

basepath = "/home/ben/data/GAR/"
coordinates = {
    "rojas": (-0.618625, -90.3673),
    "militar": (-0.489962, -90.2808),
    "puertoayora": (-0.743708, -90.3027),
    "puertovillamil": (-0.946400, -90.9741),
    "puertobaccerizomoreno": (-0.89515, -89.6068),
    "eljunco": (-0.893768, -89.4804),
    "lagalapaguera": (-0.91197, -89.4387),
    "cuevadesucre": (-0.843216, -91.0284),
    "negra": (-0.848344, -91.1312),
    "crocker": (-0.642398, -90.326),
    "rosa": (-0.65453, -90.4035),
    # "met-e_bellavista": (-0.692384, -90.3282),
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
    "v10": ["WD"],
    "u10": ["WD"],
    "rh": ["RH"],
}

measured_units = {
    "PCP_tot_bucket": Unit("mm"),
    "PCP_diff_radar": Unit("mm"),
    "PCP_diffmin_radar": Unit("mm/min"),
    "PCP_acoustic": Unit("mm"),
    "T": Unit("C"),
    "RH": Unit("%"),
    "Pabs": Unit("hPa"),
    "WS": Unit("m/s"),
    "WSmax": Unit("m/s"),
    "WD": Unit("degree"),
    "FOG": Unit("g"),
    "SLR": Unit("W/m2"),
    "ST": Unit("C"),
    "Vwc": Unit("count"),
}

color_map = LinearSegmentedColormap.from_list(
    "temperature",
    ["white", "steelblue", "c", "khaki", "orange", "orangered", "r", "darkred"],
)
