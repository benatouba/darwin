from matplotlib.colors import LinearSegmentedColormap
basepath = "/home/ben/data/GAR/"
coordinates = {
    "minasrojas": (-0.618625, -90.3673),
    "militar": (-0.489962, -90.2808),
    "puertoayora": (-0.743708, -90.3027),
    "puertovillamil": (-0.946400, -90.9741),
    "puertobacceriomoreno": (-0.89515, -89.6068),
    "eljunco": (-0.893768, -89.4804),
    "lagalapaguera": (-0.91197, -89.4387),
    "cuevadesucre": (-0.843216, -91.0284),
    "negra": (-0.848344, -91.1312),
    "crocker": (-0.642398, -90.326),
    "rosa": (-0.65453, -90.4035),
    # "met-e_bellavista": (-0.692384, -90.3282),
    # "met-e_puerto_ayora": (-0.743708, -90.3027)
}

color_map = LinearSegmentedColormap.from_list(
    "temperature",
    ["white", "steelblue", "c", "khaki", "orange", "orangered", "r", "darkred"],
)
