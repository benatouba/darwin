import os
import warnings

import contextily as cx
import dotenv
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pyproj import Proj, Transformer
from shapely.geometry import Polygon

# Suppress warnings
warnings.filterwarnings("ignore")
loaded = dotenv.load_dotenv()


def parse_wps_namelist(fpath):
    """Parse WPS namelist file to extract domain parameters."""
    with open(fpath) as f:
        lines = f.readlines()

    params = {}
    for line in lines:
        s = line.split("=")
        if len(s) < 2:
            continue
        key = s[0].strip().upper()
        values = list(filter(None, s[1].strip().replace("\n", "").replace("'", "").split(",")))

        if key == "PARENT_ID":
            params["parent_id"] = [int(v) for v in values]
        elif key == "PARENT_GRID_RATIO":
            params["parent_ratio"] = [int(v) for v in values]
        elif key == "I_PARENT_START":
            params["i_parent_start"] = [int(v) for v in values]
        elif key == "J_PARENT_START":
            params["j_parent_start"] = [int(v) for v in values]
        elif key == "E_WE":
            params["e_we"] = [int(v) for v in values]
        elif key == "E_SN":
            params["e_sn"] = [int(v) for v in values]
        elif key == "DX":
            params["dx"] = float(values[0])
        elif key == "DY":
            params["dy"] = float(values[0])
        elif key == "MAP_PROJ":
            params["map_proj"] = values[0].strip().upper()
        elif key == "REF_LAT":
            params["ref_lat"] = float(values[0])
        elif key == "REF_LON":
            params["ref_lon"] = float(values[0])
        elif key == "TRUELAT1":
            params["truelat1"] = float(values[0])
        elif key == "TRUELAT2":
            params["truelat2"] = float(values[0])
        elif key == "STAND_LON":
            params["stand_lon"] = float(values[0])

    return params


def compute_domain_corners(params):
    """Compute corner coordinates for all domains."""
    # Create pyproj Proj object for coordinate transformations
    if params["map_proj"] == "LAMBERT":
        proj_str = (
            f"+proj=lcc +lat_1={params['truelat1']} +lat_2={params['truelat2']} "
            f"+lat_0={params['ref_lat']} +lon_0={params['stand_lon']} "
            f"+x_0=0 +y_0=0 +a=6370000 +b=6370000"
        )
    elif params["map_proj"] == "MERCATOR":
        proj_str = (
            f"+proj=merc +lat_ts={params['truelat1']} +lon_0={params['stand_lon']} "
            f"+x_0=0 +y_0=0 +a=6370000 +b=6370000"
        )
    else:  # POLAR
        proj_str = (
            f"+proj=stere +lat_ts={params['truelat1']} +lat_0=90.0 +lon_0={params['stand_lon']} "
            f"+x_0=0 +y_0=0 +a=6370000 +b=6370000"
        )

    p = Proj(proj_str)

    # Parent domain (d01)
    e_center, n_center = p(params["ref_lon"], params["ref_lat"])
    nx, ny = params["e_we"][0] - 1, params["e_sn"][0] - 1
    dx, dy = params["dx"], params["dy"]

    half_w = nx * dx / 2.0
    half_h = ny * dy / 2.0

    x_min = e_center - half_w
    x_max = e_center + half_w
    y_min = n_center - half_h
    y_max = n_center + half_h

    domains = []

    # Convert to lon/lat for parent domain
    lons = [x_min, x_max, x_max, x_min, x_min]
    lats = [y_min, y_min, y_max, y_max, y_min]
    lon_corners, lat_corners = p(lons, lats, inverse=True)

    domains.append(
        {
            "corners_xy": (x_min, x_max, y_min, y_max),
            "corners_lonlat": (list(lon_corners), list(lat_corners)),
            "extent": [min(lon_corners), max(lon_corners), min(lat_corners), max(lat_corners)],
        }
    )

    # Child domains
    for i in range(1, len(params["e_we"])):
        ratio = params["parent_ratio"][i]
        ips = params["i_parent_start"][i] - 1
        jps = params["j_parent_start"][i] - 1

        parent_dom = domains[params["parent_id"][i] - 1]
        parent_x_min, parent_x_max, parent_y_min, parent_y_max = parent_dom["corners_xy"]
        parent_nx = params["e_we"][params["parent_id"][i] - 1] - 1
        parent_ny = params["e_sn"][params["parent_id"][i] - 1] - 1
        parent_dx = (parent_x_max - parent_x_min) / parent_nx
        parent_dy = (parent_y_max - parent_y_min) / parent_ny

        child_x_min = parent_x_min + ips * parent_dx
        child_y_min = parent_y_min + jps * parent_dy

        child_nx = params["e_we"][i] - 1
        child_ny = params["e_sn"][i] - 1
        child_dx = parent_dx / ratio
        child_dy = parent_dy / ratio

        child_x_max = child_x_min + child_nx * child_dx
        child_y_max = child_y_min + child_ny * child_dy

        # Convert to lon/lat
        lons = [child_x_min, child_x_max, child_x_max, child_x_min, child_x_min]
        lats = [child_y_min, child_y_min, child_y_max, child_y_max, child_y_min]
        lon_corners, lat_corners = p(lons, lats, inverse=True)

        domains.append(
            {
                "corners_xy": (child_x_min, child_x_max, child_y_min, child_y_max),
                "corners_lonlat": (list(lon_corners), list(lat_corners)),
                "extent": [min(lon_corners), max(lon_corners), min(lat_corners), max(lat_corners)],
            }
        )

    return domains


def create_domain_geodataframes(domains):
    """Create GeoDataFrames for each domain."""
    gdfs = []

    for i, domain in enumerate(domains):
        lons, lats = domain["corners_lonlat"]
        # Create polygon from first 4 corners (5th is duplicate for closing)
        coords = list(zip(lons[:4], lats[:4]))
        polygon = Polygon(coords)

        gdf = gpd.GeoDataFrame(
            {"domain": [f"d{i + 1:02d}"], "geometry": [polygon]},
            crs="EPSG:4326",  # WGS84 lon/lat
        )
        gdfs.append(gdf)

    return gdfs


def format_lon(x, pos=None):
    """Format longitude values for axis labels."""
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    lon, _ = transformer.transform(x, 0)
    # round to 2 decimal places
    lon = round(lon, 2)
    if lon >= 0:
        return f"{lon}°E"
    return f"{-lon}°W"


def format_lat(y, pos=None):
    """Format latitude values for axis labels."""
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    _, lat = transformer.transform(0, y)
    # round to 2 decimal places
    lat = round(lat, 2)
    if lat >= 0:
        return f"{lat}°N"
    return f"{-lat}°S"


def plot_wrf_domains(
    fpath: str,
    output_file: str = "test.png",
    *,
    basemap_source: str = "OpenStreetMap.Mapnik",
    skip_basemap: bool = False,
    zoom_auto: bool = True,
):
    """Plot WRF domains side by side with basemap using geopandas and
    contextily.

    Parameters
    ----------
    fpath : str
        Path to WPS namelist file
    output_file : str
        Output filename for plot
    basemap_source : str
        Contextily basemap provider (e.g., 'OpenStreetMap.Mapnik', 'OpenTopoMap')
        See: cx.providers for options
    skip_basemap : bool
        If True, skip basemap and only show domain boundaries
    zoom_auto : bool
        If True, auto-calculate zoom level. If False, use contextilys default
    """
    params = parse_wps_namelist(fpath)
    domains = compute_domain_corners(params)
    gdfs = create_domain_geodataframes(domains)

    n_domains = len(domains)
    _, axes = plt.subplots(1, n_domains, figsize=(5 * n_domains, 7))

    # Handle single domain case
    if n_domains == 1:
        axes = [axes]

    # Plot each domain
    for i, (domain, gdf, ax) in enumerate(zip(domains, gdfs, axes, strict=True)):
        # Convert to Web Mercator for contextily
        gdf_wm = gdf.to_crs(epsg=3857)

        # Plot domain boundary
        gdf_wm.plot(ax=ax, facecolor="none", linewidth=2, alpha=0.8)

        # Add basemap if requested
        if not skip_basemap:
            # Calculate zoom level based on extent
            if zoom_auto:
                lon_span = domain["extent"][1] - domain["extent"][0]
                zoom = int(np.clip(6 + np.log2(max(360.0 / max(lon_span, 1e-6), 1.0)), 1, 14))
            else:
                zoom = "auto"

            # Access provider
            provider = eval(f"cx.providers.{basemap_source}")

            if provider["name"].startswith("Stadia"):
                STADIA_API_KEY = os.getenv("STADIA_API_KEY")
                if STADIA_API_KEY is None:
                    raise ValueError(
                        "STADIA_API_KEY not found in environment variables. "
                        "Please set it to use Stadia Maps."
                    )
                # Update URL with API key
                provider = provider.copy()
                provider["url"] += f"?api_key={STADIA_API_KEY}"
            cx.add_basemap(ax, source=provider, zoom=zoom, attribution=False)

        # Set extent using Web Mercator coordinates
        ax.set_xlim(gdf_wm.total_bounds[0], gdf_wm.total_bounds[2])
        ax.set_ylim(gdf_wm.total_bounds[1], gdf_wm.total_bounds[3])

        # Add domain label
        ax.text(
            0.05,
            0.95,
            f"d{i + 1:02d}",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
            zorder=10,
        )

        # Add gridlines (manual implementation for Web Mercator)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        # ax.set_xlabel("Easting (m)", fontsize=10)
        # ax.set_ylabel("Northing (m)", fontsize=10)

        # Format x-axis (longitude) and y-axis (latitude) labels
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_lon))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_lat))

        ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        # Turn off axis for cleaner look (optional)
        # ax.set_axis_off()

    # Add child domain shading on parent plots
    for i in range(1, n_domains):
        parent_idx = params["parent_id"][i] - 1
        parent_ax = axes[parent_idx]

        # Get child domain polygon in Web Mercator
        child_gdf_wm = gdfs[i].to_crs(epsg=3857)

        # Plot child footprint on parent with semi-transparent fill
        child_gdf_wm.plot(
            ax=parent_ax, facecolor="grey", edgecolor="darkgrey", alpha=0.4, linewidth=1, zorder=8
        )

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output_file}")


if __name__ == "__main__":
    # Run the visualization
    fpath = "./wps_namelist_template"

    # Option 3: Without basemap (offline mode)
    plot_wrf_domains(
        fpath,
        output_file="wrf_domains_no_basemap.png",
        skip_basemap=True
    )

    # # Option 1: With OpenStreetMap basemap (free, no API key needed)
    plot_wrf_domains(
        fpath, output_file="wrf_domains_osm.png", basemap_source="OpenStreetMap.Mapnik"
    )

    # # Option 2: With OpenTopoMap (good for topography, free)
    plot_wrf_domains(
        fpath,
        output_file="wrf_domains_topo.png",
        basemap_source="OpenTopoMap"
    )

    # Option with Stadia Maps (requires API key)
    plot_wrf_domains(
        "./wps_namelist_template",
        output_file="wrf_stadia_outdoors.png",
        basemap_source="Stadia.Outdoors",
        # basemap_source=f"https://tiles.stadiamaps.com/tiles/outdoors/{{z}}/{{x}}/{{y}}.png?api_key={STADIA_API_KEY}"
    )

    # To see all available providers:
    # import contextily as cx
    # print(cx.providers.keys())
