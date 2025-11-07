"""Plot WRF domains with digital elevation model at each domain's resolution.

This module creates visualizations of WRF domains with DEM data at
appropriate resolutions for each nested domain. Uses GMTED2010 global
elevation data.
"""

import tempfile
import warnings
from pathlib import Path

import cartopy.feature as cfeature
import elevation
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import rasterio
import xarray as xr
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import ConnectionPatch
from pyproj import Proj, Transformer
from shapely.geometry import Polygon

# Suppress warnings
warnings.filterwarnings("ignore")


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
            params.setdefault("truelat2", float(values[0]))
        elif key == "STAND_LON":
            params["stand_lon"] = float(values[0])

    return params


def create_land_colormap():
    """Create a terrain colormap for land elevations only (no ocean blues).

    Returns a colormap with greens/browns for elevations, starting from
    0m.
    """
    # Define color transitions for land-only terrain
    # Based on natural terrain colors: lowlands (green) -> hills (yellow/brown) -> mountains (brown/gray/white)
    colors_list = [
        (0.0, "#2d5016"),  # Dark green (0m - sea level)
        (0.15, "#4a7c1f"),  # Green (low elevations)
        (0.30, "#6b9d3a"),  # Light green (plains)
        (0.45, "#a8b965"),  # Yellow-green (hills)
        (0.60, "#d4c482"),  # Tan (highlands)
        (0.75, "#8b6f47"),  # Brown (mountains)
        (0.85, "#6b5b4a"),  # Dark brown (high mountains)
        (0.95, "#9e9e9e"),  # Gray (peaks)
        (1.0, "#ffffff"),  # White (snow/ice)
    ]

    positions = [c[0] for c in colors_list]
    colors = [c[1] for c in colors_list]

    cmap = LinearSegmentedColormap.from_list("terrain_land", list(zip(positions, colors)), N=256)
    return cmap


def compute_domain_corners(params):
    """Compute corner coordinates for all domains."""
    # Create pyproj Proj object for coordinate transformations
    if params["map_proj"] == "LAMBERT":
        proj_str = (
            f"+proj=lcc +lat_1={params['truelat1']} +lat_2={params.get('truelat2', params['truelat1'])} "
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

    # Calculate actual grid resolution for this domain
    actual_dx = params["dx"]
    actual_dy = params["dy"]

    domains.append(
        {
            "corners_xy": (x_min, x_max, y_min, y_max),
            "corners_lonlat": (list(lon_corners), list(lat_corners)),
            "extent": [min(lon_corners), max(lon_corners), min(lat_corners), max(lat_corners)],
            "resolution_m": (actual_dx, actual_dy),
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

        # Calculate actual grid resolution
        actual_dx = parent_dom["resolution_m"][0] / ratio
        actual_dy = parent_dom["resolution_m"][1] / ratio

        domains.append(
            {
                "corners_xy": (child_x_min, child_x_max, child_y_min, child_y_max),
                "corners_lonlat": (list(lon_corners), list(lat_corners)),
                "extent": [
                    min(lon_corners),
                    max(lon_corners),
                    min(lat_corners),
                    max(lat_corners),
                ],
                "resolution_m": (actual_dx, actual_dy),
            }
        )

    return domains


def fetch_srtm_elevation(extent, cache_dir=None):
    """Fetch SRTM elevation data using the elevation package.

    Parameters
    ----------
    extent : list
        [lon_min, lon_max, lat_min, lat_max]
    cache_dir : str, optional
        Directory to cache downloaded DEM data

    Returns
    -------
    tuple
        (elevation_array, transform, crs) from rasterio
    """
    lon_min, lon_max, lat_min, lat_max = extent

    # Calculate extent area to determine if it's too large
    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min
    area_deg2 = lon_span * lat_span

    # If domain is very large (>100 deg²), use synthetic data instead
    # SRTM1 tiles are 1°x1°, so this prevents downloading 100s of tiles
    if area_deg2 > 100:
        print(f"  Domain too large for SRTM download ({area_deg2:.1f} deg²)")
        print("  Using synthetic elevation data...")
        return generate_synthetic_dem(extent)

    # Add buffer for better edge handling
    buffer = 0.1  # degrees
    bounds = (lon_min - buffer, lat_min - buffer, lon_max + buffer, lat_max + buffer)

    # Use temporary file for DEM
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_file:
        output_file = tmp_file.name

    try:
        print(f"  Downloading SRTM data for extent: {bounds}")

        # Download SRTM data (don't clean - preserve cache)
        elevation.clip(bounds=bounds, output=output_file)

        # Read with rasterio
        with rasterio.open(output_file) as src:
            elevation_data = src.read(1)
            transform = src.transform
            crs = src.crs

        # Clean up temporary file
        Path(output_file).unlink(missing_ok=True)

        return elevation_data, transform, crs

    except Exception as e:
        print(f"  Warning: Error fetching SRTM data: {e}")
        print("  Using synthetic elevation data...")

        # Clean up temporary file
        Path(output_file).unlink(missing_ok=True)

        return generate_synthetic_dem(extent)


def generate_synthetic_dem(extent):
    """Generate synthetic DEM data for demonstration.

    Parameters
    ----------
    extent : list
        [lon_min, lon_max, lat_min, lat_max]

    Returns
    -------
    tuple
        (elevation_array, transform, crs) from rasterio
    """
    lon_min, lon_max, lat_min, lat_max = extent

    # Create grid
    nx, ny = 200, 200
    lons = np.linspace(lon_min, lon_max, nx)
    lats = np.linspace(lat_min, lat_max, ny)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Generate realistic-looking synthetic topography using multiple frequencies
    elev_data = (
        800 * np.sin(lon_grid / 3) * np.cos(lat_grid / 3)
        + 400 * np.sin(lon_grid * 1.5) * np.cos(lat_grid * 1.5)
        + 200 * np.sin(lon_grid * 4) * np.cos(lat_grid * 4)
        + 500
    )

    # Add some random noise for texture
    np.random.seed(42)
    elev_data += np.random.normal(0, 50, elev_data.shape)

    # Create transform
    from rasterio.transform import from_bounds

    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, nx, ny)
    crs = rasterio.crs.CRS.from_epsg(4326)

    return elev_data, transform, crs


def fetch_gmted2010_data(extent, target_resolution_deg=0.125):
    """Fetch GMTED2010 data from local cache or TEMIS CloudFront.

    Parameters
    ----------
    extent : list
        [lon_min, lon_max, lat_min, lat_max]
    target_resolution_deg : float
        Desired resolution in degrees (0.0625, 0.125, 0.25, 0.5, 0.75, 1.0)

    Returns
    -------
    xarray.DataArray
        Elevation data with coordinates
    """
    lon_min, lon_max, lat_min, lat_max = extent

    # Map resolution to TEMIS filenames
    res_map = {
        0.0625: ("GMTED2010_15n015_00625deg.nc", "00625"),
        0.125: ("GMTED2010_15n030_0125deg.nc", "0125"),
        0.25: ("GMTED2010_15n060_0250deg.nc", "0250"),
        0.5: ("GMTED2010_15n120_0500deg.nc", "0500"),
        0.75: ("GMTED2010_15n180_0750deg.nc", "0750"),
        1.0: ("GMTED2010_15n240_1000deg.nc", "1000"),
    }

    # Find closest available resolution
    available = np.array(list(res_map.keys()))
    idx = np.argmin(np.abs(available - target_resolution_deg))
    selected_res = available[idx]
    filename, res_str = res_map[selected_res]

    # Check for local cached file first
    cache_dir = Path.home() / ".cache" / "gmted2010"
    local_file = cache_dir / filename

    if local_file.exists():
        print(f"  Using cached GMTED2010 at {selected_res}° resolution...")
        file_to_open = str(local_file)
    else:
        url = f"https://d1qb6yzwaaq4he.cloudfront.net/data/gmted2010/{filename}"
        print(f"  Fetching GMTED2010 at {selected_res}° resolution from TEMIS...")
        file_to_open = url

    try:
        # Open NetCDF file (local or remote)
        ds = xr.open_dataset(file_to_open, engine="netcdf4")

        # Extract elevation variable
        if "elevation" in ds:
            elev = ds["elevation"]
        elif "z" in ds:
            elev = ds["z"]
        else:
            # Try to find any data variable
            data_vars = list(ds.data_vars)
            if data_vars:
                elev = ds[data_vars[0]]
            else:
                raise ValueError("No data variables found in dataset")

        # Get coordinate dimensions
        dims = list(elev.dims)
        print(f"    Dimensions: {dims}")
        print(f"    Coordinates: {list(elev.coords)}")

        # Identify lon/lat coordinates
        lon_coord = None
        lat_coord = None
        for coord in elev.coords:
            coord_lower = coord.lower()
            if "lon" in coord_lower or coord == "x":
                lon_coord = coord
            elif "lat" in coord_lower or coord == "y":
                lat_coord = coord

        if not lon_coord or not lat_coord:
            raise ValueError(f"Cannot identify lat/lon coordinates: {list(elev.coords)}")

        print(f"    Using lon={lon_coord}, lat={lat_coord}")

        # Get coordinate values
        lons = elev[lon_coord].values
        lats = elev[lat_coord].values

        # Add buffer for interpolation
        buffer = 1.0  # degree

        # Find indices for subset (handle global wrap-around)
        lon_mask = (lons >= (lon_min - buffer)) & (lons <= (lon_max + buffer))
        lat_mask = (lats >= (lat_min - buffer)) & (lats <= (lat_max + buffer))

        # Extract subset
        elev_subset = elev.isel(
            {
                lon_coord: lon_mask,
                lat_coord: lat_mask,
            }
        )

        print(f"    Extracted subset: {elev_subset.shape}")

        return elev_subset

    except Exception as e:
        print(f"    Error: {e}")
        raise


def generate_simple_synthetic_dem(extent):
    """Generate simple synthetic DEM data with ocean and land.

    Parameters
    ----------
    extent : list
        [lon_min, lon_max, lat_min, lat_max]

    Returns
    -------
    tuple
        (elevation_array, transform, crs) from rasterio
    """
    lon_min, lon_max, lat_min, lat_max = extent

    # Create grid
    nx, ny = 200, 200
    lons = np.linspace(lon_min, lon_max, nx)
    lats = np.linspace(lat_min, lat_max, ny)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Generate simple topography
    # Base elevation centered around 0 (sea level)
    elev_data = (
        1200 * np.sin(lon_grid / 3) * np.cos(lat_grid / 3)
        + 600 * np.sin(lon_grid * 1.5) * np.cos(lat_grid * 1.5)
        + 300 * np.sin(lon_grid * 4) * np.cos(lat_grid * 4)
    )

    # Add some "land masses" - regions above sea level
    # Create a land mask based on distance from certain points
    land_center_lon = (lon_min + lon_max) / 2
    land_center_lat = (lat_min + lat_max) / 2

    # Distance from center
    dist_from_center = np.sqrt(
        (lon_grid - land_center_lon) ** 2 + (lat_grid - land_center_lat) ** 2
    )

    # Make center higher (land) and edges lower (ocean)
    land_pattern = 1500 * np.exp(-dist_from_center / 2)
    elev_data = elev_data + land_pattern

    # Add texture
    np.random.seed(42)
    elev_data += np.random.normal(0, 50, elev_data.shape)

    # Create transform
    from rasterio.transform import from_bounds

    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, nx, ny)
    crs = rasterio.crs.CRS.from_epsg(4326)

    return elev_data, transform, crs


def fetch_srtm_data_for_domain(extent, product="SRTM1"):
    """Fetch SRTM data for domains.

    Parameters
    ----------
    extent : list
        [lon_min, lon_max, lat_min, lat_max]
    product : str
        SRTM product: 'SRTM1' (30m, 1 arc-second) or 'SRTM3' (90m, 3 arc-second)

    Returns
    -------
    xarray.DataArray
        Elevation data with coordinates
    """
    lon_min, lon_max, lat_min, lat_max = extent

    # Calculate domain size
    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min
    area_deg2 = lon_span * lat_span

    # SRTM1 (30m): only for very small domains (< 2 deg²)
    # SRTM3 (90m): for medium domains (< 50 deg²)
    if product == "SRTM1" and area_deg2 > 2:
        raise ValueError(f"Domain too large for SRTM1 ({area_deg2:.1f} deg²), use SRTM3")
    if product == "SRTM3" and area_deg2 > 50:
        raise ValueError(f"Domain too large for SRTM3 ({area_deg2:.1f} deg²)")

    # Use temporary file for DEM
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_file:
        output_file = tmp_file.name

    try:
        resolution_str = "30m" if product == "SRTM1" else "90m"
        print(f"  Downloading SRTM {resolution_str} data...")

        # Add small buffer
        buffer = 0.05
        bounds = (lon_min - buffer, lat_min - buffer, lon_max + buffer, lat_max + buffer)

        # Download SRTM data
        elevation.clip(bounds=bounds, output=output_file, product=product)

        # Read with rasterio
        with rasterio.open(output_file) as src:
            elevation_data = src.read(1)
            bounds = src.bounds
            width = src.width
            height = src.height

            # Create coordinate arrays
            lons = np.linspace(bounds.left, bounds.right, width)
            lats = np.linspace(bounds.top, bounds.bottom, height)

        # Clean up
        Path(output_file).unlink(missing_ok=True)

        # Convert to xarray
        elev_xr = xr.DataArray(
            elevation_data,
            coords={"latitude": lats, "longitude": lons},
            dims=["latitude", "longitude"],
        )

        print(f"    Successfully fetched SRTM {resolution_str} data ({elevation_data.shape})")
        return elev_xr

    except Exception as e:
        print(f"    SRTM fetch failed: {e}")
        # Clean up
        Path(output_file).unlink(missing_ok=True)
        raise


def fetch_wrf_static_data(domain_name):
    """Fetch pre-processed WRF static height data.

    Parameters
    ----------
    domain_name : str
        Domain name: 'd10km' or 'd02km'

    Returns
    -------
    tuple
        (hgt_array, lon_2d, lat_2d) - elevation and coordinate grids
    """
    cache_dir = Path.home() / ".cache" / "gmted2010"
    filename = f"GAR_{domain_name}_static_hgt.nc"
    lmfilename = f"GAR_{domain_name}_static_landmask.nc"
    filepath = cache_dir / filename
    lmpath = cache_dir / lmfilename

    if not filepath.exists():
        raise FileNotFoundError(f"WRF static file not found: {filepath}")

    print(f"  Using WRF static file: {filename}")

    # Open dataset
    ds = xr.open_dataset(filepath)
    lm_ds = xr.open_dataset(lmpath)

    # Extract height (remove time dimension if present)
    if "hgt" in ds:
        hgt = ds["hgt"]
        if "time" in hgt.dims:
            hgt = hgt.isel(time=0)
    else:
        raise ValueError(f"No 'hgt' variable found in {filename}")

    if "landmask" in lm_ds:
        landmask = lm_ds["landmask"]
        if "time" in landmask.dims:
            landmask = landmask.isel(time=0)
    else:
        raise ValueError(f"No 'landmask' variable found in {lmfilename}")

    # Extract lon/lat coordinates
    if "lon" in ds and "lat" in ds:
        lon_2d = ds["lon"].values
        lat_2d = ds["lat"].values
    else:
        raise ValueError(f"No 'lon'/'lat' coordinates found in {filename}")

    hgt_array = hgt.values * landmask.values  # Mask ocean values

    print(f"    Loaded data: {hgt_array.shape}")
    print(f"    Elevation range: {np.nanmin(hgt_array):.1f} to {np.nanmax(hgt_array):.1f} m")

    ds.close()

    return hgt_array, lon_2d, lat_2d


def fetch_dem_data(extent, target_resolution_m, domain_idx=None):
    """Fetch DEM data from appropriate source based on resolution.

    Parameters
    ----------
    extent : list
        [lon_min, lon_max, lat_min, lat_max]
    target_resolution_m : float
        Target resolution in meters
    domain_idx : int, optional
        Domain index (0=d01, 1=d02, 2=d03) for WRF static files

    Returns
    -------
    xarray.DataArray or tuple
        Elevation data with coordinates, or (hgt, lon_2d, lat_2d) for WRF static
    """
    # For domains 1 and 2, try WRF static files first
    if domain_idx == 0:  # d01 (10 km)
        try:
            return fetch_wrf_static_data("d10km")
        except Exception as e:
            print(f"  WRF static unavailable: {e}")
            print("  Falling back to GMTED2010...")
    elif domain_idx == 1:  # d02 (2 km)
        try:
            return fetch_wrf_static_data("d02km")
        except Exception as e:
            print(f"  WRF static unavailable: {e}")
            print("  Falling back to SRTM3...")

    # Convert target resolution from meters to degrees (approximate at equator)
    # 1 degree ≈ 111 km
    target_resolution_deg = target_resolution_m / 111000.0

    # For all fine and medium resolution domains (< 5km), use SRTM 90m (SRTM3)
    if target_resolution_deg < 0.05:  # < ~5 km (e.g., 200m or 2km domains)
        try:
            return fetch_srtm_data_for_domain(extent, product="SRTM3")
        except Exception as e:
            print(f"  SRTM3 unavailable: {e}")
            print("  Using GMTED2010...")

    # Use finest GMTED2010 resolution (7 km / 0.0625°) for all coarse domains
    fetch_res = 0.0625  # ~7 km

    try:
        return fetch_gmted2010_data(extent, fetch_res)
    except Exception as e:
        print(f"  Failed to fetch GMTED2010: {e}")
        print("  Using synthetic data...")
        # Return as xarray for consistency
        elev_data, transform, crs = generate_simple_synthetic_dem(extent)
        # Convert to xarray
        lons = np.linspace(extent[0], extent[1], elev_data.shape[1])
        lats = np.linspace(extent[2], extent[3], elev_data.shape[0])
        return xr.DataArray(
            elev_data,
            coords={"latitude": lats, "longitude": lons},
            dims=["latitude", "longitude"],
        )


def resample_dem_to_domain(elev_xr, domain, target_resolution_m):
    """Resample DEM data to match domain grid resolution using xarray.

    Parameters
    ----------
    elev_xr : xarray.DataArray
        Elevation data with lat/lon coordinates
    domain : dict
        Domain information including extent
    target_resolution_m : tuple
        Target resolution in meters (dx, dy)

    Returns
    -------
    tuple
        (resampled_elevation_array, lon_grid, lat_grid)
    """
    extent = domain["extent"]  # [lon_min, lon_max, lat_min, lat_max]

    # Calculate target resolution in degrees (approximate at domain center)
    lat_center = (extent[2] + extent[3]) / 2
    # Account for longitude convergence with latitude
    # 1 degree longitude = 111 km * cos(lat)
    # 1 degree latitude = 111 km
    target_res_lon = target_resolution_m[0] / (111000 * np.cos(np.radians(lat_center)))
    target_res_lat = target_resolution_m[1] / 111000

    print(f"    Target resolution: {target_res_lon:.6f}° lon, {target_res_lat:.6f}° lat")

    # Create target grid
    n_lon = max(int((extent[1] - extent[0]) / target_res_lon), 10)
    n_lat = max(int((extent[3] - extent[2]) / target_res_lat), 10)

    lon_grid = np.linspace(extent[0], extent[1], n_lon)
    lat_grid = np.linspace(extent[2], extent[3], n_lat)

    print(f"    Interpolating to {n_lon} x {n_lat} grid")

    # Identify coordinate names in the xarray
    lon_coord = None
    lat_coord = None
    for coord in elev_xr.coords:
        coord_lower = coord.lower()
        if "lon" in coord_lower or coord == "x":
            lon_coord = coord
        elif "lat" in coord_lower or coord == "y":
            lat_coord = coord

    # Interpolate using xarray (which handles coordinates properly)
    try:
        elev_resampled = elev_xr.interp(
            {
                lon_coord: lon_grid,
                lat_coord: lat_grid,
            },
            method="linear",
        )

        # Convert to numpy array
        elev_array = elev_resampled.values

        print(f"    Resampled shape: {elev_array.shape}")
        print(f"    Elevation range: {np.nanmin(elev_array):.1f} to {np.nanmax(elev_array):.1f} m")

        return elev_array, lon_grid, lat_grid

    except Exception as e:
        print(f"    Interpolation error: {e}")
        # Fallback: create empty grid
        elev_array = np.zeros((n_lat, n_lon))
        return elev_array, lon_grid, lat_grid


def create_domain_geodataframes(domains):
    """Create GeoDataFrames for each domain."""
    gdfs = []

    for i, domain in enumerate(domains):
        lons, lats = domain["corners_lonlat"]
        # Create polygon from first 4 corners
        coords = list(zip(lons[:4], lats[:4]))
        polygon = Polygon(coords)

        gdf = gpd.GeoDataFrame(
            {"domain": [f"d{i + 1:02d}"], "geometry": [polygon]},
            crs="EPSG:4326",
        )
        gdfs.append(gdf)

    return gdfs


def format_lon(x, pos=None):
    """Format longitude values for axis labels."""
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    lon, _ = transformer.transform(x, 0)
    lon = round(lon, 2)
    if lon >= 0:
        return f"{lon}°E"
    return f"{-lon}°W"


def format_lat(y, pos=None):
    """Format latitude values for axis labels."""
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    _, lat = transformer.transform(0, y)
    lat = round(lat, 2)
    if lat >= 0:
        return f"{lat}°N"
    return f"{-lat}°S"


def plot_wrf_domains_with_dem(
    fpath: str,
    output_file: str = "wrf_domains_dem.png",
):
    """Plot WRF domains with DEM at appropriate resolution for each domain.

    Parameters
    ----------
    fpath : str
        Path to WPS namelist file
    output_file : str
        Output filename for plot
    """
    params = parse_wps_namelist(fpath)
    domains = compute_domain_corners(params)
    gdfs = create_domain_geodataframes(domains)

    n_domains = len(domains)
    fig, axes = plt.subplots(1, n_domains, figsize=(6 * n_domains, 7))

    if n_domains == 1:
        axes = [axes]

    print(f"\nProcessing {n_domains} domains...")

    # Create land-only colormap
    land_cmap = create_land_colormap()

    # Plot each domain
    for i, (domain, gdf, ax) in enumerate(zip(domains, gdfs, axes, strict=True)):
        print(f"\nDomain d{i + 1:02d}:")
        print(f"  Resolution: {domain['resolution_m'][0] / 1000:.1f} km")
        print(f"  Extent: {domain['extent']}")

        # Fetch DEM data (WRF static for d01/d02, SRTM3 for d03)
        dem_result = fetch_dem_data(domain["extent"], domain["resolution_m"][0], domain_idx=i)

        # Check if we got WRF static data (tuple) or regular DEM (xarray)
        if isinstance(dem_result, tuple):
            # WRF static data - already in correct grid, no interpolation needed
            dem_interp, lon_2d, lat_2d = dem_result
            print(f"    WRF static data: {dem_interp.shape}")
        else:
            # Regular DEM data - needs interpolation
            elev_xr = dem_result
            dem_interp, lon_grid, lat_grid = resample_dem_to_domain(
                elev_xr, domain, domain["resolution_m"]
            )
            # Create meshgrid for plotting
            lon_2d, lat_2d = np.meshgrid(lon_grid, lat_grid)
            print(f"    Interpolated data: {dem_interp.shape}")

        # Set all values <= 0 to NaN (mask ocean/below sea level)
        dem_interp = np.where(dem_interp <= 0, np.nan, dem_interp)

        print(f"    Elevation range: {np.nanmin(dem_interp):.1f} to {np.nanmax(dem_interp):.1f} m")

        # Convert to Web Mercator for plotting
        gdf_wm = gdf.to_crs(epsg=3857)

        # Transform to Web Mercator
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        x_wm, y_wm = transformer.transform(lon_2d, lat_2d)

        # Get elevation range (only land values, NaN excluded)
        elev_min = np.nanmin(dem_interp)
        elev_max = np.nanmax(dem_interp)

        # **KEY FIX**: Normalize starting from 0 (sea level) for land-only colormap
        norm = colors.Normalize(vmin=0, vmax=elev_max)

        # Plot elevation data with land-only colormap
        im = ax.pcolormesh(
            x_wm,
            y_wm,
            dem_interp,
            cmap=land_cmap,
            norm=norm,
            shading="auto",
            zorder=1,
        )

        # Add geographic features using cartopy
        coastlines = cfeature.NaturalEarthFeature(
            "physical", "coastline", "10m", edgecolor="black", facecolor="none", linewidth=0.8
        )

        borders = cfeature.NaturalEarthFeature(
            "cultural",
            "admin_0_boundary_lines_land",
            "10m",
            edgecolor="gray",
            facecolor="none",
            linewidth=0.5,
            linestyle=":",
        )

        rivers = cfeature.NaturalEarthFeature(
            "physical",
            "rivers_lake_centerlines",
            "10m",
            edgecolor="blue",
            facecolor="none",
            linewidth=0.4,
            alpha=0.6,
        )

        # Add features by converting their geometries to Web Mercator
        for geom in coastlines.geometries():
            if geom.intersects(gdf.geometry.values[0]):
                gdf_coast = gpd.GeoDataFrame({"geometry": [geom]}, crs="EPSG:4326")
                gdf_coast_wm = gdf_coast.to_crs(epsg=3857)
                gdf_coast_wm.plot(ax=ax, edgecolor="black", linewidth=0.8, zorder=5)

        for geom in borders.geometries():
            if geom.intersects(gdf.geometry.values[0]):
                gdf_border = gpd.GeoDataFrame({"geometry": [geom]}, crs="EPSG:4326")
                gdf_border_wm = gdf_border.to_crs(epsg=3857)
                gdf_border_wm.plot(
                    ax=ax, edgecolor="gray", linewidth=0.5, linestyle=":", alpha=0.7, zorder=5
                )

        for geom in rivers.geometries():
            if geom.intersects(gdf.geometry.values[0]):
                gdf_river = gpd.GeoDataFrame({"geometry": [geom]}, crs="EPSG:4326")
                gdf_river_wm = gdf_river.to_crs(epsg=3857)
                gdf_river_wm.plot(ax=ax, edgecolor="#4169E1", linewidth=0.4, alpha=0.6, zorder=5)

        # Plot domain boundary
        gdf_wm.plot(
            ax=ax,
            facecolor="none",
            edgecolor="black",
            linewidth=2.5,
            alpha=0.9,
            zorder=10,
        )

        # Set extent
        ax.set_xlim(gdf_wm.total_bounds[0], gdf_wm.total_bounds[2])
        ax.set_ylim(gdf_wm.total_bounds[1], gdf_wm.total_bounds[3])

        # Add domain label
        ax.text(
            0.05,
            0.95,
            f"d{i + 1:02d}\n{domain['resolution_m'][0] / 1000:.1f} km",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
            zorder=11,
        )

        # Format axes
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5, zorder=5)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_lon))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_lat))
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
        plt.setp(ax.get_xticklabels(), fontsize=9)
        plt.setp(ax.get_yticklabels(), fontsize=9)

        # Add colorbar
        cbar = plt.colorbar(
            im,
            ax=ax,
            orientation="horizontal",
            pad=0.08,
            shrink=0.8,
            label="Elevation (m)",
        )
        cbar.ax.tick_params(labelsize=8)

    for i in range(1, n_domains):
        parent_idx = params["parent_id"][i] - 1
        parent_ax = axes[parent_idx]
        child_ax = axes[i]

        child_gdf_wm = gdfs[i].to_crs(epsg=3857)

        # Plot child domain box on parent
        child_gdf_wm.plot(
            ax=parent_ax,
            facecolor="gray",
            edgecolor="black",
            alpha=0.4,
            linewidth=1.5,
            zorder=9,
        )

        # Get child domain corners in Web Mercator
        bounds = child_gdf_wm.total_bounds  # [xmin, ymin, xmax, ymax]

        # Define corner points
        upper_left_parent = (bounds[0], bounds[3])  # (xmin, ymax)
        upper_right_parent = (bounds[2], bounds[3])  # (xmax, ymax)

        # Child subplot corners (in axes coordinates)
        upper_left_child = (0, 1)  # top-left of child subplot
        upper_right_child = (1, 1)  # top-right of child subplot

        # Create connection lines from parent domain box to child subplot
        # Upper left connection
        con_ul = ConnectionPatch(
            xyA=upper_left_child,
            coordsA="axes fraction",
            xyB=upper_left_parent,
            coordsB="data",
            axesA=child_ax,
            axesB=parent_ax,
            color="darkgray",
            linewidth=1.2,
            linestyle="--",
            zorder=12,
        )
        fig.add_artist(con_ul)

        # Upper right connection
        con_ur = ConnectionPatch(
            xyA=upper_right_child,
            coordsA="axes fraction",
            xyB=upper_right_parent,
            coordsB="data",
            axesA=child_ax,
            axesB=parent_ax,
            color="darkgray",
            linewidth=1.2,
            linestyle="--",
            zorder=12,
        )
        fig.add_artist(con_ur)

    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to {output_file}")


if __name__ == "__main__":
    # Run the visualization
    fpath = "./wps_namelist_template"

    # Plot with colorbar
    plot_wrf_domains_with_dem(fpath, output_file="wrf_domains_dem.png")
