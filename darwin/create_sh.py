"""Create the specific humidity data for the Darwin Dashboard."""

from __future__ import annotations

from argparse import ArgumentParser

import xarray as xr
from defaults import gar_path
from metpy.calc import specific_humidity_from_mixing_ratio
from metpy.units import units

parser = ArgumentParser()
parser.add_argument(
    "-f", "--frequency", choices=("y", "m", "d", "h"), type=str, help="The frequency of the data"
)
args = parser.parse_args()
frequency = args.frequency

pabs = xr.open_dataset(gar_path / "MM" / f"MM_d02km_{frequency}_2d_psfc.nc")
temp = xr.open_dataset(gar_path / "MM" / f"MM_d02km_{frequency}_2d_t2.nc")
rh = xr.open_dataset(gar_path / "MM" / f"MM_d02km_{frequency}_2d_rh2.nc")
q = xr.open_dataset(gar_path / "MM" / f"MM_d02km_{frequency}_2d_q2.nc")
sh = (
    specific_humidity_from_mixing_ratio(mixing_ratio=q["q2"].to_numpy() * units("kg/kg"))
    .to("g/kg")
    .magnitude
)
# sh = mixing_ratio_from_relative_humidity(
#     pressure=pabs["psfc"].to_numpy() * units.Pa,
#     temperature=temp["t2"].to_numpy() * units.K,
#     relative_humidity=rh["rh2"].to_numpy() * units.percent,
# ).to("g/kg").magnitude
pabs["psfc"].values = sh
pabs = pabs.rename_vars({"psfc": "sh2"})
pabs.attrs["VARNAME"] = "sh2"
pabs["sh2"].attrs["long_name"] = "Specific Humidity"
pabs["sh2"].attrs["units"] = "g/kg"
pabs.to_netcdf(gar_path / "MM" / f"MM_d02km_{frequency}_2d_sh2.nc")
