"""Historical smoke inputs, now preserved as separate native-cadence datasets.

The 2020 monthly model and 2025 observations are NOT a valid paired comparison.
Building timestamped stores is not acceptance, publication or a scientific claim.
Naive observation timestamps require a reviewed timezone declaration from the caller.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from darwin.publisher.station_series import MISSING_TOKENS as MISSING_TOKENS
from darwin.publisher.station_series import _station_point as _station_point
from darwin.publisher.station_series import build_station_series

GAR_SOURCE = "GAR_d02km_m_2d_prcp_2020.nc"
AWS_SOURCE = "data/11_AWS-PCerroCrocker_data.csv"
AWS_META = "data/11_AWS-PCerroCrocker_meta.xml"
GAR_STORE = "GAR-CerroCrocker-sampled_2020.zarr"
AWS_STORE = "AWS-CerroCrocker_2025-01.zarr"
AWS_YEAR = 2025
AWS_MONTH = 1


def build_comparison_series(
    repo_root: str | Path,
    out_dir: str | Path,
    *,
    obs_timezone: str | None = None,
) -> dict[str, Path]:
    """Prepare native series, preserving the legacy increment-sum policy explicitly.

    Callers must review input hashes before invoking this legacy convenience wrapper;
    unlike the general builder, this historical entry point has no pinned hash inputs.
    Existing output stores are refused rather than relabelled or overwritten.
    """
    root = Path(repo_root)
    result = build_station_series(
        root,
        out_dir,
        gar_source=GAR_SOURCE,
        gar_variable="prcp",
        obs_csv=AWS_SOURCE,
        obs_meta=AWS_META,
        obs_year=AWS_YEAR,
        obs_month=AWS_MONTH,
        gar_store_name=GAR_STORE,
        obs_store_name=AWS_STORE,
        mode="sum",
        gar_sha256=hashlib.sha256((root / GAR_SOURCE).read_bytes()).hexdigest(),
        obs_sha256=hashlib.sha256((root / AWS_SOURCE).read_bytes()).hexdigest(),
        obs_timezone=obs_timezone,
        precipitation_kind="increments",
    )
    return {"gar_sampled": result["gar_sampled"], "aws_january": result["obs"]}
