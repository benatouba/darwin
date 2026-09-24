"""Shared fixtures for the publisher test seams."""

import pytest

AGGREGATION_MANIFEST_TEXT = """\
name: d02km-aggregation-contract
release: v1
products:
  - product_id: GAR_d02km_m_2d_prcp_2020-01
    kind: slice
    aggregations:
      - method: sum
        interval: daily
        missing_data: require-complete
      - method: mean
        interval: monthly
        missing_data: exclude-missing
  - product_id: GAR_d02km_static_landmask
    kind: reference
"""


@pytest.fixture
def aggregation_manifest_path(tmp_path):
    path = tmp_path / "manifest.yaml"
    path.write_text(AGGREGATION_MANIFEST_TEXT)
    return path
