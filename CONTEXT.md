# GAR Climate Data

This repository processes and visualizes the Galapagos Archipelago Refined analysis (GAR), a WRF-derived climate dataset. Its products support reproducible scientific exploration of gridded climate data and comparisons with observations.

## Language

**Climate normal**:
A monthly climatological reference product calculated over an inclusive reference period. GAR's default reference period is 1991-2020.
_Avoid_: average weather, baseline climate

**Normal product**:
A published GAR climate-normal layer: monthly mean daily maximum or minimum 2 m temperature, or mean monthly total precipitation. It includes its period, units, aggregation method, and provenance.
_Avoid_: normal, summary layer

**GAR product**:
A versioned spatial dataset derived from GAR source data and described by its domain, temporal resolution, variable, coverage, coordinates, metadata, and provenance.
_Avoid_: file, output

**Product candidate**:
A validated GAR product version awaiting explicit scientific approval before publication.
_Avoid_: draft output

**Native GAR projection**:
The spatial reference system declared in a GAR product's metadata and used for that product's spatial coordinates.
_Avoid_: WGS 84 input

**Source data**:
The original GAR NetCDF data published by the GAR archive before a GAR product is derived from it.
_Avoid_: raw data

**AWS observation product**:
A versioned, quality-controlled automatic weather-station time series with its source metadata and changelog.
_Avoid_: AWS data, measurements

**Observation-source register**:
A version-controlled list of reviewed public catalogue pages from which processed automatic weather-station observation products may be imported.
_Avoid_: crawl list

**Station location record**:
The canonical identity, display name, coordinates, elevation, exposure, sensor heights, and validity interval for an automatic weather station.
_Avoid_: station

**GAR station-sampled product**:
A GAR product sampled at a station location using a stated spatial sampling method and recording the sampled grid coordinates.
_Avoid_: extracted data

**Earth-relative wind component**:
A wind-vector component expressed eastward or northward after rotation from model-grid axes using the GAR product's projection metadata.
_Avoid_: unqualified u/v component

**Prescribed sampling point**:
A pre-analysed GAR grid cell selected as the most comparable point for a station.
_Avoid_: best comparable point

**Comparison interval**:
The common valid timestamps remaining after time-zone normalization and completeness filtering of a GAR station-sampled product and an AWS observation product.
_Avoid_: overlap

**Comparison variable specification**:
The declared pairing of a GAR variable and AWS observation field, including units, conversion, and temporal aggregation.
_Avoid_: variable mapping

**Comparison result**:
The paired values, exclusions, and validation statistics produced for a comparison variable specification over a comparison interval.
_Avoid_: comparison
