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
An immutable GAR product version Darwin prepared from a reviewed release manifest. It awaits a dashboard publication decision; its scientific status is recorded separately and is not changed by that decision.
_Avoid_: draft output, approved product

**Product handoff**:
Darwin's versioned, publication-ready description of prepared product candidates (`darwin.product-handoff/2`): exact artifacts and checksums, source and recipe identities, execution completeness, scientific status and acceptance references. Preparing one never publishes, moves a current-version pointer or grants access.
_Avoid_: release (for the document), published handoff

**Execution completeness**:
Whether the run that produced a product finished (`complete`) or not (`partial`), as reported with its execution evidence. It says nothing about scientific validity.
_Avoid_: success, done

**Assessment evidence**:
A Darwin receipt (`darwin.assessment-evidence/1`) of one check outcome for one criterion of an exact requirements revision and an exact dataset. Evidence is never a decision.
_Avoid_: validation result, approval

**Acceptance assessment**:
A portable record (setups contract `darwin-wrf-setups.acceptance-assessment/1`) stating that an exact dataset is accepted, rejected or unassessed under an exact requirements revision, citing exact evidence and, for a decision, the reviewer or authorized policy that made it. A decision is valid only with a detached SSH signature over its exact bytes by a key the reviewed requirements pin for that authority. It binds the product's dataset identity, never a single store. A reassessment supersedes an earlier record without rewriting it.
_Avoid_: approval flag, QC pass

**Scientific status**:
The status of a product's latest acceptance assessment: accepted, rejected or unassessed (no assessment). Independent of execution completeness and publication.
_Avoid_: approved, validated

**Publication**:
The dashboard catalogue's decision to make an exact product version visible and accessible, and to move its current-version pointer. Darwin never publishes.
_Avoid_: release (for the decision), acceptance

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
