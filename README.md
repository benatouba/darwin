# Analysis tools for the [DARWIN](https://darwin-rain.org) project and GAR data

The GAR dataset is modelled on a WRF Mercator grid with 2 km grid spacing. The
[salem](https://salem.readthedocs.io/en/stable/index.html) package provides methods for easy access
to the data that are used in this project.

If there are any issues with metadata, inhibiting correct data access via
[salem](https://salem.readthedocs.io/en/stable/index.html), the `darwin/change_gar_names.py` script
can be used to fix the metadata.

```python
python darwin/change_gar_names.py -h # for help
```

The `darwin/core.py` module contains the main classes and methods for data access and analysis. The
`darwin/defaults.py` module contains default values for the project.

There are scripts specifically to cater to the
[darwin-dashboard](https://gitlab.klima.tu-berlin.de/schmidt/darwin-dashboard)
project. The script `darwin/combine_to_measurement_period.py` is used to combine the GAR data to
the measurement period of the DARWIN measurement network.

BEWARE: This project is a WIP