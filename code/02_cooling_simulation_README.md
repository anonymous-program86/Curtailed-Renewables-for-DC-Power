# Hourly data center cooling simulation

Evaluates power usage effectiveness (PUE) and water usage effectiveness (WUE) for each weather record and parameter sample. Cases 11 and 12 return separate metrics for two immersion cooling configurations.

## Inputs

Hourly weather CSVs, the selected case CSV, the external simulation_funs_DC.py module, and any model files it loads. Set CASE_FILE, WEATHER_DIR and OUTPUT_DIR before running.

## Outputs

Per-location hourly PUE and WUE CSVs under the selected case output directory.
