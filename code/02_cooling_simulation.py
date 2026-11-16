# Portable data paths; execute notebooks from the package root.
import os
from pathlib import Path
_review_base = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
DATA_ROOT = Path(os.environ.get("REVIEW_DATA_ROOT", str(_review_base / "data"))).resolve()
RESULTS_ROOT = Path(os.environ.get("REVIEW_RESULTS_ROOT", str(_review_base / "results"))).resolve()
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

# -*- coding: utf-8 -*-
# =====================================================================
# Hourly PUE and WUE for every data center and every LHS sample.


#
# Three nested loops, nothing more:
#   for each weather file (= one data center)
#     for each row of case_N.csv (= one sample, 50 rows total)
#       for each hour of the year (8760 hours)
#         compute PUE and WUE for that hour
#


#   PUE_280868.csv   8760 rows x 50 columns, headers case_N_1 ... case_N_50
# =====================================================================

import os
import glob
import sys
import time
import warnings
import re
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import multiprocessing as mp
import pandas as pd
# import relocated below portable module-path setup



os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


MODULE_DIR = str(DATA_ROOT / 'cooling/')
os.chdir(MODULE_DIR)
sys.path.insert(0, MODULE_DIR)
import simulation_funs_DC as sim

# Silence the sklearn notice about loading old .pkl files, otherwise it floods the log
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# 1. Settings - only this block needs editing per case
# ---------------------------------------------------------------------

WEATHER_DIR = str(DATA_ROOT / 'cooling/Province_Hourly/')                        # holds files like 280868.csv
CASE_FILE   = str(DATA_ROOT / 'cooling/HyperCubicSampling/case_7.csv')    
OUTPUT_DIR  = str(DATA_ROOT / 'cooling/Results/case_7/')                  


CASE_NUMBER = int(re.search(r"case_(\d+)", CASE_FILE).group(1))     

SKIP_DONE = True    # True = skip a data center if its output files already exist
N_JOBS    = 1        
SEED      = 20260817  

COL_T, COL_RH, COL_P, COL_TIME = "T2m", "RH", "SP", "time(UTC)"

# ---------------------------------------------------------------------



#



# ---------------------------------------------------------------------

CASE_FUNCS = {
    1:  "PUE_WUE_AE_Chiller",                          
    2:  "PUE_WUE_Chiller_Watereconomier",               
    3:  "PUE_WUE_AE_Chiller_Colo",                      
    4:  "PUE_WUE_WE_Chiller_Colo",                       
    5:  "PUE_WUE_Chiller",                              
    6:  "PUE_WUE_AE_AIRChiller",                         
    7:  "PUE_WUE_AIRChiller",                            
    8:  "PUE_WUE_Chiller",                               
    9:  "PUE_WUE_AIRChiller",                            
    10: "PUE_WUE_DX",                                    
    11: "PUE_WUE_AE_Immersion_Chiller",                  
    12: "PUE_WUE_Immersion_Chiller_Watereconomier",      
}

_ORDER_AE_BASE = [                                       
    "UPS_e", "PD_lr", "L_percentage", "delta_T_air", "Fan_Pressure_CRAC",
    "Fan_e_CRAC", "Pump_Pressure_HD", "Pump_e_HD", "AT_CT", "Chiller_load",
    "delta_T_water", "Pump_Pressure_CW", "Pump_e_CW", "delta_T_CT",
    "Pump_Pressure_CT", "Pump_e_CT", "Windage_p", "CC", "Fan_Pressure_CT",
    "Fan_e_CT", "SHR", "LGRatio", "T_up", "T_lw", "dp_up", "dp_lw",
    "RH_up", "RH_lw", "pcop",
]

_ORDER_WE = [                                            
    "UPS_e", "PD_lr", "L_percentage", "SHR", "delta_T_air", "Fan_Pressure_CRAC",
    "Fan_e_CRAC", "Pump_Pressure_HD", "Pump_e_HD", "HTE", "delta_T_water",
    "AT_CT", "AT_HE", "Pump_Pressure_WE", "Pump_e_WE", "Pump_Pressure_CW",
    "Pump_e_CW", "Chiller_load", "delta_T_CT", "Pump_Pressure_CT", "Pump_e_CT",
    "Windage_p", "CC", "LGRatio", "Fan_Pressure_CT", "Fan_e_CT", "T_up", "T_lw",
    "dp_up", "dp_lw", "RH_up", "RH_lw", "pcop",
]

_ORDER_CHILLER_MID_SMALL = [                             
    "UPS_e", "PD_lr", "L_percentage", "SHR", "delta_T_air",
    "Fan_Pressure_CRAC", "Fan_e_CRAC", "Pump_Pressure_HD", "Pump_e_HD", "HTE",
    "delta_T_water", "Pump_Pressure_CW", "Pump_e_CW", "AT_CT", "Chiller_load",
    "delta_T_CT", "Pump_Pressure_CT", "Pump_e_CT", "Windage_p", "CC",
    "Fan_Pressure_CT", "Fan_e_CT", "LGRatio", "T_up", "T_lw", "dp_up", "dp_lw",
    "RH_up", "RH_lw", "pcop",
]

_ORDER_AE_AIRCHILLER = [                                 # Case 6 (PUE_WUE_AE_AIRChiller)
    "UPS_e", "PD_lr", "L_percentage", "delta_T_air",
    "Fan_Pressure_CRAC", "Fan_e_CRAC", "SHR", "Pump_Pressure_HD", "Pump_e_HD",
    "HTE", "delta_T_water", "Pump_Pressure_CW", "Pump_e_CW", "pcop",
    "Chiller_load", "T_up", "T_lw", "dp_up", "dp_lw", "RH_up", "RH_lw",
]

_ORDER_AIRCHILLER = [                                    
    "UPS_e", "PD_lr", "L_percentage", "SHR", "delta_T_air",
    "Fan_Pressure_CRAC", "Fan_e_CRAC", "Pump_Pressure_HD", "Pump_e_HD", "HTE",
    "delta_T_water", "Pump_Pressure_CW", "Pump_e_CW", "Chiller_load", "pcop",
    "T_up", "T_lw", "dp_up", "dp_lw", "RH_up", "RH_lw",
]

_ORDER_DX = [                                            # Case 10 (PUE_WUE_DX)
    "UPS_e", "PD_lr", "L_percentage", "SHR", "delta_T_air",
    "Fan_Pressure_CRAC", "Fan_e_CRAC", "T_up", "T_lw", "dp_up", "dp_lw",
    "RH_up", "RH_lw", "pcop",
]

_IMMERSION_EXTRA = ["rho_CL", "f_CL", "Pump_Pressure_CL", "Pump_e_CL"]  
_WE_EXTRA        = ["HTE", "AT_HE", "Pump_Pressure_WE", "Pump_e_WE"]    


PARAM_ORDER = {
    1:  _ORDER_AE_BASE,
    2:  _ORDER_WE,
    3:  _ORDER_AE_BASE,
    4:  _ORDER_WE,
    5:  _ORDER_CHILLER_MID_SMALL,
    6:  _ORDER_AE_AIRCHILLER,
    7:  _ORDER_AIRCHILLER,
    8:  _ORDER_CHILLER_MID_SMALL,
    9:  _ORDER_AIRCHILLER,
    10: _ORDER_DX,
    11: _ORDER_AE_BASE + _IMMERSION_EXTRA,                 # 33
    12: _ORDER_AE_BASE + _WE_EXTRA + _IMMERSION_EXTRA,     # 37
}


OUT_NAMES = {
    1:  ["PUE", "WUE"],
    2:  ["PUE", "WUE"],
    3:  ["PUE", "WUE"],
    4:  ["PUE", "WUE"],
    5:  ["PUE", "WUE"],
    6:  ["PUE", "WUE"],
    7:  ["PUE", "WUE"],
    8:  ["PUE", "WUE"],
    9:  ["PUE", "WUE"],
    10: ["PUE", "WUE"],
    11: ["PUE_single", "PUE_two", "WUE_single", "WUE_two"],
    12: ["PUE_single", "PUE_two", "WUE_single", "WUE_two"],
}


# ---------------------------------------------------------------------

# ---------------------------------------------------------------------

FUNC_NAME  = CASE_FUNCS[CASE_NUMBER]
MODEL_FUNC = getattr(sim, FUNC_NAME)
ORDER      = PARAM_ORDER[CASE_NUMBER]
OUT_LIST   = OUT_NAMES[CASE_NUMBER]
N_OUT      = len(OUT_LIST)


def load_params(case_file):
    'Load params.'
    df = pd.read_csv(case_file)

    if set(ORDER).issubset(df.columns):
        df = df[ORDER]                         
        print(f" {len(ORDER)} ")
    else:
        missing = [p for p in ORDER if p not in df.columns]
        if df.shape[1] != len(ORDER):
            raise ValueError(
                f"{FUNC_NAME} (Case {CASE_NUMBER} {len(ORDER)} "
                f"{os.path.basename(case_file)} \u53ea\u6709 {df.shape[1]} \u5217。\n"
                f"\u7f3a\u5931\u7684\u53c2\u6570: {missing}\n"
                f"\u9700\u8981\u7684\u987a\u5e8f: {ORDER}"
            )
        print(f"Warning: {missing} "
              f"\u8bf7\u81ea\u884c\u786e\u8ba4 csv \u5217\u987a\u5e8f == {ORDER}")

    return df.to_numpy(dtype=float)


PARAMS   = load_params(CASE_FILE)
N_SAMPLE = PARAMS.shape[0]
N_PARAM  = PARAMS.shape[1]
COL_NAMES = [f"case_{CASE_NUMBER}_{i + 1}" for i in range(N_SAMPLE)]

print(f"case {CASE_NUMBER} -> {FUNC_NAME}")
print(f"{N_SAMPLE} samples x {N_PARAM} {N_OUT} {OUT_LIST}")


# ---------------------------------------------------------------------

# ---------------------------------------------------------------------

def run_one_dc(weather_file):
    dc_id = os.path.splitext(os.path.basename(weather_file))[0]

    out_paths = {name: os.path.join(OUTPUT_DIR, f"{name}_{dc_id}.csv")
                 for name in OUT_LIST}

    if SKIP_DONE and all(os.path.exists(p) for p in out_paths.values()):
        return dc_id, "skipped", 0, 0.0

    t0 = time.perf_counter()

    try:
        weather  = pd.read_excel(weather_file)
        temp     = weather[COL_T].to_numpy(dtype=float)
        rh       = weather[COL_RH].to_numpy(dtype=float)
        pressure = weather[COL_P].to_numpy(dtype=float)
    except Exception as e:
        return dc_id, f"weather read failed: {e}", 0, 0.0

    n_hour = len(temp)

    
    tables = [np.full((n_hour, N_SAMPLE), np.nan) for _ in range(N_OUT)]

    
    rng_seed = (SEED + abs(hash(dc_id)) % 100000) % (2 ** 31)
    np.random.seed(rng_seed)

    n_fail = 0
    model_input = np.zeros(3 + N_PARAM)

    for s in range(N_SAMPLE):
        model_input[3:] = PARAMS[s]        

        for h in range(n_hour):
            model_input[0] = temp[h]
            model_input[1] = rh[h]
            model_input[2] = pressure[h]

            try:
                res = MODEL_FUNC(model_input)
                for k in range(N_OUT):
                    tables[k][h, s] = res[k]
            except Exception:
                
                
                n_fail += 1

    
    for k, name in enumerate(OUT_LIST):
        df = pd.DataFrame(tables[k], columns=COL_NAMES)
        if COL_TIME in weather.columns:
            df.insert(0, "time", weather[COL_TIME])
        df.to_csv(out_paths[name], index=False, float_format="%.6f")

    n_nan = int(sum(np.isnan(t).sum() for t in tables))
    return dc_id, "done", n_nan, (time.perf_counter() - t0) / 60


# ---------------------------------------------------------------------

# ---------------------------------------------------------------------

def main():
    weather_files = sorted(glob.glob(os.path.join(WEATHER_DIR, "*.xlsx")))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n_dc = len(weather_files)
    print(f"{n_dc} data centers found, N_JOBS = {N_JOBS}\n", flush=True)

    t_start = time.perf_counter()
    done = 0

    if N_JOBS == 1:
        results = map(run_one_dc, weather_files)
        for dc_id, status, n_nan, mins in results:
            done += 1
            note = f", {n_nan} NaN" if n_nan else ""
            print(f"[{done}/{n_dc}] {dc_id} {status} "
                  f"({mins:.1f} min{note}), "
                  f"total {(time.perf_counter() - t_start) / 60:.1f} min", flush=True)
    else:
        
        with mp.Pool(processes=N_JOBS) as pool:
            for dc_id, status, n_nan, mins in pool.imap_unordered(
                    run_one_dc, weather_files, chunksize=1):
                done += 1
                note = f", {n_nan} NaN" if n_nan else ""
                print(f"[{done}/{n_dc}] {dc_id} {status} "
                      f"({mins:.1f} min{note}), "
                      f"total {(time.perf_counter() - t_start) / 60:.1f} min", flush=True)

    print(f"\nAll done in {(time.perf_counter() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
