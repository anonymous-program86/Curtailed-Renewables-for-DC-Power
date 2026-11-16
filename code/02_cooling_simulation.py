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
# 通用版：支持 Case 1-12（含 Case 11/12 浸没冷却 DLC），只需要改
# CASE_FILE 指向对应的 case_N.csv，脚本会自动挑对函数、对参数顺序。
#
# Three nested loops, nothing more:
#   for each weather file (= one data center)
#     for each row of case_N.csv (= one sample, 50 rows total)
#       for each hour of the year (8760 hours)
#         compute PUE and WUE for that hour
#
# Output: 每个 data center 输出若干个 csv 文件（Case 1-10 是 PUE/WUE 两个；
#   Case 11/12 是 PUE_single/PUE_two/WUE_single/WUE_two 四个）
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
# import simulation_funs_DC as sim   # <-- 改成合并后的完整版模块（含全部16个函数，Case1-12都能跑）

# 必须在 import numpy / sklearn 之前设置，否则每个子进程都会开满线程互相抢核
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# pkl 和 simulation_funs_DC.py 所在的目录
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
CASE_FILE   = str(DATA_ROOT / 'cooling/HyperCubicSampling/case_7.csv')    # 改这里选择 case_1.csv ... case_12.csv
OUTPUT_DIR  = str(DATA_ROOT / 'cooling/Results/case_7/')                  # 建议跟着 CASE_FILE 的编号一起改


CASE_NUMBER = int(re.search(r"case_(\d+)", CASE_FILE).group(1))     # 1-12，从文件名里自动识别

SKIP_DONE = True    # True = skip a data center if its output files already exist
N_JOBS    = 1        # 并行进程数，按申请到的核数改；1 = 单进程
SEED      = 20260817  # 固定种子保证可复现

COL_T, COL_RH, COL_P, COL_TIME = "T2m", "RH", "SP", "time(UTC)"

# ---------------------------------------------------------------------
# 2. case -> 函数名 + w[] 参数顺序
#    顺序都是直接从 simulation_funs_DC.py 源码里 w[i] 的定义核对抄下来的，
#    并用示例数据实测跑通过，不是凭猜测排的。
#
#    注意：Case 5/Case 8 都调用 PUE_WUE_Chiller（中型/小型只是取值区间不同，
#    定义在 Table_B1 里，不体现在这里）；Case 7/Case 9 都调用
#    PUE_WUE_AIRChiller，同理。
# ---------------------------------------------------------------------

CASE_FUNCS = {
    1:  "PUE_WUE_AE_Chiller",                          # 29 params, 返回 2 个值 (PUE, WUE)
    2:  "PUE_WUE_Chiller_Watereconomier",               # 33 params, 返回 2 个值
    3:  "PUE_WUE_AE_Chiller_Colo",                      # 29 params, 返回 2 个值
    4:  "PUE_WUE_WE_Chiller_Colo",                       # 33 params, 返回 2 个值
    5:  "PUE_WUE_Chiller",                              # 30 params, 返回 2 个值 (midsize)
    6:  "PUE_WUE_AE_AIRChiller",                         # 21 params, 返回 2 个值
    7:  "PUE_WUE_AIRChiller",                            # 21 params, 返回 2 个值 (midsize)
    8:  "PUE_WUE_Chiller",                               # 30 params, 返回 2 个值 (small)
    9:  "PUE_WUE_AIRChiller",                            # 21 params, 返回 2 个值 (small)
    10: "PUE_WUE_DX",                                    # 14 params, 返回 2 个值
    11: "PUE_WUE_AE_Immersion_Chiller",                  # 33 params, 返回 4 个值 (AE + DLC)
    12: "PUE_WUE_Immersion_Chiller_Watereconomier",      # 37 params, 返回 4 个值 (WE + DLC)
}

_ORDER_AE_BASE = [                                       # Case 1 / 3 / 11(基础部分) 共用
    "UPS_e", "PD_lr", "L_percentage", "delta_T_air", "Fan_Pressure_CRAC",
    "Fan_e_CRAC", "Pump_Pressure_HD", "Pump_e_HD", "AT_CT", "Chiller_load",
    "delta_T_water", "Pump_Pressure_CW", "Pump_e_CW", "delta_T_CT",
    "Pump_Pressure_CT", "Pump_e_CT", "Windage_p", "CC", "Fan_Pressure_CT",
    "Fan_e_CT", "SHR", "LGRatio", "T_up", "T_lw", "dp_up", "dp_lw",
    "RH_up", "RH_lw", "pcop",
]

_ORDER_WE = [                                            # Case 2 / 4 共用
    "UPS_e", "PD_lr", "L_percentage", "SHR", "delta_T_air", "Fan_Pressure_CRAC",
    "Fan_e_CRAC", "Pump_Pressure_HD", "Pump_e_HD", "HTE", "delta_T_water",
    "AT_CT", "AT_HE", "Pump_Pressure_WE", "Pump_e_WE", "Pump_Pressure_CW",
    "Pump_e_CW", "Chiller_load", "delta_T_CT", "Pump_Pressure_CT", "Pump_e_CT",
    "Windage_p", "CC", "LGRatio", "Fan_Pressure_CT", "Fan_e_CT", "T_up", "T_lw",
    "dp_up", "dp_lw", "RH_up", "RH_lw", "pcop",
]

_ORDER_CHILLER_MID_SMALL = [                             # Case 5 / 8 共用 (PUE_WUE_Chiller)
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

_ORDER_AIRCHILLER = [                                    # Case 7 / 9 共用 (PUE_WUE_AIRChiller)
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

_IMMERSION_EXTRA = ["rho_CL", "f_CL", "Pump_Pressure_CL", "Pump_e_CL"]  # 浸没冷却液 4 个新参数
_WE_EXTRA        = ["HTE", "AT_HE", "Pump_Pressure_WE", "Pump_e_WE"]    # waterside economizer 4 个参数

# w[3:] 的真实顺序（前 3 位固定是 T_oa, RH_oa, P_oa，运行时逐小时填入，不在这里列）
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

# 返回值的名字，决定输出几个文件
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
# 3. 读参数并按函数的真实顺序重排
# ---------------------------------------------------------------------

FUNC_NAME  = CASE_FUNCS[CASE_NUMBER]
MODEL_FUNC = getattr(sim, FUNC_NAME)
ORDER      = PARAM_ORDER[CASE_NUMBER]
OUT_LIST   = OUT_NAMES[CASE_NUMBER]
N_OUT      = len(OUT_LIST)


def load_params(case_file):
    """读 case csv，按函数期望的顺序重排列。
    有表头就按列名对齐（能自动修正抽样脚本里的顺序错误）；
    没表头就只能按位置用，并且会警告。"""
    df = pd.read_csv(case_file)

    if set(ORDER).issubset(df.columns):
        df = df[ORDER]                         # 按名字重排，最安全
        print(f"参数按列名重排完成 ({len(ORDER)} 列)")
    else:
        missing = [p for p in ORDER if p not in df.columns]
        if df.shape[1] != len(ORDER):
            raise ValueError(
                f"{FUNC_NAME} (Case {CASE_NUMBER}) 需要 {len(ORDER)} 个参数，"
                f"{os.path.basename(case_file)} 只有 {df.shape[1]} 列。\n"
                f"缺失的参数: {missing}\n"
                f"需要的顺序: {ORDER}"
            )
        print(f"警告: 列名对不上（缺 {missing}），只能按位置读取，"
              f"请自行确认 csv 列顺序 == {ORDER}")

    return df.to_numpy(dtype=float)


PARAMS   = load_params(CASE_FILE)
N_SAMPLE = PARAMS.shape[0]
N_PARAM  = PARAMS.shape[1]
COL_NAMES = [f"case_{CASE_NUMBER}_{i + 1}" for i in range(N_SAMPLE)]

print(f"case {CASE_NUMBER} -> {FUNC_NAME}")
print(f"{N_SAMPLE} samples x {N_PARAM} params, 输出 {N_OUT} 个指标: {OUT_LIST}")


# ---------------------------------------------------------------------
# 4. 单个数据中心的计算（子进程里执行）
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

    # 每个指标一张 8760 x n_sample 的表
    tables = [np.full((n_hour, N_SAMPLE), np.nan) for _ in range(N_OUT)]

    # 固定随机种子：同一个 dc + 同一个 case 每次跑结果一致
    rng_seed = (SEED + abs(hash(dc_id)) % 100000) % (2 ** 31)
    np.random.seed(rng_seed)

    n_fail = 0
    model_input = np.zeros(3 + N_PARAM)

    for s in range(N_SAMPLE):
        model_input[3:] = PARAMS[s]        # 参数全年不变，填一次就行

        for h in range(n_hour):
            model_input[0] = temp[h]
            model_input[1] = rh[h]
            model_input[2] = pressure[h]

            try:
                res = MODEL_FUNC(model_input)
                for k in range(N_OUT):
                    tables[k][h, s] = res[k]
            except Exception:
                # 某些极端气象点会让 CoolProp / fsolve 失败，
                # 留 NaN 继续跑，不要让一个小时毁掉整个数据中心
                n_fail += 1

    # 写盘
    for k, name in enumerate(OUT_LIST):
        df = pd.DataFrame(tables[k], columns=COL_NAMES)
        if COL_TIME in weather.columns:
            df.insert(0, "time", weather[COL_TIME])
        df.to_csv(out_paths[name], index=False, float_format="%.6f")

    n_nan = int(sum(np.isnan(t).sum() for t in tables))
    return dc_id, "done", n_nan, (time.perf_counter() - t0) / 60


# ---------------------------------------------------------------------
# 5. 主循环
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
        # chunksize=1: 每个数据中心是个大任务，不需要打包
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
