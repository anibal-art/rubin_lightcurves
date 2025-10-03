# main.py
import multiprocessing as mp
import numpy as np
import json
from pathlib import Path
from getpass import getpass

from dl import authClient as ac, queryClient as qc
from dl.helpers.utils import convert

from simulator import sim_rubin_event, build_mu_rel_pairs
from sim_parallel import run_parallel  # <-- Approach B module (parquet/initializer)

def _as_bool(x):
    # Accept true, "true", "True", etc.
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", "y"}
    return bool(x)

def cli():
    script_dir = Path.cwd()

    # ---- Login (parent only) ----
    token = ac.login(input("Enter user name: (+ENTER) "), getpass("Enter password: (+ENTER) "))
    ac.whoAmI()
    if not ac.isValidToken(token):
        raise RuntimeError(f"Invalid AstroDataLab login/token: {token}")

    # ---- Params ----
    cfg_path = script_dir / "config_file.json"
    with cfg_path.open() as f:
        params = json.load(f)

    ra_center = params["ra"]
    dec_center = params["dec"]
    radius    = params["radius"]              # degrees
    Ds_max    = params["Ds_max"]
    N         = int(params["N"])
    if params["t0_range"]==False:
        t0_range=None
    else:
        t0_range  = tuple(params["t0_range"])
        
    model     = params["model"]
    system_type = params["system_type"]
    path_to_save_model = params["path_save"]
    run_para  = _as_bool(params.get("run_parallel", True))

    print(f"Ds_max: {Ds_max} pc")
    print(f"Generate {N} events at (ra,dec)=({ra_center},{dec_center})")

    mu0_max = 5 * np.log10(float(Ds_max)) - 5
    query = f"""
        SELECT *
        FROM lsst_sim.simdr2
        WHERE q3c_radial_query(ra, dec, {ra_center}, {dec_center}, {radius})
          AND mu0 < ({mu0_max})
        LIMIT {N+1000}
    """

    print("Requesting data from AstroDataLab ...")
    res = qc.query(sql=query, format="csv")
    df_raw = convert(res, "pandas")

    # Build source–lens pairs (parent only)
    df_pairs = build_mu_rel_pairs(df_raw, N, offset=0.1, min_D=1.0, random_state=None)
    del df_raw
    print("Data downloaded")

    # Prepare the DataFrame **with the columns your simulator expects**
    # sim_event reads: data["u"],["g"],["r"],["i"],["z"],["Y"]  (note the uppercase Y)
    # Rename umag->u, gmag->g, ..., ymag->Y
    col_map = {"umag": "u", "gmag": "g", "rmag": "r", "imag": "i", "zmag": "z", "ymag": "Y","logl": "logL","logte": "logTe"}
    keep_cols = ["D_S", "D_L", "mu_rel", "logl", "logte", "ra", "dec",
                 "umag", "gmag", "rmag", "imag", "zmag", "ymag"]
    TRILEGAL_data = df_pairs[keep_cols].rename(columns=col_map)

    # Ensure output folder exists
    Path(path_to_save_model).mkdir(parents=True, exist_ok=True)

    print("run_parallel:", run_para)
    N_workers =  params["N_tr"]
    if run_para:
        N_tr = int(params.get("N_tr", N_workers))  # this is workers count
        run_parallel(
            TRILEGAL_data=TRILEGAL_data,
            path_to_save_model=path_to_save_model,
            model=model,
            system_type=system_type,
            N_tr=N_tr,
            t0_range=t0_range,
            total_events=N,
            max_in_flight=None,
        )
    else:
        # Serial fallback
        for i in range(N):
            row = TRILEGAL_data.iloc[i]
            print("Generating event", i)
            my_own_model, pyLIMA_parameters, decision = sim_rubin_event(
                i, system_type, model, row, path_to_save_model, t0_range
            )

if __name__ == "__main__":
    mp.freeze_support()  # harmless on Linux; required on Windows
    cli()
