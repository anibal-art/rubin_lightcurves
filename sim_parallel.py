# runner_rr_parallel.py
import os, signal, tempfile, logging
import concurrent.futures as cf
import multiprocessing as mp
from pathlib import Path
from typing import Iterable, Tuple, Optional, Dict, Any, Callable

import pandas as pd  # needed in parent and children

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ---------- Thread/BLAS limits ----------
ENV_NO_OMP = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

def _blas_sanitize():
    """Limit per-process thread pools to avoid oversubscription."""
    for k, v in ENV_NO_OMP.items():
        os.environ.setdefault(k, v)

# ---------- TMP handling ----------
def _set_tmpdir() -> str:
    tmp = os.environ.get("TMPDIR") or tempfile.gettempdir()
    tempfile.tempdir = tmp
    return tmp

# ---------- Workers’ shared DataFrame ----------
_DF: Optional[pd.DataFrame] = None

def _child_init(df_path: str):
    """
    Runs in each child process once:
    - set BLAS limits
    - load TRILEGAL DataFrame into a module-global
    """
    _blas_sanitize()
    global _DF
    # Use a fast columnar format; Parquet requires pyarrow or fastparquet.
    _DF = pd.read_parquet(df_path)

# ---------- Utility ----------
def _slurm_workers() -> int:
    try:
        return max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    except Exception:
        return max(1, os.cpu_count() or 1)

def _bounded_submit(
    ex: cf.Executor,
    job_iter: Iterable[Tuple[Callable, tuple, dict]],
    max_in_flight: int,
    stop_flag: Dict[str, bool],
):
    """Keep at most `max_in_flight` tasks in-flight; propagate worker errors."""
    in_flight = set()
    it = iter(job_iter)

    # Prime queue
    try:
        for _ in range(max_in_flight):
            func, args, kwargs = next(it)
            in_flight.add(ex.submit(func, *args, **kwargs))
    except StopIteration:
        pass

    while in_flight and not stop_flag["flag"]:
        done, in_flight = cf.wait(in_flight, return_when=cf.FIRST_COMPLETED)
        for fut in done:
            fut.result()  # raises if worker failed
        try:
            for _ in range(len(done)):
                func, args, kwargs = next(it)
                in_flight.add(ex.submit(func, *args, **kwargs))
        except StopIteration:
            pass

# ---------- Worker ----------
def _worker_sim(
    i: int,
    system_type: str,
    model: str,
    path_to_save_model: str,
    t0_range: Tuple[float, float],
) -> int:
    """
    Runs in child. Fetch row i from global _DF and call simulator.sim_rubin_event.
    """
    try:
        from simulator import sim_rubin_event  # import inside child to avoid early BLAS init
        row = _DF.iloc[i]  # pandas Series (positional index)
        # If your function expects a dict instead of a Series, use: row = row.to_dict()
        sim_rubin_event(i, system_type, model, row, path_to_save_model, t0_range)
        logging.info(f"[sim:{i}] OK")
        return i
    except Exception as e:
        logging.exception(f"[sim:{i}] FAILED: {e}")
        raise

# ---------- Job generator ----------
def _iter_sim_jobs(
    total_events: int,
    system_type: str,
    model: str,
    path_to_save_model: str,
    t0_range: Tuple[float, float],
):
    for i in range(int(total_events)):
        yield (_worker_sim, (i, system_type, model, path_to_save_model, t0_range), {})

# ---------- Public API ----------
def run_parallel(
    TRILEGAL_data: "pd.DataFrame",
    path_to_save_model: str | Path,
    model: str,
    system_type: str,
    N_tr: Optional[int],
    t0_range: Tuple[float, float],
    total_events: int = 250_000,
    max_in_flight: Optional[int] = None,
):
    """
    Parallel runner:
      - Spawns processes with 'spawn'
      - Sets BLAS thread caps in child initializer
      - Loads TRILEGAL_data once per child (from Parquet)
      - Uses bounded in-flight to avoid blowing RAM / scheduler
    """
    _set_tmpdir()

    # Persist the DF once; children will map it into memory from disk.
    df_tmp_path = Path(tempfile.gettempdir()) / f"trilegal_{os.getpid()}.parquet"
    TRILEGAL_data = TRILEGAL_data.reset_index(drop=True)
    TRILEGAL_data.to_parquet(df_tmp_path, index=False)

    workers = N_tr or _slurm_workers()
    if max_in_flight is None:
        max_in_flight = max(4, workers * 20)

    # Don’t overrun the DataFrame
    total_events = min(int(total_events), len(TRILEGAL_data))
    if total_events <= 0:
        logging.warning("No events to run (total_events <= 0).")
        return

    # Cooperative stop on SIGTERM (e.g., Slurm --signal=SIGTERM@60)
    stop = {"flag": False}
    def _term_handler(signum, frame):
        stop["flag"] = True
        logging.warning("SIGTERM recibido: deteniendo consumo y cancelando tareas...")
    signal.signal(signal.SIGTERM, _term_handler)

    ctx = mp.get_context("spawn")
    try:
        with cf.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_child_init,
            initargs=(str(df_tmp_path),),
        ) as ex:
            jobs = _iter_sim_jobs(
                total_events=total_events,
                system_type=system_type,
                model=model,
                path_to_save_model=str(path_to_save_model),
                t0_range=t0_range,
            )
            _bounded_submit(ex, jobs, max_in_flight=max_in_flight, stop_flag=stop)
    finally:
        # Cleanup temp parquet
        try:
            df_tmp_path.unlink(missing_ok=True)
        except Exception:
            logging.warning(f"No se pudo borrar el archivo temporal {df_tmp_path}")

# ---------- Main (optional quick test) ----------
if __name__ == "__main__":
    mp.freeze_support()
    print("runner_rr loaded")
