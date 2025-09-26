import os, signal, time, tempfile, logging, re, sys
import concurrent.futures as cf
import multiprocessing as mp
from pathlib import Path
from typing import Iterable, Tuple, List, Optional

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ---------- Helpers ----------
ENV_NO_OMP = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

def _set_tmpdir():
    tmp = os.environ.get("TMPDIR") or tempfile.gettempdir()
    tempfile.tempdir = tmp  # ayuda a evitar NFS en cancelaciones
    return tmp

def _blas_sanitize():
    # Evita over-subscription dentro de cada proceso hijo
    for k, v in ENV_NO_OMP.items():
        os.environ.setdefault(k, v)

def _slurm_workers() -> int:
    try:
        return max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    except Exception:
        return max(1, os.cpu_count() or 1)

def _bounded_submit(
    ex: cf.Executor,
    job_iter: Iterable[Tuple[callable, tuple, dict]],
    max_in_flight: int,
    stop_flag: dict,
):
    """Mantiene un número acotado de tareas en vuelo."""
    in_flight = set()
    it = iter(job_iter)

    # Priming
    try:
        for _ in range(max_in_flight):
            func, args, kwargs = next(it)
            fut = ex.submit(func, *args, **kwargs)
            in_flight.add(fut)
    except StopIteration:
        pass

    # Consume resultados y repone
    while in_flight and not stop_flag["flag"]:
        done, in_flight = cf.wait(in_flight, return_when=cf.FIRST_COMPLETED)
        for fut in done:
            # Propaga excepción del worker si la hubo (queda log abajo)
            fut.result()
        # Reponer hasta mantener el cupo
        try:
            for _ in range(len(done)):
                func, args, kwargs = next(it)
                fut = ex.submit(func, *args, **kwargs)
                in_flight.add(fut)
        except StopIteration:
            # no hay más trabajos -> drenamos lo que queda
            pass

    # Si nos pidieron frenar, no seguimos reponiendo; dejamos que se apaguen
    return

# ---------- Workers (procesos) ----------
def _worker_sim_fit(i: int,
                    system_type: str, model: str, algo: str,
                    path_TRILEGAL_set: str, path_GENULENS_set: str,
                    path_to_save_model: str, path_to_save_fit: str,
                    path_ephemerides: str, path_dataslice: str):
    """Worker que llama directamente a functions_roman_rubin.sim_fit(...) en un proceso hijo."""
    _blas_sanitize()
    try:
        from simulator import sim_rubin_event
        sim_rubin_event(i, system_type, model, algo,
                path_TRILEGAL_set, path_GENULENS_set,
                path_to_save_model, path_to_save_fit,
                path_ephemerides, path_dataslice)
        logging.info(f"[sim:{i}] OK")
        return 0
    except Exception as e:
        logging.exception(f"[sim:{i}] FAILED: {e}")
        # Lanza para que el futuro marque error y aparezca en el .result()
        raise

def _worker_read_fit(nsource: int, nset: int, path_run: str,
                     model: str, algo: str, path_to_save_fit: str,
                     path_ephemerides: str):
    """Worker que llama directamente a functions_roman_rubin.read_fit(...) en un proceso hijo."""
    _blas_sanitize()
    try:
        from functions_roman_rubin import read_fit
        read_fit(nsource, str(nset), path_run, model, algo,
                 path_to_save_fit, path_ephemerides)
        logging.info(f"[read:{nset}:{nsource}] OK")
        return 0
    except Exception as e:
        logging.exception(f"[read:{nset}:{nsource}] FAILED: {e}")
        raise

# ---------- Generadores de trabajos ----------
def _iter_sim_jobs(total_events: int,
                   system_type: str, model: str, algo: str,
                   path_TRILEGAL_set: str, path_GENULENS_set: str,
                   path_to_save_model: str, path_to_save_fit: str,
                   path_ephemerides: str, path_dataslice: str):
    for i in range(int(total_events)):
        yield (_worker_sim_fit,
               (i, system_type, model, algo,
                path_TRILEGAL_set, path_GENULENS_set,
                path_to_save_model, path_to_save_fit,
                path_ephemerides, path_dataslice),
               {})

def _list_event_numbers(h5_dir: Path) -> List[int]:
    patt = re.compile(r"Event_(\d+)\.h5$")
    nums: List[int] = []
    for f in h5_dir.glob("*.h5"):
        m = patt.search(f.name)
        if m:
            nums.append(int(m.group(1)))
    nums.sort()
    return nums

def _iter_readfit_jobs(nset: int, path_run: str, model: str, algo: str,
                       path_to_save_fit: str, path_ephemerides: str):
    directory = Path(path_run) / f"set_sim{nset}"
    for nsource in _list_event_numbers(directory):
        yield (_worker_read_fit,
               (nsource, nset, path_run, model, algo, path_to_save_fit, path_ephemerides),
               {})

# ---------- API pública ----------
def run_parallel(path_ephemerides, path_dataslice, path_TRILEGAL_set, path_GENULENS_set,
                 path_to_save_fit, path_to_save_model, model, system_type, algo,
                 N_tr,
                 total_events: int = 250_000,
                 max_in_flight: Optional[int] = None):
    """
    Versión adaptada al snippet:
    - ProcessPoolExecutor con 'spawn'
    - Respeta SLURM_CPUS_PER_TASK
    - Terminación cooperativa por SIGTERM
    - TMPDIR local
    - Back-pressure (max_in_flight) para no saturar memoria/planificador
    """
    _set_tmpdir()
    workers = N_tr or _slurm_workers()
    if max_in_flight is None:
        # Mantener algunas docenas por worker suele ir bien (I/O, RAM)
        max_in_flight = max(4, workers * 20)

    # Señal de stop cooperativa (e.g., Slurm --signal=SIGTERM@60)
    stop = {"flag": False}
    def _term_handler(signum, frame):
        stop["flag"] = True
        logging.warning("SIGTERM recibido: deteniendo consumo y cancelando tareas...")
    signal.signal(signal.SIGTERM, _term_handler)

    ctx = mp.get_context("spawn")
    ex = cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx)

    jobs = _iter_sim_jobs(
        total_events, system_type, model, algo,
        path_TRILEGAL_set, path_GENULENS_set,
        path_to_save_model, path_to_save_fit,
        path_ephemerides, path_dataslice
    )

    try:
        _bounded_submit(ex, jobs, max_in_flight=max_in_flight, stop_flag=stop)
    finally:
        # Apagado rápido y cancelación de futuros no iniciados
        ex.shutdown(wait=False, cancel_futures=True)

def run_parallel_read_fit(nset, path_run, path_ephemerides, path_to_save_fit,
                          model, algo, N_tr,
                          max_in_flight: Optional[int] = None):
    _set_tmpdir()
    workers = N_tr or _slurm_workers()
    if max_in_flight is None:
        max_in_flight = max(4, workers * 20)

    stop = {"flag": False}
    def _term_handler(signum, frame):
        stop["flag"] = True
        logging.warning("SIGTERM recibido: deteniendo consumo y cancelando tareas...")
    signal.signal(signal.SIGTERM, _term_handler)

    ctx = mp.get_context("spawn")
    ex = cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx)

    jobs = _iter_readfit_jobs(nset, path_run, model, algo, path_to_save_fit, path_ephemerides)

    try:
        _bounded_submit(ex, jobs, max_in_flight=max_in_flight, stop_flag=stop)
    finally:
        ex.shutdown(wait=False, cancel_futures=True)

# ---------- Main opcional para pruebas ----------
if __name__ == "__main__":
    mp.freeze_support()
    # Pequeña prueba sintética (reemplaza por tus calls reales)
    # run_parallel(...); run_parallel_read_fit(...)
    print("runner_rr loaded")
