import argparse, csv, gc, platform, random, time, tracemalloc, sys
import numpy as np, pandas as pd
from typing import Callable, Dict, List, Tuple

def parse_args():
    ap = argparse.ArgumentParser(description="Pandas PE benchmark")
    ap.add_argument("--sizes", default="100000,1000000,2000000",
                    help="comma-separated Ns (default: 100k,1M,2M)")
    ap.add_argument("--repeat", type=int, default=3, help="timed runs per case (default 3)")
    ap.add_argument("--warmup", type=int, default=1, help="warmup runs per case (default 1)")
    ap.add_argument("--micros-only", action="store_true", help="run micro tasks only")
    ap.add_argument("--macros-only", action="store_true", help="run macro tasks only")
    ap.add_argument("--trace-mem", action="store_true", help="enable tracemalloc peak memory")
    ap.add_argument("--out", default="benchmark_results.csv", help="CSV output path")
    return ap.parse_args()

def now() -> float:
    return time.perf_counter()

def measure(fn: Callable[[], pd.DataFrame], repeat: int, warmup: int, trace_mem: bool) -> Tuple[float, float, int]:
    for _ in range(warmup):
        _ = fn()
    times = []
    if trace_mem:
        tracemalloc.start()
    try:
        for _ in range(repeat):
            gc.collect()
            t0 = now()
            out = fn()
            dt = now() - t0
            times.append(dt)
            _ = out.shape  
            del out
        peak = tracemalloc.get_traced_memory()[1] if trace_mem else -1
    finally:
        if trace_mem:
            tracemalloc.stop()
    return float(np.mean(times)), float(np.std(times)), int(peak)

def frames_equal(a: pd.DataFrame, b: pd.DataFrame, tol: float = 1e-12, sort_keys: List[str] = None) -> bool:
    if sort_keys and all(k in a for k in sort_keys) and all(k in b for k in sort_keys):
        a = a.sort_values(sort_keys).reset_index(drop=True)
        b = b.sort_values(sort_keys).reset_index(drop=True)
    if list(a.columns) != list(b.columns): return False
    if a.shape != b.shape: return False
    if tuple(a.dtypes.astype(str)) != tuple(b.dtypes.astype(str)): return False
    for col in a.columns:
        x, y = a[col], b[col]
        if pd.api.types.is_float_dtype(x):
            if not np.allclose(x.to_numpy(), y.to_numpy(), equal_nan=True, rtol=0, atol=tol): return False
        else:
            if not x.equals(y): return False
    return True

def df_bytes(df: pd.DataFrame) -> int:
    try: return int(df.memory_usage(index=True, deep=True).sum())
    except Exception: return -1

def env_info() -> Dict[str, str]:
    return dict(
        python=platform.python_version(),
        pandas=pd.__version__,
        numpy=np.__version__,
        cpu=platform.processor() or platform.machine(),
        system=f"{platform.system()} {platform.release()}"
    )

def save_results(path: str, env: Dict[str, str], rows: List[Dict]):
    write_header = True
    try:
        with open(path, "r", encoding="utf-8"):  
            write_header = False
    except FileNotFoundError:
        pass
    mode = "a" if not write_header else "w"
    with open(path, mode, newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["python", env["python"], "pandas", env["pandas"], "numpy", env["numpy"], "cpu", env["cpu"], "system", env["system"]])
            w.writerow(["name","N","t_orig","sd_orig","t_resid","sd_resid","speedup","equal","peak_py_bytes_orig","peak_py_bytes_resid","out_bytes"])
        for r in rows:
            w.writerow([
                r["name"], r["N"],
                f"{r['t_orig']:.6f}", f"{r['sd_orig']:.6f}",
                f"{r['t_resid']:.6f}", f"{r['sd_resid']:.6f}",
                f"{r['speedup']:.3f}", r["equal"],
                r["peak_py_bytes_orig"], r["peak_py_bytes_resid"], r["out_bytes"]
            ])

def run_case(name: str,
             make_original: Callable[[], pd.DataFrame],
             make_residual: Callable[[], pd.DataFrame],
             repeat: int, warmup: int, trace_mem: bool,
             sort_keys: List[str] = None) -> Dict:
    t_o, s_o, peak_o = measure(make_original, repeat, warmup, trace_mem)
    t_r, s_r, peak_r = measure(make_residual, repeat, warmup, trace_mem)
    a = make_original(); b = make_residual()
    ok = frames_equal(a, b, sort_keys=sort_keys); out_bytes = max(df_bytes(a), df_bytes(b))
    del a, b; gc.collect()
    sp = (t_o / t_r) if t_r > 0 else float("inf")
    print(f"{name:<32} orig={t_o:.4f}s±{s_o:.4f}  resid={t_r:.4f}s±{s_r:.4f}  speedup={sp:.2f}  equal={ok}")
    return dict(name=name, t_orig=t_o, sd_orig=s_o, t_resid=t_r, sd_resid=s_r,
                speedup=sp, equal=ok, peak_py_bytes_orig=peak_o, peak_py_bytes_resid=peak_r, out_bytes=out_bytes)

def constant_folding_test(N: int):
    df = pd.DataFrame({"A": np.arange(N)})
    def run_original():
        d = df.copy(); d["Z"] = d["A"] + (10 + 5); d["TAG"] = "T" + str(7); return d
    def run_residual():
        d = df.copy(); d["Z"] = d["A"] + 15;         d["TAG"] = "T7";     return d
    return run_original, run_residual

def loop_unrolling_test(N: int):
    df = pd.DataFrame({"COUNT": np.arange(N)})
    def run_original():
        d = df.copy()
        for i in range(3): d[f"N{i}"] = d["COUNT"] + i
        return d
    def run_residual():
        d = df.copy()
        d["N0"] = d["COUNT"] + 0; d["N1"] = d["COUNT"] + 1; d["N2"] = d["COUNT"] + 2
        return d
    return run_original, run_residual

def udf_inlining_test(N: int):
    df = pd.DataFrame({"x": np.arange(N)})
    def square(v): return v * v
    def run_original():
        d = df.copy(); d["sq"] = d["x"].apply(square); return d
    def run_residual():
        d = df.copy(); d["sq"] = d["x"] * d["x"];     return d
    return run_original, run_residual

def apply_fusion_test(N: int):
    df = pd.DataFrame({"val": np.arange(N)})
    def inc(v): return v + 1
    def triple(v): return v * 3
    def run_original():
        d = df.copy(); d["val"] = d["val"].apply(inc).apply(triple); return d
    def run_residual():
        d = df.copy(); d["val"] = d["val"].apply(lambda v: triple(inc(v))); return d
    return run_original, run_residual

def function_specialization_test(N: int):
    df = pd.DataFrame({"VAL": np.arange(N)})
    def run_original():
        d = df.copy(); d["VAL"] = d["VAL"] * 2; d["TAG"] = "T" + str(7); return d
    def run_residual():
        d = df.copy(); d["VAL"] = d["VAL"] * 2; d["TAG"] = "T7";        return d
    return run_original, run_residual

def column_operation_test(N: int):
    df = pd.DataFrame({"COUNT": np.arange(N), "POP": np.arange(N, 2*N)})
    def run_original():
        d = df.copy(); d["RATIO"] = d["COUNT"] / d["POP"]; d["LABEL"] = "A" + str(1); return d
    def run_residual():
        d = df.copy(); d.loc[:, "RATIO"] = d["COUNT"] / d["POP"]; d.loc[:, "LABEL"] = "A1"; return d
    return run_original, run_residual

def make_cols(N: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "YEAR": rng.choice([2019, 2020, 2021], size=N),
        "EVENT_TYPE": rng.choice(["Mortality", "B", "X", ""], size=N),
        "POP": rng.integers(1000, 50000, size=N).astype(float),
        "COUNT": rng.integers(1, 10, size=N),
        "CITY": rng.choice(["Pune","Delhi","Bengaluru","Mumbai"], size=N),
    })
    if N > 0:
        miss = rng.choice(N, size=max(1, N//20), replace=False)
        df.loc[miss, "POP"] = np.nan
    return df

def macro_groupby_agg(N: int):
    df = make_cols(N, seed=1)
    def run_original():
        return df.groupby("CITY", dropna=False).agg(mean_pop=("POP","mean"), total=("COUNT","sum")).reset_index()
    def run_residual():
        return df.groupby("CITY", dropna=False).agg(mean_pop=("POP","mean"), total=("COUNT","sum")).reset_index()
    return run_original, run_residual, ["CITY"]

def macro_join_merge(N: int):
    dfL = make_cols(N, seed=2)[["YEAR","COUNT","CITY"]]
    dfR = make_cols(N, seed=3)[["YEAR","POP"]]
    def run_original(): return dfL.merge(dfR, on="YEAR", how="left")
    def run_residual(): return dfL.merge(dfR, on="YEAR", how="left")
    return run_original, run_residual, ["YEAR","CITY","COUNT","POP"]

def macro_string_clean(N: int):
    df = make_cols(N, seed=4)[["CITY"]].copy()
    def run_original():
        d = df.copy()
        d["CITY2"] = d["CITY"].str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
        return d
    def run_residual():
        d = df.copy()
        d["CITY2"] = d["CITY"].str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
        return d
    return run_original, run_residual, ["CITY2"]

def main():
    args = parse_args()
    Ns = [int(x) for x in args.sizes.split(",") if x.strip()]
    e = env_info()
    print("Environment:")
    print(f" python={e['python']} pandas={e['pandas']} numpy={e['numpy']} cpu={e['cpu']} system={e['system']}\n")

    tests_micro = [
        ("1 Constant folding", constant_folding_test),
        ("2 Loop unrolling", loop_unrolling_test),
        ("3 UDF lifting", udf_inlining_test),
        ("4 Apply fusion", apply_fusion_test),
        ("5 Function specialisation", function_specialization_test),
        ("6 Column assignment", column_operation_test),
    ]
    tests_macro = [
        ("1 Groupby and aggregation", macro_groupby_agg),
        ("2 Join and merge",         macro_join_merge),
        ("3 String cleaning",        macro_string_clean),
    ]

    want_micro = not args.macros_only
    want_macro = not args.micros_only

    all_rows: List[Dict] = []
    try:
        for N in Ns:
            print(f"\n=== size N={N} ===")
            if want_micro:
                for name, maker in tests_micro:
                    make_o, make_r = maker(N)
                    res = run_case(f"{name} (N={N})", make_o, make_r,
                                   repeat=args.repeat, warmup=args.warmup, trace_mem=args.trace_mem)
                    res["N"] = N
                    all_rows.append(res)
            if want_macro:
                for name, maker in tests_macro:
                    make_o, make_r, sort_keys = maker(N)
                    res = run_case(f"{name} (N={N})", make_o, make_r,
                                   repeat=args.repeat, warmup=args.warmup, trace_mem=args.trace_mem,
                                   sort_keys=sort_keys)
                    res["N"] = N
                    all_rows.append(res)
            save_results(args.out, e, all_rows); all_rows.clear()
    except MemoryError:
        print("\nMemoryError: reducing sizes or disable --trace-mem may help.", file=sys.stderr)
    finally:
        if all_rows:
            save_results(args.out, e, all_rows)
        print(f"\nSaved: {args.out}")

if __name__ == "__main__":
    main()
