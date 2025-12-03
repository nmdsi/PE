#Evaluation
import pandas as pd
import numpy as np
import time


def measure(fn, repeat=3):
    times = []
    for _ in range(repeat):
        start = time.time()
        fn()
        times.append(time.time() - start)
    return sum(times) / repeat


def constant_folding_test():
    print("\n[1] Constant Folding")
    N = 1_000_000
    df = pd.DataFrame({"A": np.arange(N)})

    def run_original():
        df1 = df.copy()
        df1["Z"] = df1["A"] + (10 + 5)  
        df1["TAG"] = "T" + str(7)       
        return df1

    def run_residual():
        df1 = df.copy()
        df1["Z"] = df1["A"] + 15
        df1["TAG"] = "T7"
        return df1

    t_o, t_r = measure(run_original), measure(run_residual)
    ok = run_original().equals(run_residual())
    print(f"Original {t_o:.4f}s, Residual {t_r:.4f}s, Speedup {t_o/t_r:.2f}×, Equal={ok}")
    return ("Constant Folding", t_o, t_r, t_o / t_r, ok)


def loop_unrolling_test():
    print("\n[2] Loop Unrolling")
    N = 1_000_000
    df = pd.DataFrame({"COUNT": np.arange(N)})

    def run_original():
        df1 = df.copy()
        for i in range(3):
            df1[f"N{i}"] = df1["COUNT"] + i
        return df1

    def run_residual():
        df1 = df.copy()
        df1["N0"] = df1["COUNT"] + 0
        df1["N1"] = df1["COUNT"] + 1
        df1["N2"] = df1["COUNT"] + 2
        return df1

    t_o, t_r = measure(run_original), measure(run_residual)
    ok = run_original().equals(run_residual())
    print(f"Original {t_o:.4f}s, Residual {t_r:.4f}s, Speedup {t_o/t_r:.2f}×, Equal={ok}")
    return ("Loop Unrolling", t_o, t_r, t_o / t_r, ok)


def udf_inlining_test():
    print("\n[3] UDF Inlining for .apply")
    N = 1_000_000
    df = pd.DataFrame({"x": np.arange(N)})

    def square(v): return v * v

    def run_original():
        df1 = df.copy()
        df1["sq"] = df1["x"].apply(square)
        return df1

    def run_residual():
        df1 = df.copy()
        df1["sq"] = df1["x"] * df1["x"]
        return df1

    t_o, t_r = measure(run_original), measure(run_residual)
    ok = run_original().equals(run_residual())
    print(f"Original {t_o:.4f}s, Residual {t_r:.4f}s, Speedup {t_o/t_r:.2f}×, Equal={ok}")
    return ("UDF Inlining", t_o, t_r, t_o / t_r, ok)


def apply_fusion_test():
    print("\n[4] Apply Fusion")
    N = 1_000_000
    df = pd.DataFrame({"val": np.arange(N)})

    def inc(v): return v + 1
    def triple(v): return v * 3

    def run_original():
        df1 = df.copy()
        df1["val"] = df1["val"].apply(inc).apply(triple)
        return df1

    def run_residual():
        df1 = df.copy()
        df1["val"] = df1["val"].apply(lambda v: triple(inc(v)))
        return df1

    t_o, t_r = measure(run_original), measure(run_residual)
    ok = run_original().equals(run_residual())
    print(f"Original {t_o:.4f}s, Residual {t_r:.4f}s, Speedup {t_o/t_r:.2f}×, Equal={ok}")
    return ("Apply Fusion", t_o, t_r, t_o / t_r, ok)


def function_specialization_test():
    print("\n[5] Function Specialization (static arguments)")
    N = 1_000_000
    df = pd.DataFrame({"VAL": np.arange(N)})

    def scale_and_tag(df, scale, tag):
        df["VAL"] = df["VAL"] * scale
        df["TAG"] = "T" + str(tag)
        return df

    def run_original():
        df1 = df.copy()
        return scale_and_tag(df1, 2, 7)

    def run_residual():
        df1 = df.copy()

        def scale_and_tag__S1(df):
            df["VAL"] = df["VAL"] * 2
            df["TAG"] = "T7"
            return df

        return scale_and_tag__S1(df1)

    t_o, t_r = measure(run_original), measure(run_residual)
    ok = run_original().equals(run_residual())
    print(f"Original {t_o:.4f}s, Residual {t_r:.4f}s, Speedup {t_o/t_r:.2f}×, Equal={ok}")
    return ("Function Specialization", t_o, t_r, t_o / t_r, ok)


def column_operation_test():
    print("\n[6] Column Operations (static expressions)")
    N = 1_000_000
    df = pd.DataFrame({"COUNT": np.arange(N), "POP": np.arange(N, 2 * N)})

    def run_original():
        df1 = df.copy()
        df1["RATIO"] = df1["COUNT"] / df1["POP"]
        df1["LABEL"] = "A" + str(1)
        return df1

    def run_residual():
        df1 = df.copy()
        df1.loc[:, "RATIO"] = df1["COUNT"] / df1["POP"]
        df1.loc[:, "LABEL"] = "A1"
        return df1

    t_o, t_r = measure(run_original), measure(run_residual)
    ok = run_original().equals(run_residual())
    print(f"Original {t_o:.4f}s, Residual {t_r:.4f}s, Speedup {t_o/t_r:.2f}×, Equal={ok}")
    return ("Column Ops", t_o, t_r, t_o / t_r, ok)





def main():
    print("Evaluation\n")
    results = []
    results.append(constant_folding_test())
    results.append(loop_unrolling_test())
    results.append(udf_inlining_test())
    results.append(apply_fusion_test())
    results.append(function_specialization_test())
    results.append(column_operation_test())

    print("\n=== Summary ===")
    print(f"{'Benchmark':<28} {'Orig(s)':>10} {'Resid(s)':>10} {'Speedup':>10} {'Equal':>10}")
    for name, t_o, t_r, sp, ok in results:
        print(f"{name:<28} {t_o:>10.4f} {t_r:>10.4f} {sp:>10.2f} {str(ok):>10}")


if __name__ == "__main__":
    main()
