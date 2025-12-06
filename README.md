# Partial Evaluation for Pandas in Python
Partial evaluation (PE) framework that optimizes Pandas programs by eliminating unnecessary Python-level work such as row-wise `apply`, small static loops, repeated constant computations, and simple UDF calls. The system parses a Pandas-oriented DSL, performs binding-time analysis, and generates optimized residual Pandas code.

---

## Key Features
- Pandas-focused DSL + AST normalization  
- Two-point binding-time analysis (Static, Dynamic)  
- Online partial evaluation (no user annotations)  
- Constant folding, loop unrolling  
- UDF inlining and apply fusion  
- Function specialization for static arguments  
- Clean residual code generation  
- Benchmark suite with micro + macro workloads  

---

## Why Partial Evaluation?
Pandas scripts often contain patterns that degrade performance:
- `Series.apply` with Python UDFs  
- Small fixed-range loops  
- Repeated string/arithmetical expressions  
- Chained applies  

These prevent vectorization and introduce Python overhead.  
Partial evaluation simplifies programs *before execution* and exposes fast Pandas operations.

---

## Example Optimization

**Original**
```python
def square(x): return x * x
for i in range(2):
    df[f"Z{i}"] = df["POP"] + i
df["sq"] = df["COUNT"].apply(square)
```

**Residual**
```python
df["Z0"] = df["POP"] + 0
df["Z1"] = df["POP"] + 1
df["sq"] = df["COUNT"] * df["COUNT"]
```
## Benchmarks (Summary)

- Up to **112×** speedup for UDF `.apply` removal  
- Apply fusion: **~1.5×**  
- Loop unrolling, constant folding, specialization: **~1.1–1.3×**  
- Geometric mean across tasks: **2.3×**  

All residual programs preserve Pandas semantics.

---

## Installation

```bash
git clone https://github.com/uwm-se/PE.git
cd PE
pip install pandas numpy lark-parser
```

## Usage

### Partial Evaluator
```bash
python main.py
```

### Evaluate Speedups
```bash
python eval.py
```

### Run Benchmarks
```bash
python bench.py --sizes 100000,1000000 --repeat 3
```


## Outputs saved to:
```
benchmark_results.csv
```
## Repository Structure
```
main.py     # DSL parser, AST, partial evaluator, residual code emitter
bench.py    # Benchmark suite
eval.py     # Quick evaluation runner
m3d1.csv    #Data file1
m3d2.csv    #Data file 2
README.md
```
## Limitations

* Limited support for complex groupby/join transformations
* UDF purity not automatically verified
* DSL excludes advanced Python features (classes, generators, contexts)

