# Spec sheet — `bench/tritonbench_g/matmul_persistent_triton/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/matmul_persistent_triton/matmul_persistent_triton.py`

## Public theorem: `source_equivalence`

<details><summary>docstring</summary>

```
/-- Exact complete typed-source equivalence under every backend algebra for
both original wrapper storage profiles. The sole rewrite is an exact signed
integer address multiplication; all floating computations, casts, persistent
loop iterations, masks, stores, resets, failures and full states are retained.
No floating numerical assumption or successful-execution premise is used.
This is the formal typed source; native fixed-width integer correspondence
and legal tile/dtype/launch admission remain separate obligations. -/
```
</details>

**Statement:**
```lean
specification source_equivalence (p : Parameters) (storage : TypedSource.Storage) :
    program p storage Bool.false ≡[assumptions] program p storage Bool.true
```

**Closed-form spec defs (transitive):** `program`, `assumptions`, `kernel`

<details><summary><code>program</code></summary>

```lean
def program (p : Parameters) (storage : TypedSource.Storage) (v : Bool) : Program where
  kernel := MatmulPersistentTriton.kernel p storage v
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

<details><summary><code>kernel</code></summary>

```lean
def kernel (p : Parameters) (storage : TypedSource.Storage) (v : Bool) : ComputeKernel :=
  p.source storage v
```
</details>
