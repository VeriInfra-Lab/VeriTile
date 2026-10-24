# Spec sheet — `bench/tritonbench_g/square_matrix/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/square_matrix/square_matrix.py`

## Public theorem: `square_index_equiv`

<details><summary>docstring</summary>

```
/-- Exact index arithmetic preserves complete typed execution, including every
register and memory cell and failure. No floating-point atom is needed. -/
```
</details>

**Statement:**
```lean
specification square_index_equiv (storage : Storage) (out inp : RegionName)
    (inputStride outputStride n B : Nat) (_hB : 0 < B) :
    program storage Bool.false out inp inputStride outputStride n B ≡[assumptions]
      program storage Bool.true out inp inputStride outputStride n B
```

**Assumptions / layout contracts:**
- `_hB : 0 < B`

**Closed-form spec defs (transitive):** `program`, `assumptions`, `typedKernel`

<details><summary><code>program</code></summary>

```lean
def program (storage : Storage) (commuted : Bool) (out inp : RegionName)
    (inputStride outputStride n B : Nat) : Program where
  kernel := typedKernel storage commuted out inp inputStride outputStride n B
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

<details><summary><code>typedKernel</code></summary>

```lean
def typedKernel (storage : Storage) (commuted : Bool) : Source :=
  match storage, commuted with
  | .fp32, Bool.false => squareSource%[tl.float32, original]
  | .fp32, Bool.true => squareSource%[tl.float32, indexCommuted]
  | .fp16, Bool.false => squareSource%[tl.float16, original]
  | .fp16, Bool.true => squareSource%[tl.float16, indexCommuted]
  | .bf16, Bool.false => squareSource%[tl.bfloat16, original]
  | .bf16, Bool.true => squareSource%[tl.bfloat16, indexCommuted]
```
</details>
