# Spec sheet — `bench/tritonbench_g/matrix_transpose/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/matrix_transpose/matrix_transpose.py`

## Public theorem: `transpose_index_equiv`

<details><summary>docstring</summary>

```
/-- Full float16 execution agrees, including all registers, memory, and failure.
Only natural-number address multiplication changes, so no floating atom is used. -/
```
</details>

**Statement:**
```lean
specification transpose_index_equiv (M Out : RegionName) (sx sy ox oy m d : Nat) :
    program Bool.false M Out sx sy ox oy m d ≡[assumptions]
      program Bool.true M Out sx sy ox oy m d
```

**Closed-form spec defs (transitive):** `program`, `assumptions`, `kernel`, `typedKernel`

<details><summary><code>program</code></summary>

```lean
def program (v : Bool) (M Out : RegionName) (sx sy ox oy m d : Nat) : Program where
  kernel := typedKernel v M Out sx sy ox oy m d
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

<details><summary><code>kernel</code></summary>

```
/-- Faithful 1:1 transcription of `matrix_transpose.py`'s `kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `SIZE_M: tl.constexpr` / `D_HEAD: tl.constexpr` → Lean `Nat`
  parameters. -/
```
```lean
def kernel
    (M Out : RegionName)
    (matrix_stridex matrix_stridey out_stridex out_stridey
      SIZE_M D_HEAD : Nat) :
    ComputeKernel := triton {
  size_m_arange = tl.arange(0, $(SIZE_M))
  d_head_arange = tl.arange(0, $(D_HEAD))
  matrix_ptr = M + size_m_arange[:, None] * $(matrix_stridex)
                + d_head_arange[None, :] * $(matrix_stridey)
  out_ptr = Out + d_head_arange[:, None] * $(out_stridex)
             + size_m_arange[None, :] * $(out_stridey)
  matrix = tl.load(matrix_ptr)
  tl.store(out_ptr, tl.trans(matrix))
}


/-! The wrapper allocates fp16 input and output. Both pointer tiles retain
all four supplied strides. The candidate changes only exact index multiplication. -/
syntax "transposeSource%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(transposeSource%[$storage:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let inputPtr : TSyntax `tritonStmt ←
      if version.getId == `original then
        `(tritonStmt| matrix_ptr = M + size_m_arange[:, None] * $(matrix_stridex)
          + d_head_arange[None, :] * $(matrix_stridey))
      else `(tritonStmt| matrix_ptr = M + $(matrix_stridex) * size_m_arange[:, None]
          + d_head_arange[None, :] * $(matrix_stridey))
    `(fun (M Out : RegionName)
        (matrix_stridex matrix_stridey out_stridex out_stridey SIZE_M D_HEAD : Nat) =>
      ComputeKernel.mk [M] [Out] (triton {
  tl.region M = $storage, Out = $storage
  size_m_arange = tl.arange(0, $(SIZE_M))
  d_head_arange = tl.arange(0, $(D_HEAD))
  $inputPtr
  out_ptr = Out + d_head_arange[:, None] * $(out_stridex)
    + size_m_arange[None, :] * $(out_stridey)
  matrix = tl.load(matrix_ptr)
  tl.store(out_ptr, tl.trans(matrix))
}).surfaceBody)
```
</details>

<details><summary><code>typedKernel</code></summary>

```
/-- Complete float16 source used by the Python wrapper. -/
```
```lean
def typedKernel (commuted : Bool) : Source :=
  if commuted then transposeSource%[tl.float16, indexCommuted]
  else transposeSource%[tl.float16, original]
```
</details>
