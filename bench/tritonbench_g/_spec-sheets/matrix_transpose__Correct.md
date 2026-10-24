# Spec sheet — `bench/tritonbench_g/matrix_transpose/Correct.lean`

**Python source:** `bench/tritonbench_g/matrix_transpose/matrix_transpose.py`

## Public theorem: `transpose_typed_real_correct`

<details><summary>docstring</summary>

```
/-- Both float16 source versions transpose the input window, terminate with
successful flat execution, and preserve every cell outside the output window.
Output-address injectivity is necessary for a per-lane result. -/
```
</details>

**Statement:**
```lean
specification transpose_typed_real_correct (v : Bool) (M Out : RegionName)
    (sx sy ox oy m d : Nat)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [d, m] => outAddr ox oy idx)) :
    VeriTile.Spec.Real (typedIO v M Out sx sy ox oy m d ⊨ fun _ _ xs j => xs j)
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun idx : TileIndex [d, m] => outAddr ox oy idx)`

**Closed-form spec defs (transitive):** `typedIO`, `transposeIO`, `kernel`, `typedKernel`

<details><summary><code>typedIO</code></summary>

```
/-- The full typed source retains the independently declared transposed input
window and strided output window under its checked mathematical projection. -/
```
```lean
def typedIO (v : Bool) (M Out : RegionName) (sx sy ox oy m d : Nat) : Masked2DKernelIO₁ :=
  { transposeIO M Out sx sy ox oy m d with
    kernel := (typedKernel v M Out sx sy ox oy m d).eraseDType
    projection := typed_supported v M Out sx sy ox oy m d }
```
</details>

<details><summary><code>transposeIO</code></summary>

```
/-- `kernel`'s **IO signature** — the whole kernel-specific audit surface of
the `⊨` headline:

* `inp`/`out` — which buffer is which argument (`M`, `Out`);
* `B = D_HEAD * SIZE_M` — the output tile flattened row-major into lanes,
  lane `j` = logical output cell `(j / SIZE_M, j % SIZE_M)` (`laneIdx`);
* `read` — lane `j`'s **transposed** source address
  `m * matrix_stridex + d * matrix_stridey`;
* `write` — lane `j`'s output address `d * out_stridex + m * out_stridey`;
* `mask` — every lane (the kernel's load and store are both unmasked), so
  `writeMask` keeps the struct default.

The grid is `(1,)`, so no field depends on either program id. -/
```
```lean
def transposeIO (M Out : RegionName)
    (matrix_stridex matrix_stridey out_stridex out_stridey
      SIZE_M D_HEAD : Nat) :
    Masked2DKernelIO₁ where
  kernel := kernel M Out matrix_stridex matrix_stridey out_stridex out_stridey
    SIZE_M D_HEAD
  inp := M
  out := Out
  B := D_HEAD * SIZE_M
  read := fun _ _ j =>
    matrixAddr matrix_stridex matrix_stridey (Lane2D.decode j)
  write := fun _ _ j =>
    outAddr out_stridex out_stridey (Lane2D.decode j)
  mask := fun _ _ _ => True
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
