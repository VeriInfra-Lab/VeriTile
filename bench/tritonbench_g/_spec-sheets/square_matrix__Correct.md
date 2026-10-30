# Spec sheet — `bench/tritonbench_g/square_matrix/Correct.lean`

**Python source:** `bench/tritonbench_g/square_matrix/square_matrix.py`

## Public theorem: `square_typed_real_correct`

<details><summary>docstring</summary>

```
/-- Both complete typed source versions square the active input lanes. This
includes successful flat execution and the full frame outside active writes;
different input/output row strides and partial final tiles remain symbolic. -/
```
</details>

**Statement:**
```lean
specification square_typed_real_correct (storage : Storage) (commuted : Bool)
    (out inp : RegionName) (si so n B : Nat) (_hB : 0 < B) :
    VeriTile.Spec.Real (typedIO storage commuted out inp si so n B ⊨
      fun xs i => xs i * xs i)
```

**Assumptions / layout contracts:**
- `_hB : 0 < B`

**Closed-form spec defs (transitive):** `typedIO`, `squareIO`, `typedKernel`, `square_kernel`

<details><summary><code>typedIO</code></summary>

```
/-- Source-facing IO retains explicit storage in the Compute source and uses its
checked real projection for the independent mathematical contract. -/
```
```lean
def typedIO (storage : Storage) (commuted : Bool) (out inp : RegionName)
    (si so n B : Nat) : MaskedKernelIO₁ :=
  { squareIO out inp si so n B with
    kernel := (typedKernel storage commuted out inp si so n B).eraseDType
    projection := typed_supported storage commuted out inp si so n B }
```
</details>

<details><summary><code>squareIO</code></summary>

```
/-- `square_kernel`'s masked **IO signature** — the whole kernel-specific
audit surface of the `⊨` headline:

* `inp`/`out` — which buffer is which argument (the wiring);
* `B = BLOCK_SIZE` — the row window each program owns;
* `read`/`write` — program `pid` reads its row at `pid * input_row_stride` and
  writes it at `pid * output_row_stride` (the host-side one-program-per-row
  launch convention);
* `mask` — the active lanes `j < n_cols`, **the same for every program**: the
  row prefix that actually exists in the matrix. Inactive lanes (the padding
  of `BLOCK_SIZE = next_power_of_2(n_cols)`) carry no obligations on either
  side.

The windows and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the active lanes. -/
```
```lean
def squareIO (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) :
    MaskedKernelIO₁ where
  kernel := square_kernel output_ptr input_ptr input_row_stride
    output_row_stride n_cols BLOCK_SIZE
  inp := input_ptr
  out := output_ptr
  B := BLOCK_SIZE
  read := fun pid => pid * input_row_stride
  write := fun pid => pid * output_row_stride
  mask := fun _ j => j.val < n_cols
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

<details><summary><code>square_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `square_matrix.py`'s `square_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter. -/
```
```lean
def square_kernel
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  row_start_ptr = input_ptr + row_idx * $(input_row_stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  input_ptrs = row_start_ptr + col_offsets
  row = tl.load(input_ptrs, mask=col_offsets < $(n_cols), other=-float("inf"))
  square_output = row * row
  output_row_start_ptr = output_ptr + row_idx * $(output_row_stride)
  output_ptrs = output_row_start_ptr + col_offsets
  tl.store(output_ptrs, square_output, mask=col_offsets < $(n_cols))
}


/-! Storage is explicit in the source-facing family. The candidate commutes
the integer row/stride product before pointer addition; floating multiplication,
negative-infinity padding and both masks are unchanged. -/
syntax "squareSource%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(squareSource%[$storage:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let rowStart : TSyntax `tritonStmt ←
      if version.getId == `original then
        `(tritonStmt| row_start_ptr = input_ptr + row_idx * $(input_row_stride))
      else `(tritonStmt| row_start_ptr = input_ptr + $(input_row_stride) * row_idx)
    `(fun (output_ptr input_ptr : RegionName)
        (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) =>
      ComputeKernel.mk [input_ptr] [output_ptr] (triton {
  tl.region input_ptr = $storage, output_ptr = $storage
  row_idx = tl.program_id(0)
  $rowStart
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  input_ptrs = row_start_ptr + col_offsets
  row = tl.load(input_ptrs, mask=col_offsets < $(n_cols), other=-float("inf"))
  square_output = row * row
  output_row_start_ptr = output_ptr + row_idx * $(output_row_stride)
  output_ptrs = output_row_start_ptr + col_offsets
  tl.store(output_ptrs, square_output, mask=col_offsets < $(n_cols))
}).surfaceBody)

inductive Storage where
  | fp32 | fp16 | bf16
  deriving DecidableEq
```
</details>
