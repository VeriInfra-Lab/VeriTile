# Spec sheet — `bench/tritonbench_g/rmsnorm_triton/Correct.lean`

**Python source:** `bench/tritonbench_g/rmsnorm_triton/rmsnorm_triton.py`

## Public theorem: `rmsnorm_full_output_summary`

<details><summary>docstring</summary>

```
/-- **Full output summary**: the surface lowers to the algorithm layer, and the
masked store realizes the genuine multi-block RMS-norm closed form at every
global lane `k < N_SIZE`. Holds for arbitrary `N_SIZE` — no
`N_SIZE ≤ BLOCK_N_SIZE` hypothesis. -/
```
</details>

**Statement:**
```lean
specification rmsnorm_full_output_summary
    (x w o : RegionName) (sxb sxm sxk srw sob som sok N B : Nat) (eps : ℝ)
    (s : BlockState) (hB : 0 < B) (hNpos : 0 < N)
    (hox : o ≠ x) (how : o ≠ w)
    (hsok : 0 < sok) :
    (∃ alg, (VeriTile.Bench.TritonBenchG.RmsnormTriton.rmsnorm_triton x w o
        sxb sxm sxk srw sob som sok N B eps).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := VeriTile.Bench.TritonBenchG.RmsnormTriton.rmsnorm_triton x w o
        sxb sxm sxk srw sob som sok N B eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin N => True)
        (fun k => (o, outOff s sob som sok k.val)))
      (expected := fun k : Fin N => rmsSpecFull s x w sxb sxm sxk srw N eps k.val)
```

**Assumptions / layout contracts:**
- `hB : 0 < B`
- `hNpos : 0 < N`
- `hox : o ≠ x`
- `how : o ≠ w`
- `hsok : 0 < sok`

**Closed-form spec defs (transitive):** `rmsnorm_triton`

<details><summary><code>rmsnorm_triton</code></summary>

```
/-- Faithful transcription of `rmsnorm_triton.py`'s `rmsnorm_triton`.

Allowed mechanical Lean-syntax-only changes:
- Python `N_SIZE: tl.constexpr` / `eps: tl.constexpr` / `BLOCK_N_SIZE: tl.constexpr`
  -> Lean `Nat` / `ℝ` parameters. -/
```
```lean
def rmsnorm_triton
    (x_ptr rms_w_ptr output_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k : Nat)
    (N_SIZE BLOCK_N_SIZE : Nat) (eps : ℝ) :
    ComputeKernel := triton {
  pid_batch = tl.program_id(0)
  pid_m = tl.program_id(1)
  offs_m = pid_batch * $(stride_x_batch) + pid_m * $(stride_x_m)
  block_N = tl.arange(0, $(BLOCK_N_SIZE))
  var = tl.zeros([$(BLOCK_N_SIZE)], tl.float32)
  for block_n_start_idx in range(0, $(N_SIZE), $(BLOCK_N_SIZE)) {
    offs_n = block_n_start_idx + block_N
    x_ptr_mask = offs_n < $(N_SIZE)
    x = tl.load(x_ptr + offs_m + offs_n * $(stride_x_k), mask=x_ptr_mask, other=0.0)
    var += (x).to(tl.float32) * (x).to(tl.float32)
  }
  var = tl.sum(var, axis=0) / $(N_SIZE)
  rstd = tl.math.rsqrt(var + $(eps))
  for block_n_start_idx in range(0, $(N_SIZE), $(BLOCK_N_SIZE)) {
    offs_n = block_n_start_idx + block_N
    x_ptr_mask = offs_n < $(N_SIZE)
    rms_w = tl.load(rms_w_ptr + offs_n * $(stride_rms_w), mask=x_ptr_mask)
    x = tl.load(x_ptr + offs_m + offs_n * $(stride_x_k), mask=x_ptr_mask, other=0.0).to(tl.float32)
    x_hat = x * rstd
    out = x_hat * rms_w
    out_off = pid_batch * $(stride_out_batch) + pid_m * $(stride_out_m) +
      offs_n * $(stride_out_k)
    tl.store(output_ptr + out_off, out, mask=x_ptr_mask)
  }
}


open Lean
syntax "rmsnormPowiTyped%" "[" tritonDType "," tritonDType "," tritonDType "]" : term
set_option hygiene false in
macro_rules
  | `(rmsnormPowiTyped%[$input:tritonDType, $weight:tritonDType, $output:tritonDType]) => do
    let wide := match weight with
      | `(tritonDType| tl.float64) => Bool.true
      | _ => Bool.false
    let same := match output with
      | `(tritonDType| tl.float64) => wide
      | `(tritonDType| tl.float32) => !wide
      | _ => Bool.false
    let store : TSyntax `tritonStmt ← if same then
      `(tritonStmt| tl.store(output_ptr + out_off, out, mask=x_ptr_mask))
      else `(tritonStmt| tl.store(output_ptr + out_off, out.to($output), mask=x_ptr_mask))
    `(fun (x_ptr rms_w_ptr output_ptr : RegionName)
        (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) =>
      ComputeKernel.mk [x_ptr, rms_w_ptr] [output_ptr] (triton {
  tl.region x_ptr = $input, rms_w_ptr = $weight, output_ptr = $output
  pid_batch = tl.program_id(0)
  pid_m = tl.program_id(1)
  offs_m = pid_batch * $(sxb) + pid_m * $(sxm)
  block_N = tl.arange(0, $(B))
  var = tl.zeros([$(B)], tl.float32)
  for block_n_start_idx in range(0, $(N), $(B)) {
    offs_n = block_n_start_idx + block_N
    x_ptr_mask = offs_n < $(N)
    x = tl.load(x_ptr + offs_m + offs_n * $(sxk), mask=x_ptr_mask, other=0.0)
    var += tl.extra.cuda.libdevice.pow((x).to(tl.float32), 2)
  }
  var = tl.sum(var, axis=0) / $(N)
  rstd = tl.math.rsqrt(var + $(eps))
  for block_n_start_idx in range(0, $(N), $(B)) {
    offs_n = block_n_start_idx + block_N
    x_ptr_mask = offs_n < $(N)
    rms_w = tl.load(rms_w_ptr + offs_n * $(sw), mask=x_ptr_mask)
    x = tl.load(x_ptr + offs_m + offs_n * $(sxk), mask=x_ptr_mask, other=0.0).to(tl.float32)
    x_hat = x * rstd
    out = x_hat * rms_w
    out_off = pid_batch * $(sob) + pid_m * $(som) +
      offs_n * $(sok)
    $store
  }
}).surfaceBody)

/-- Independent input, weight and output storage; the wrapper uses output=input. -/
inductive Storage where
  | fp32 | fp64 | fp16 | bf16 | f8e4 | f8e5
  deriving DecidableEq, Repr
```
</details>

## Public theorem: `rmsnorm_full_io_correctness`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` streaming headline (wave-5 S3 per-step emit genre).** For
every rounding model `R`, the faithful `rmsnorm_triton` surface implements,
on its `StreamEmitMasked2DKernelIO₂` signature, the **ideal ℝ two-pass
RMS-norm** over the streamed tiles: emitted window `(t, j)` holds
`x[t,j] · (√(Σ_guarded x²/N_SIZE + eps))⁻¹ · w[t,j]`, where the guarded
double sum folds the *entire* `x` stream — the spec `f` is exact real
arithmetic. The kernel has **zero rounding events** (loads, both passes'
arithmetic and the per-step stores are all at `.real`; the erased
`.to(tl.float32)`s are `.real → .real`), so the skin's boundary
quantization degenerates: the readback contract's `R.round .real` is the
identity by the model's defining `round_real`, and the `.real` in-loop
stores are exact under `execR R` — the ∀-`R` face holds via the
`RoundingModel` `.real` identity fields, not as a `.triv` special case.

Layer map: both passes are cast-free, so under `execR R` they collapse
verbatim onto the exact stepper and the proven
`preLoop` / `varInv` / `postStep` / `wbInv` invariant stack above is reused
unchanged; the `⊨[R]` face adds the `TraceSafeR` walk, the per-cell memory
frame (`rms_wbBody_step_frame`, the `mem` twin of `wbStep`), and the
stream-lane spec bridge (`sum_sq_mean` re-blocking `k ↔ (k/B, k%B)`).

All four hypotheses are truth-forced (exactly the exact headline
`rmsnorm_full_output_summary`'s side conditions):

* `hB : 0 < BLOCK_N_SIZE` — both loops step by `BLOCK_N_SIZE`
  (`range(0, N_SIZE, BLOCK_N_SIZE)`); at `BLOCK_N_SIZE = 0` and
  `N_SIZE > 0` neither `forRange` advances, `execR` cannot terminate the
  way the invariant stack requires, and the step index `i / B` is
  meaningless. It holds for every real launch.
* `hox : o ≠ x`, `how : o ≠ w` — pass 2 stores into `out` **between** its
  re-reads of `x` and `rms_w`; if `out` aliased either input, later blocks
  would re-read already-overwritten values and the closed form would be
  false.
* `hsok : 0 < stride_out_k` — the per-lane write window
  `pid₀·sob + pid₁·som + k·sok` is injective over global lanes only when
  the output column stride is nonzero (`outOff_injective_of_pos`); with
  `sok = 0` all lanes collide and the per-lane readback would be
  last-writer-wins. Always true for a real tensor.

Relation to the exact surface: the exact headline
`rmsnorm_full_output_summary` (`Realizes_without_Rounding`) above is
retained unchanged; this `⊨[R]` face restates the same RMS-norm content on
the streaming emit skin, for every `R` at once (at the `.real` grid the two
faces carry the same exact cell). Both faces are kept per the
rounding-as-default doctrine. -/
```
</details>

**Statement:**
```lean
specification rmsnorm_full_io_correctness (R : RoundingModel)
    (x w o : RegionName) (sxb sxm sxk srw sob som sok N B : Nat) (eps : ℝ)
    (hB : 0 < B) (hox : o ≠ x) (how : o ≠ w) (hsok : 0 < sok) :
    rmsnormKernelIO x w o sxb sxm sxk srw sob som sok N B eps ⊨[R]
      fun _ _ xs ws t j => rmsStreamSpec N B eps xs ws t j
```

**Assumptions / layout contracts:**
- `hB : 0 < B`
- `hox : o ≠ x`
- `how : o ≠ w`
- `hsok : 0 < sok`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `typed_real_correct`

<details><summary>docstring</summary>

```
/-- Every modeled floating-storage profile, for each complete source variant,
implements the independent full-row RMS formula. The theorem includes access
safety, successful flattened execution and preservation of every cell outside
the active output window.
Integer native power is linked to the historical square model only over ℝ. -/
```
</details>

**Statement:**
```lean
specification typed_real_correct (input weight output : Storage) (v : Bool)
    (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ)
    (hB : 0 < B) (hOX : O ≠ X) (hOW : O ≠ W) (hsok : 0 < sok) :
    ExactIO (typedIO input weight output v X W O sxb sxm sxk sw sob som sok N B eps)
      (fun _ _ xs ws t j => rmsStreamSpec N B eps xs ws t j)
```

**Assumptions / layout contracts:**
- `hB : 0 < B`
- `hOX : O ≠ X`
- `hOW : O ≠ W`
- `hsok : 0 < sok`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.
