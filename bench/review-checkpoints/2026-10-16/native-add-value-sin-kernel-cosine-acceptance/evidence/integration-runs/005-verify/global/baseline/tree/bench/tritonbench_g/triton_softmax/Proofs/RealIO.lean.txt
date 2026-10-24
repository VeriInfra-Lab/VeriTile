import bench.tritonbench_g.triton_softmax.Proofs.Real
import VeriTile.Meta.StatementAudit

/-! Internal mathematical IO bridge used by the public typed-source contract. -/
namespace VeriTile.Bench.TritonBenchG.TritonSoftmax
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedKernelIO₁

/-- `softmax_kernel`'s masked **IO signature** — the whole kernel-specific
audit surface of the `⊨` headline:

* `inp`/`out` — which buffer is which argument (the wiring);
* `B = BLOCK_SIZE` — the row window each program owns;
* `read`/`write` — program `pid` reads its row at `pid * input_row_stride` and
  writes it at `pid * output_row_stride` (the host-side one-program-per-row
  launch convention);
* `mask` — the active lanes `j < n_cols`, **the same for every program**: the
  row prefix that actually exists in the matrix. Inactive lanes (the padding
  of `BLOCK_SIZE = min(next_power_of_2(n_cols), 1024)`) carry no obligations
  on either side.

The windows and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the active lanes. -/
def softmaxIO (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) :
    MaskedKernelIO₁ where
  kernel := softmax_kernel output_ptr input_ptr input_row_stride
    output_row_stride n_cols BLOCK_SIZE
  inp := input_ptr
  out := output_ptr
  B := BLOCK_SIZE
  read := fun pid => pid * input_row_stride
  write := fun pid => pid * output_row_stride
  mask := fun _ j => j.val < n_cols

/-- Internal real IO bridge: `softmax_kernel` implements the exact stable softmax
over the active row prefix on its masked IO signature — for every disjoint
flat placement of the two buffers, every program id whose active lanes are in
bounds, and every launch state whose active input-row lanes hold `xs`, the
translated pointer kernel terminates, every active output-row lane `j` holds
`softmaxSpec n_cols BLOCK_SIZE xs j`, and every other memory cell is
unchanged. `0 < BLOCK_SIZE` is required: the kernel's `max` reduce (like
`Finset.sup'`) is only defined on non-empty tiles. Proof: `Implements.intro`
assembles the region-model masked triple with the bridge side conditions. -/
theorem softmax_kernel_real_io
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) :
    softmaxIO output_ptr input_ptr input_row_stride output_row_stride
        n_cols BLOCK_SIZE ⊨
      fun xs i => softmaxSpec n_cols BLOCK_SIZE xs i := by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact softmax_kernel_flattenOk output_ptr input_ptr input_row_stride
      output_row_stride n_cols BLOCK_SIZE
  · intro bounds s h1 h2 _
    exact softmax_kernel_traceSafe output_ptr input_ptr input_row_stride
      output_row_stride n_cols BLOCK_SIZE hB bounds s h1 h2
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hframe⟩ := softmax_kernel_region_run output_ptr
      input_ptr input_row_stride output_row_stride n_cols BLOCK_SIZE hB s₀ xs hx
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

#axiomsClean softmax_kernel_real_io
end VeriTile.Bench.TritonBenchG.TritonSoftmax
