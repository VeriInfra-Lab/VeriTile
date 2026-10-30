import bench.repairs.tritonbench_g.quantize_kv_transform.IO
import bench.repairs.tritonbench_g.quantize_kv_transform.Proofs.RealIO

namespace VeriTile.Bench.Repairs.QuantizeKvTransform.ReaderBefore
open VeriTile.Triton FP.Structural
open KernelSource
open scoped VeriTile.Triton.FP.Structural.ProgramIO
noncomputable section

/-- IO.lean displays the actual token schedule, FP32 inputs, signed destination
metadata and both INT8 code/FP32 scale outputs, with their scatter addresses. -/
abbrev quantizeIO (rest : Algebra ℝ) (p : Parameters) (count : Nat) (order : List Nat) :=
  IO.quantizeIO (KernelExecution.backend (KernelRealModel.algebra rest)
    KernelRealModel.convert KernelRealModel.equal) p count order

/- The reused mathematicalMaximum is the shared logical maximum with
M(x,0)=0 and M(x,n+1)=max (M(x,n)) |x n|. Thus its extent below is the
logical row extent; no execution or padded reduction is part of this formula. -/

/-- Both logical-row outputs: signed-byte toward-zero codes and pre-selection
scales. The maximum uses columns, not the padded source tile width. -/
def mathematical (p : Parameters) (count : Nat) (vals : IO.Inputs ℝ p count) :=
  IO.outputValues (p := p) (count := count)
    (fun t h c =>
      let x := IO.values vals t.val h.val
      if mathematicalMaximum x p.columns=0 then 0 else
        let z := x c.val/(mathematicalMaximum x p.columns/127)
        (BitVec.ofInt 8 (if z<0 then -Int.floor (-z) else Int.floor z)).toInt)
    (fun t h => mathematicalMaximum (IO.values vals t.val h.val) p.columns/127)


/-- Every actual token source is safe and succeeds. Codes and scales satisfy
independent logical-row quantization, and all exterior cells are preserved. -/
theorem kernel_real (rest : Algebra ℝ) (p : Parameters)
    (positive : 0 < p.columns) (headCover : p.heads ≤ p.height) (colCover : p.columns ≤ p.width)
    (count : Nat) (order : List Nat) (coverage : order.Perm (List.range count)) :
    Spec.Real (quantizeIO rest p count order ⊨{IO.Destinations p count} mathematical p count) := by
  simpa only [mathematical] using
    RealIO.implements rest p positive headCover colCover count order coverage

end
end VeriTile.Bench.Repairs.QuantizeKvTransform.ReaderBefore
