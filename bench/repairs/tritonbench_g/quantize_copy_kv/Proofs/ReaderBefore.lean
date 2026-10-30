import bench.repairs.tritonbench_g.quantize_copy_kv.Proofs.RealIO

namespace VeriTile.Bench.Repairs.QuantizeCopyKv.ReaderBefore
open VeriTile.Triton FP.Structural
open KernelSource
open scoped VeriTile.Triton.FP.Structural.ProgramIO
noncomputable section

/-- Full token-grid IO with explicit INT64 destinations, FP32 input,
INT8 output codes and FP16 scales. Version selects the two existing integer
address spellings; IO.lean displays every channel and actual invocation. -/
def quantizeIO (rest : Algebra ℝ) (p : Parameters) (count : Nat)
    (dest : Nat → Nat) (version : Bool) : ProgramIO ℝ ℝ :=
  Interface.programIO (KernelRealModel.algebra rest) KernelRealModel.convert KernelRealModel.equal
    p count dest version

/- The reused mathematicalMaximum is the shared logical maximum with
M(x,0)=0 and M(x,n+1)=max (M(x,n)) |x n|. Thus its extent below is the
logical row extent; no execution or padded reduction is part of this formula. -/

/-- Independent row maximum, scale and signed-byte quantization formula.
Real semantics erase floating rounding and retain the integer conversion. -/
def mathematical (p : Parameters) (count : Nat) (values : Interface.Inputs ℝ p count) :
    Interface.Outputs ℝ p count := Interface.result
  (fun t h c =>
    let x := Interface.payload values t h
    if mathematicalMaximum x p.width=0 then 0 else
      let z := x c/(mathematicalMaximum x p.width/127)
      (BitVec.ofInt 8 (if z<0 then -Int.floor (-z) else Int.floor z)).toInt)
  (fun t h => mathematicalMaximum (Interface.payload values t h) p.width/127)


theorem kernel_correct (rest : Algebra ℝ) (p : Parameters) (count : Nat)
    (dest : Nat → Nat) (width : 0 < p.width) (covered : p.heads ≤ p.height)
    (unique : ∀ a b, a < count → b < count → dest a = dest b → a = b) :
    Spec.Real (∀ version, quantizeIO rest p count dest version ⊨{Interface.metadataMatches p count dest} mathematical p count) := by
  intro version
  simpa only [mathematical] using
    Interface.real_implements rest p count dest version width covered unique

end
end VeriTile.Bench.Repairs.QuantizeCopyKv.ReaderBefore
