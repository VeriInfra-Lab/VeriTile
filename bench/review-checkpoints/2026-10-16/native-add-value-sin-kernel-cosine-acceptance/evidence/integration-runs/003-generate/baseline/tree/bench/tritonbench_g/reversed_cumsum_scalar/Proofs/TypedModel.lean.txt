import bench.tritonbench_g.reversed_cumsum.Proofs.TypedModel

/-! A concrete mathematical consistency algebra for the exact typed source.
The forward scan and sum use the library's mathematical tile operations.
Casts are identities in this witness; no IEEE or native rounding claim follows. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section

def model : Algebra (WithBot ℝ) :=
  { ReversedCumsum.TypedExecution.model with
    scan := fun _ {_} op axis direction v => some (Tile.scan op axis direction ⟨v⟩).data }

def initial : State (WithBot ℝ) where
  mem := fun r _ => if r = "s" then .mk .bf16 (some 2) else .mk .fp16 (some (-9))
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun d _ _ => ReversedCumsum.TypedExecution.paddingValue d

end
end VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.TypedExecution
