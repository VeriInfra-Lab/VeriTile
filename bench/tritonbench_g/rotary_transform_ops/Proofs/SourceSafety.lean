import bench.tritonbench_g.rotary_transform_ops.Proofs.SafetyMode00
import bench.tritonbench_g.rotary_transform_ops.Proofs.SafetyMode01
import bench.tritonbench_g.rotary_transform_ops.Proofs.SafetyMode10
import bench.tritonbench_g.rotary_transform_ops.Proofs.SafetyMode11
namespace VeriTile.Bench.Repairs.RotaryTransformOps.SourceSafety
open VeriTile.Triton FP.Structural TypedExecution
noncomputable section
variable {α : Type} [Inhabited α]
theorem source_safe (p : Ops α) (s : Shape) (u : State α)
    (cu64 off64 : Bool) (bounds : RegionName → Nat)
    (hcu : s.varlen = Bool.true → u.pids 1 + 1 < bounds "cu")
    (hoff : s.offsetTensor = Bool.true → u.pids 1 < bounds "offsets")
    (hx : ∀ lane col, ProgramModel.active s u lane col Bool.true →
      ProgramModel.dataOffset s u lane col < bounds "x")
    (hy : ∀ lane col, ProgramModel.active s u lane col Bool.true →
      ProgramModel.dataOffset s u lane col < bounds (KernelAlgebra.regionName s.output))
    (hcos : ∀ lane pair, ProgramModel.hasCoefficient s u lane pair →
      ProgramModel.coefficientOffset s u lane pair < bounds "cos")
    (hsin : ∀ lane pair, ProgramModel.hasCoefficient s u lane pair →
      ProgramModel.coefficientOffset s u lane pair < bounds "sin") :
    TraceSafe (KernelAlgebra.backend p) bounds (kernel s cu64 off64) u := by
  cases hi : s.interleaved <;> cases hc : s.conjugate
  · exact source_safe_00 p s u cu64 off64 bounds hcu hoff hx hy hcos hsin hi hc
  · exact source_safe_01 p s u cu64 off64 bounds hcu hoff hx hy hcos hsin hi hc
  · exact source_safe_10 p s u cu64 off64 bounds hcu hoff hx hy hcos hsin hi hc
  · exact source_safe_11 p s u cu64 off64 bounds hcu hoff hx hy hcos hsin hi hc

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.RotaryTransformOps.SourceSafety
