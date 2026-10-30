import bench.repairs.tritonbench_g.rotary_transform_ops.Proofs.SourceCompleteSafety

namespace VeriTile.Bench.Repairs.RotaryTransformOps.SourceSafety
open VeriTile.Triton FP.Structural TypedExecution
noncomputable section
variable {α : Type} [Inhabited α]

/-- The unchanged public access hypotheses imply safety of the entire pasted
source through its complete execution and immutable-branch proofs. -/
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
  have hout : KernelAlgebra.regionName s.output=outputRegion s := by
    cases hs : s.inplace <;> simp [KernelAlgebra.regionName,Shape.output,outputRegion,hs]
  rw [hout] at hy
  have bound : SourceSafetyTools.AccessBounds s u bounds :=
    ⟨fun lane col h => ⟨hx lane col h,hy lane col h⟩,
     fun lane pair h => ⟨hcos lane pair h,hsin lane pair h⟩⟩
  exact SourceSafetyTools.source_safe (KernelAlgebra.backend p) s bounds u bound hcu hoff cu64 off64

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.RotaryTransformOps.SourceSafety
