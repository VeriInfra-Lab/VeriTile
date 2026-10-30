import bench.tritonbench_g.rotary_transform_ops.Proofs.TypedResult
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.RotaryTransformOps.SourceSafety
open VeriTile.Triton FP.Structural TypedExecution
open VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits (fold_writes_regs)
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
private theorem lt_twice {i h : Nat} (small : i<h) : i<2*h := by omega
private theorem add_lt_twice {i h : Nat} (small : i<h) : i+h<2*h := by omega

/-- Actual masked accesses of the complete quotation, including coefficient
loads in padded rows and both signed metadata branches. -/
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
  have hcu0 : s.varlen=Bool.true → u.pids 1<bounds "cu" := by
    intro packed
    have := hcu packed
    omega
  rw [kernel_carrier]
  by_cases hv : s.varlen = Bool.true <;> by_cases ht : s.offsetTensor = Bool.true <;>
    by_cases hi : s.interleaved = Bool.true <;> by_cases hc : s.conjugate = Bool.true
  all_goals
    simp (config := { maxSteps := 200000 }) [TraceSafe,kernel,KernelSource.source,
      KernelSource.source64_64,ComputeKernel.surfaceBody,hv,ht,hi,hc,
      Safety.traceSafeList_cons,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
      Safety.maskActive_unfold,Safety.rangeSafe_unfold,Safety.precisionContext,
      FP.Structural.run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,numeric,natLt,numericLt,numericLe,
      numericEq,bop,Region.cast,State.setReg_same,State.setReg_ne_name,State.setReg_mem,
      State.setReg_pids,State.setReg_undef,FP.Structural.store,Option.bind,
      ComputeDType.eraseDType,TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
      TileShape.insertAxis,TileShape.dropInsertedIndex,Broadcast.leftIndex,Broadcast.rightIndex,
      integralFloorDiv,integralMod,ofFloat,toFloat,FloatDType.numericDType,
      FloatDType.toTileDType,fold_writes_regs]
    simp only [ProgramModel.active,ProgramModel.dataOffset,ProgramModel.dataBase,
      ProgramModel.row,ProgramModel.sequenceLength,ProgramModel.hasCoefficient,
      ProgramModel.coefficientOffset,ProgramModel.shift,hv,ht,if_true,if_false,
      Int.natCast_mul,Int.natCast_add,Bool.false_eq_true,Bool.true_eq_false] at hx hy hcos hsin
    try simp only [hcu0,hcu,hoff,hv,ht,true_and]
    intro early
    simp only [←Int.natCast_ediv,←Int.natCast_mul,←Int.natCast_add,
      Int.toNat_natCast,Int.ofNat_lt,Nat.add_assoc,Nat.mul_assoc,
      Bool.false_eq_true,Bool.true_eq_false,if_false,if_true] at *
    repeat' (first | apply And.intro | intro)
    all_goals
      try simp only [←Int.natCast_ediv,←Int.natCast_mul,←Int.natCast_add,
        Int.toNat_natCast,Int.ofNat_lt,Nat.add_assoc,Nat.mul_assoc] at *
      first
      | apply hcos
      | apply hsin
      | apply hx
      | apply hy
      constructor <;> first
      | assumption
      | exact lt_twice (by assumption)
      | exact add_lt_twice (by assumption)
      | omega

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.RotaryTransformOps.SourceSafety
