import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Proofs.IOExecution
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.PublicExecution
open VeriTile.Triton FP.Structural Direct Direct.FPValues Signed.Program
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {A : Type} [Inhabited A]

theorem prelude_safe (M : Algebra A) (R : Reduction A) (h : Call)
    (batch row : Nat) (s : State A) (bounds : RegionName → Nat)
    (metadata : batch<bounds "Indices") :
    Safety.traceSafeList (machine M R) bounds (prelude h) (entry h batch row s) := by
  simpa [prelude,code,Surface.source,ComputeKernel.surfaceBody,entry,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,run,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    Region.cast,State.setReg] using metadata

theorem setup_safe (M : Algebra A) (R : Reduction A) (h : Call)
    (batch row slot : Nat) (s : State A) (metadata : (data s).indices batch=(slot:Int))
    (bounds : RegionName → Nat)
    (input : ∀ j : Fin h.BK,h.even=Bool.true ∨ j.val<h.K → batch*h.K+j.val<bounds "X") :
    Safety.traceSafeList (machine M R) bounds (setup h) (preguard M R h batch row s).val := by
  rcases h with ⟨batches,adapters,N,K,columns,slice,BK,positive,covered,inside,ev,evExact,adds,ity,wty,oty⟩
  cases ev
  all_goals
    simp (maxSteps := 500000) [setup,active,code,Surface.source,ComputeKernel.surfaceBody,
      Source.conditional?,Source.intRange?,preguard,entry,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
      Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      step,run,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      machine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,numericLt,numericLe,numericEq,natLt,bop,
      State.setReg,ofFloat,toFloat,Region.cast,
      ←Int.natCast_add,←Int.natCast_mul,-Nat.cast_add,-Nat.cast_mul]
  · intro j hj
    exact input j (Or.inr (by exact_mod_cast hj))
  · intro j
    exact input j (Or.inl rfl)

theorem body_safe (M : Algebra A) (R : Reduction A) (h : Call)
    (batch row slot : Nat) (s : State A) (metadata : (data s).indices batch=(slot:Int))
    (positiveN : 0<h.N) (bounds : RegionName → Nat)
    (weights : ∀ j : Fin h.BK,j.val<h.K → slot*(h.N*h.K)+row*h.K+j.val<bounds "W")
    (output : batch*h.columns+row+h.slice<bounds "Y") :
    Safety.traceSafeList (machine M R) bounds (body h) (firstEntry M R h batch row slot s metadata) := by
  have length := signed_length h positiveN
  have metaRead : (s.mem "Indices" batch).read .int=(slot:Int) := metadata
  rcases h with ⟨batches,adapters,N,K,columns,slice,BK,positive,covered,inside,ev,evExact,adds,ity,wty,oty⟩
  change ((N:Int)+N-1)/N=1 at length
  have nonzero : N≠0 := by change 0<N at positiveN;omega
  have length_cast : ((↑(N+N):Int)-1)/(N:Int)=1 := by simpa only [Int.natCast_add] using length
  have positive_sum : ¬((N:Int)+N<1) := by change 0<N at positiveN;omega
  cases ev <;> cases adds
  all_goals
    simp (maxSteps := 1000000) [body,active,code,Surface.source,ComputeKernel.surfaceBody,
      Source.conditional?,Source.intRange?,firstEntry,preloop,preguard,entry,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
      Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      step,run,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      machine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,numericLt,numericLe,numericEq,natLt,bop,
      State.setReg,ofFloat,toFloat,Region.cast,FP.Structural.store,
      ←Int.natCast_ediv,length,length_cast,nonzero,positive_sum,metaRead,
      ←Int.natCast_add,←Int.natCast_mul,-Nat.cast_add,-Nat.cast_mul]
  all_goals repeat' constructor
  all_goals intros
  all_goals first
    | omega
    | have zero : (‹Fin 256›).val=0 := by omega
      simpa [zero,Nat.mul_comm,Nat.add_assoc] using weights ‹Fin BK› (by omega)

#axiomsClean prelude_safe
#axiomsClean setup_safe
#axiomsClean body_safe
end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.PublicExecution
