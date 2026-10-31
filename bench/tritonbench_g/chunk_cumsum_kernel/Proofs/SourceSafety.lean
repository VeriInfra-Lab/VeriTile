import bench.tritonbench_g.chunk_cumsum_kernel.Proofs.Rewrite
import VeriTile.Triton.Float.StructuralSafetyLoops

namespace VeriTile.Bench.Repairs.ChunkCumsumKernel.SourceSafety
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

abbrev source (v : Bool) (input output : RegionName) (T BT : Nat) :=
  if v then sumLeftKernel .fp32 .fp32 input output T BT else repairedKernel .fp32 .fp32 input output T BT
abbrev body (v : Bool) (input output : RegionName) (T BT : Nat) :=
  if v then (Rewrite.candidateParts input output T BT).body else (FPIndexBridge.parts input output T BT).body

private theorem body_safe {α : Type} [Inhabited α] (ops : FPRow.Ops α) (v : Bool)
    (input output : RegionName) (T BT pid index : Nat) (bounds : RegionName → Nat)
    (s : State α) (ready : FPRow.Ready pid s)
    (hi : ∀ j,j<T → pid*T+j<bounds input)
    (ho : ∀ j,j<T → pid*T+j<bounds output) :
    Safety.traceSafeList (engine ops.algebra) bounds (body v input output T BT)
      (s.setReg "i_t" .int [] (fun _ => (index:Int))) := by
  obtain ⟨hp,hz⟩ := ready
  obtain ⟨carry,hz⟩ := Option.isSome_iff_exists.mp hz
  cases v <;>
    simp (config := {maxSteps := 300000}) [body,FPIndexBridge.parts,FPRow.parts,Rewrite.candidateParts,
      repairedKernel,sumLeftKernel,ComputeKernel.surfaceBody,Source.intRange?,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
      Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,FPRow.Ops.algebra,
      ComputeDType.eraseDType,numeric,numericLt,bop,ofFloat,toFloat,
      TileShape.axisDim,Region.cast,State.setReg,hp,hz,TileShape.indexToList]
  all_goals
    constructor
    · intro i h
      exact hi _ h
    · intro i h
      exact ho _ h

private theorem body_ready {α : Type} [Inhabited α] (ops : FPRow.Ops α) (v : Bool)
    (input output : RegionName) (T BT pid index : Nat) (s : State α)
    (ready : FPRow.Ready pid s) :
    ∃ t,run (engine ops.algebra) (body v input output T BT)
      (s.setReg "i_t" .int [] (fun _ => (index:Int)))=some t ∧ FPRow.Ready pid t := by
  cases v with
  | «false» => exact Rewrite.original_body ops input output T BT pid index s ready
  | «true» =>
    obtain ⟨hp,hz⟩ := ready
    obtain ⟨carry,hz⟩ := Option.isSome_iff_exists.mp hz
    simp (config := {maxSteps := 300000}) [body,Rewrite.candidateParts,sumLeftKernel,
      ComputeKernel.surfaceBody,Source.intRange?,run,step,evalExpr,
      evalComputeOp,evalOp_unfold,evalOffsets,store,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,FPRow.Ops.algebra,
      ComputeDType.eraseDType,numeric,numericLt,bop,ofFloat,toFloat,
      TileShape.axisDim,Region.cast,State.setReg,hp,hz,TileShape.indexToList,FPRow.Ready]

private theorem loop_safe {α : Type} [Inhabited α] (ops : FPRow.Ops α) (v : Bool)
    (input output : RegionName) (T BT pid q stop : Nat) (bounds : RegionName → Nat)
    (s : State α) (ready : FPRow.Ready pid s)
    (hi : ∀ j,j<T → pid*T+j<bounds input)
    (ho : ∀ j,j<T → pid*T+j<bounds output) :
    Safety.rangeIntSafe (engine ops.algebra) bounds "i_t" q stop 1 (body v input output T BT) s := by
  rw [Safety.rangeIntSafe_unfold,if_neg (by decide : ¬(1:Nat)=0)]
  by_cases active : q<stop
  · rw [if_pos (by exact_mod_cast active)]
    refine ⟨body_safe ops v input output T BT pid q bounds s ready hi ho,?_⟩
    obtain ⟨t,ran,ht⟩ := body_ready ops v input output T BT pid q s ready
    rw [ran]
    exact_mod_cast loop_safe ops v input output T BT pid (q+1) stop bounds t ht hi ho
  · rw [if_neg (by exact_mod_cast active)]
    trivial
termination_by stop-q
 decreasing_by omega

/-- Both actual FP32 sources are access-safe for every payload and every
unspecified tail. Safety imposes no numerical carry condition. -/
theorem trace_safe {α : Type} [Inhabited α] (ops : FPRow.Ops α) (v : Bool)
    (input output : RegionName) (T BT : Nat) (bounds : RegionName → Nat) (s : State α)
    (hi : ∀ j,j<T → s.pids 0*T+j<bounds input)
    (ho : ∀ j,j<T → s.pids 0*T+j<bounds output) :
    TraceSafe (engine ops.algebra) bounds (source v input output T BT) s := by
  have hsplit : (source v input output T BT).surfaceBody =
      (FPIndexBridge.parts input output T BT).before ++
      [.forRangeInt "i_t" (.constInt 0) (.constInt ((T+BT-1)/BT : Nat)) (.constNat 1)
        (body v input output T BT)] := by cases v <;> rfl
  unfold TraceSafe
  rw [hsplit,Safety.traceSafeList_append]
  constructor
  · simp [FPIndexBridge.parts,FPRow.parts,repairedKernel,ComputeKernel.surfaceBody,Source.intRange?,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeCompute_unfold,Safety.safeOp_unfold,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind]
  · intro t ht
    obtain ⟨first,pre,ready,_⟩ := FPRow.prelude ops input output T BT s
    rw [FPIndexBridge.before] at ht
    have eq := Option.some.inj (pre.symm.trans ht)
    subst t
    simp [Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeOp_unfold,evalOp_unfold]
    constructor
    · simpa using loop_safe ops v input output T BT (s.pids 0) 0 ((T+BT-1)/BT)
        bounds first ready hi ho
    · split <;> trivial

#axiomsClean trace_safe
end
end VeriTile.Bench.Repairs.ChunkCumsumKernel.SourceSafety
