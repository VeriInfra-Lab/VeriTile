import bench.tritonbench_g.attention_fwd_triton3.Proofs.SourceSafetySteps

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.SourceSafety
open VeriTile Triton FP.Structural Safety FP.ProfiledRewrite
open VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
open VeriTile.Bench.Repairs.ContextAttnMistral
open Reference
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
attribute [local simp 1100] FP.Structural.eval_truncDiv

theorem window_safe (c : Call) (o : Ops α) (bounds : RegionName → Nat)
    (initialState s : State α) (_live : Started c o (inputs initialState) s) :
    traceSafeList (engine o.algebra) bounds ((parts c).before.drop 21) s := by
  prove_memory_free

theorem epilogue_safe (c : Call) (o : Ops α) (bounds : RegionName → Nat)
    (a : Inputs α) (old : Carry c α) (s : State α) (live : Matches c o a old s)
    (backed : PreludeBounds c a bounds) :
    traceSafeList (engine o.algebra) bounds (parts c).after s := by
  rcases live with ⟨h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,h15,h16⟩
  rcases backed with ⟨query,center,denominator,output⟩
  have ho : ∀ (r : Fin c.BM) (d : Fin c.D),
      (optr c a).address (TileShape.indexToList [c.BM,c.D] (r,d,PUnit.unit))<bounds (optr c a).region := by
    simpa only [block_bounds_2d,BlockPtr.inBounds_nil_checkedAxes,forall_true_left] using output
  by_cases hf : c.finish=0
  all_goals
    simp (config := {maxSteps := 500000}) [parts,Kernels.source,Quoted.kernel,
      DSL.CallValue.natural,DSL.CallValue.integer,DSL.CallValue.fp32,
      sourceKernel,kernelCSEKernel,ComputeKernel.withScalarArguments,ScalarArgument.statement,ScalarArgument.fp32,
      ComputeKernel.surfaceBody,Source.intRange?,traceSafeList_unfold,stepSafe_unfold,
      safeExpr_unfold,safeCompute_unfold,safeOp_unfold,safeAddress_unfold,safeMask_unfold,
      addressSafe_unfold,maskActive_unfold,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,ComputeDType.eraseDType,State.setReg,engine,Algebra.withDefaultPrecision,
      FPBackend.Ops.algebra,resolvePrecision,numeric,numericLt,numericLe,numericEq,
      natLt,natLe,ofFloat,toFloat,Region.cast,
      h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,h15,h16,hf,
      store,bop,Broadcast.leftIndex,Broadcast.rightIndex,State.scatter_regs]
  · exact ⟨denominator (Or.inr hf),center,ho⟩
  · exact ⟨center,ho⟩

theorem prelude_safe (c : Call) (o : Ops α) (bounds : RegionName → Nat) (s : State α)
    (backed : PreludeBounds c (inputs s) bounds) :
    traceSafeList (engine o.algebra) bounds (parts c).before s := by
  obtain ⟨headerState,hheader,headerLive,headerMemory⟩ := header c o s
  obtain ⟨startedState,hstarted,startedLive,startedMemory⟩ :=
    initialization c o s headerState headerLive headerMemory
  have split : (parts c).before=(parts c).before.take 17 ++
      (((parts c).before.drop 17).take 4 ++ (parts c).before.drop 21) := rfl
  rw [split]
  exact append_of_run (header_safe c o bounds s) hheader
    (append_of_run (initialization_safe c o bounds s headerState headerLive backed) hstarted
      (window_safe c o bounds s startedState startedLive))

#axiomsClean prelude_safe
#axiomsClean epilogue_safe
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.SourceSafety
