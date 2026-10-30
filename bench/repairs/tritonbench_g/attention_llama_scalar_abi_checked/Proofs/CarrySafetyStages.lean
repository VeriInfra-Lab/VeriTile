import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.SafetyStages
import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.CarryExecution

/-! Constructive execution of the actual carry-first candidate. Source.Term,
Carries, Context, all input views, coordinates and frames are shared with the
original; only the normalizer addition order changes in the source graph. -/
namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.CarrySource
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open Source Safety
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem entry_safe (c : Config) (s : State (Term α)) (scale : Term α)
    (bounds : RegionName→Nat) (hb : AccessBounds c s bounds) :
    Safety.traceSafeList algebra bounds (parts c).before (bindScale s scale) := by
  rw [before_eq]
  exact Source.entry_safe c s scale bounds hb

theorem finish_safe (c : Config) (s t : State (Term α)) (scale : Term α) (n : Nat)
    (a : Carries α c.D) (h : Context c s scale n a t)
    (bounds : RegionName→Nat) (hb : AccessBounds c s bounds) :
    Safety.traceSafeList algebra bounds (parts c).after t := by
  rw [after_eq]
  exact Source.finish_safe c s t scale n a h bounds hb

/-- Safety of every actual K/V load, with the exact loop-state pointer carry. -/
theorem iteration_safe (c : Config) (s t : State (Term α)) (scale : Term α) (n : Nat)
    (a : Carries α c.D) (h : Context c s scale n a t) (bytes : ReachedBytes c s)
    (bounds : RegionName→Nat) (hb : AccessBounds c s bounds) :
    traceSafeList algebra bounds (parts c).body (t.setReg "start_n" .nat [] (fun _=>n*64)) := by
  have hk:=hb.key n
  have hv:=hb.value n
  have kbytes : ∀ (r : Fin 64) (d : Fin c.D), c.fp8=Bool.true →
      Float8E5Bits.isInt8 (if n*64+r.val<c.N then
        (s.mem c.K (pointers c s c.K c.skz c.skh c.skn c.skk n (r,d,())).2).read .int else 0)=Bool.true :=
    fun r d hf=>masked_key_cell c s bytes n (r,d,()) hf
  have vbytes : ∀ (r : Fin 64) (d : Fin c.D), c.fp8=Bool.true →
      Float8E5Bits.isInt8 (if n*64+r.val<c.N then
        (s.mem c.V (pointers c s c.V c.svz c.svh c.svk c.svn n (r,d,())).2).read .int else 0)=Bool.true :=
    fun r d hf=>masked_value_cell c s bytes n (r,d,()) hf
  cases hf : c.fp8 <;> cases hc : c.causal
  all_goals
    simp [parts,kernel,hf,hc,TypedSource.carryFirstHalf,TypedSource.carryFirstBytes,
      ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Option.map,Option.getD,
      traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,
      safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,algebra,
      ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,
      bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
      h.scalar,h.maximum,h.denominator,h.accumulator,h.query,h.rows,h.columns,
      h.batch,h.head,h.key,h.value,h.memory,h.pids,
      kbytes,vbytes,
      Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def,TileShape.indexToList]
  all_goals simpa only [Prod.forall] using And.intro hk hv

#axiomsClean entry_safe
#axiomsClean iteration_safe
#axiomsClean finish_safe
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.CarrySource
