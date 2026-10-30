import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.SourceOffsets
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.CandidateBackwardWhole
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.CandidateSafetyForward
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.SourceSafety
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.LayerNormLiger.CandidateSafety
open VeriTile.Triton FP.Structural Safety
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (SumPrimitive withSum)
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

private theorem append_safe (M : Algebra α) (bounds : RegionName → Nat)
    (a b : List ComputeStmt) (s t : State α)
    (safe : traceSafeList M bounds a s) (ran : run M a s=some t)
    (next : traceSafeList M bounds b t) : traceSafeList M bounds (a++b) s := by
  apply (traceSafeList_append M bounds a b s).mpr
  refine ⟨safe,?_⟩
  intro u hu
  have eq : u=t := Option.some.inj (hu.symm.trans ran)
  simpa only [eq] using next

namespace Backward
open LayerNormLiger.Backward hiding prelude body postlude Ready prelude_run body_run range_run postlude_run complete_run source_split
open LayerNormLiger.Backward.Candidate

theorem prelude_safe (M : Algebra α) (R : SumPrimitive α) (p : Parameters)
    (s : State α) (bounds : RegionName → Nat) :
    traceSafeList (withSum M R) bounds (prelude p) s := by
  simp [prelude,candidateKernel,candidateSource,backwardCandidateSource,ComputeKernel.surfaceBody,
    traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,numeric,bop,integralFloorDiv,
    numericLt,natLt,withSum,Region.cast]

theorem body_safe (M : Algebra α) (R : SumPrimitive α) (p : Parameters)
    (s t : State α) (row : Nat) (a : Acc α p) (h : Ready M p s t row a) (bounds : RegionName → Nat)
    (hx : ∀ i : Fin p.block,i.val<p.columns → row*p.xStride+i.val<bounds p.x)
    (hw : ∀ i : Fin p.block,i.val<p.columns → i.val<bounds p.weight)
    (hdy : ∀ i : Fin p.block,i.val<p.columns → row*p.dyStride+i.val<bounds p.dy)
    (hm : row<bounds p.mean) (hr : row<bounds p.rstd)
    (hdx : ∀ i : Fin p.block,i.val<p.columns → row*p.dxStride+i.val<bounds p.dx) :
    traceSafeList (withSum M R) bounds (body p) (t.setReg "_" .int [] (fun _ => (row : Int))) := by
  simp [body,candidateKernel,candidateSource,backwardCandidateSource,ComputeKernel.surfaceBody,
    traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
    safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,Option.bind,
    ComputeDType.eraseDType,FloatDType.toTileDType,FloatDType.numericDType,ofFloat,toFloat,
    numeric,bop,numericLt,natLt,Region.cast,withSum,h.mem,h.sx,h.sdx,h.sdw,h.sdb,h.sdy,h.rows,h.columns,h.group,h.one,h.pid,h.first,h.last,h.cols,h.mask,
    h.dw,h.db,h.denominator,h.xp,h.mp,h.rp,h.dxp,h.dyp,hm,hr]
  exact ⟨hx,hw,hdy,hdx⟩

theorem postlude_safe (M : Algebra α) (R : SumPrimitive α) (p : Parameters)
    (s t : State α) (row : Nat) (a : Acc α p) (h : Ready M p s t row a) (bounds : RegionName → Nat)
    (hdw : ∀ i : Fin p.block,i.val<p.columns → s.pids 0*p.dwStride+i.val<bounds p.dw)
    (hdb : ∀ i : Fin p.block,i.val<p.columns → s.pids 0*p.dbStride+i.val<bounds p.db) :
    traceSafeList (withSum M R) bounds (postlude p) t := by
  simp [postlude,candidateKernel,candidateSource,backwardCandidateSource,ComputeKernel.surfaceBody,
    traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
    safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,Option.bind,
    ComputeDType.eraseDType,FloatDType.toTileDType,FloatDType.numericDType,ofFloat,toFloat,
    numeric,bop,numericLt,natLt,Region.cast,withSum,h.mem,h.sx,h.sdx,h.sdw,h.sdb,h.sdy,h.rows,h.columns,h.group,h.one,h.pid,h.cols,h.mask,h.dw,h.db,h.denominator]
  exact ⟨hdw,hdb⟩

abbrev Bounds := SourceSafety.Backward.Bounds

theorem range_safe (M : Algebra α) (R : SumPrimitive α) (p : Parameters)
    (s t : State α) (row : Nat) (a : Acc α p) (h : Ready M p s t row a)
    (lower : start p s≤row) (bounds : RegionName → Nat) (hb : Bounds p (s.pids 0) bounds) :
    rangeIntSafe (withSum M R) bounds "_" row (stop p s) 1 (body p) t := by
  rw [rangeIntSafe_unfold,if_neg (by decide)]
  by_cases hi : row<stop p s
  · rw [if_pos (by exact_mod_cast hi : (row : Int) < (stop p s : Int))]
    obtain ⟨u,hu,ready⟩ := body_run M R p s t row a h
    refine ⟨body_safe M R p s t row a h bounds (hb.x row lower hi)
      (hb.weight (Nat.lt_of_le_of_lt lower hi)) (hb.dy row lower hi)
      (hb.mean row lower hi) (hb.rstd row lower hi) (hb.dx row lower hi),?_⟩
    rw [hu]
    simpa only [Nat.cast_add, Nat.cast_one] using range_safe M R p s u (row+1) _ ready (by omega) bounds hb
  · rw [if_neg (by exact_mod_cast hi : ¬ (row : Int) < (stop p s : Int))]
    trivial
termination_by stop p s-row
decreasing_by omega

theorem source_safe (M : Algebra α) (R : SumPrimitive α) (p : Parameters)
    (s : State α) (bounds : RegionName → Nat) (hb : Bounds p (s.pids 0) bounds) :
    TraceSafe (withSum M R) bounds (candidateKernel p) s := by
  obtain ⟨u,hu,hr⟩ := prelude_run M R p s
  obtain ⟨v,hv,k,hk⟩ := range_run M R p s u (start p s) (stop p s) (initial M p s) hr
  change traceSafeList _ _ (candidateKernel p).surfaceBody s
  rw [source_split]
  simp only [List.append_assoc]
  apply append_safe _ _ _ _ s u (prelude_safe M R p s bounds) hu
  apply append_safe _ _ _ _ u v
  · apply traceSafeList_cons_intro
    · simpa [stepSafe_unfold,safeOp_unfold,evalOp_unfold,hr.first,hr.last] using
        range_safe M R p s u (start p s) (initial M p s) hr (by rfl) bounds hb
    · intro t _
      exact traceSafeList_nil _ _ _
  · simp [run,step,evalOp_unfold,hr.first,hr.last,hv]
  · exact postlude_safe M R p s v k _ hk bounds hb.dw hb.db

end Backward
#axiomsClean forward
#axiomsClean Backward.source_safe
end
end VeriTile.Bench.Repairs.LayerNormLiger.CandidateSafety
