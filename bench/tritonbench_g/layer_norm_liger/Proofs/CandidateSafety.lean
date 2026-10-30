import bench.tritonbench_g.layer_norm_liger.Proofs.CandidateBackwardWhole
import bench.tritonbench_g.layer_norm_liger.Proofs.CandidateSafetyForward
import bench.tritonbench_g.layer_norm_liger.Proofs.SourceSafety
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.LayerNormLiger.CandidateSafety
open VeriTile.Triton FP.Structural Safety
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (SumPrimitive withSum)
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

private theorem assign_any (M : Algebra α) (bounds : RegionName → Nat)
    (s : State α) (dtype : TileDType) (shape : TileShape) (name : RegName)
    (e : ComputeExpr dtype shape) (rest : List ComputeStmt)
    (safe : safeExpr M bounds s e)
    (next : ∀ v, traceSafeList M bounds rest (s.setReg name dtype shape v)) :
    traceSafeList M bounds (.assign dtype shape name e :: rest) s := by
  apply traceSafeList_cons_intro (by simpa only [stepSafe_unfold] using safe)
  intro t ht
  cases he : evalExpr M e s with
  | none => simp [step,he] at ht
  | some v =>
    have eq : s.setReg name dtype shape v=t := by simpa [step,he] using ht
    rw [← eq]
    exact next v

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
  have split : prelude p = LayerNormLiger.Backward.prelude p ++
      ((candidateKernel p).surfaceBody.drop 12).take 1 := by rfl
  rw [split]
  obtain ⟨u,hu,ready⟩ := LayerNormLiger.Backward.prelude_run M R p s
  apply append_safe _ _ _ _ s u (SourceSafety.Backward.prelude_safe M R p s bounds) hu
  simp only [candidateKernel,candidateSource,ComputeKernel.surfaceBody,List.drop,List.take]
  apply traceSafeList_cons_intro
  · simp [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold]
  · intro t _
    exact traceSafeList_nil _ _ _

theorem body_safe (M : Algebra α) (R : SumPrimitive α) (p : Parameters)
    (s t : State α) (row : Nat) (a : Acc α p) (h : Ready M p s t row a) (bounds : RegionName → Nat)
    (hx : ∀ i : Fin p.block,i.val<p.columns → row*p.xStride+i.val<bounds p.x)
    (hw : ∀ i : Fin p.block,i.val<p.columns → i.val<bounds p.weight)
    (hdy : ∀ i : Fin p.block,i.val<p.columns → row*p.dyStride+i.val<bounds p.dy)
    (hm : row<bounds p.mean) (hr : row<bounds p.rstd)
    (hdx : ∀ i : Fin p.block,i.val<p.columns → row*p.dxStride+i.val<bounds p.dx) :
    traceSafeList (withSum M R) bounds (body p) (t.setReg "_row" .nat [] (fun _ => row)) := by
  dsimp only [body,candidateKernel,candidateSource,ComputeKernel.surfaceBody]
  apply assign_any
  · simpa [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
      safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
      evalOp_unfold,Option.bind,State.setReg,numeric,bop,numericLt,natLt,Region.cast,
      FloatDType.toTileDType,FloatDType.numericDType,h.pid,h.cols,h.mask,
      h.xp,h.mp,h.rp,h.dxp,h.dyp] using hx
  · intro value0
    apply assign_any
    · simpa [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
        safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
        evalOp_unfold,Option.bind,State.setReg,numeric,bop,numericLt,natLt,Region.cast,
        FloatDType.toTileDType,FloatDType.numericDType,h.pid,h.cols,h.mask,
        h.xp,h.mp,h.rp,h.dxp,h.dyp] using hw
    · intro value1
      apply assign_any
      · simpa [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
          safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
          evalOp_unfold,Option.bind,State.setReg,numeric,bop,numericLt,natLt,Region.cast,
          FloatDType.toTileDType,FloatDType.numericDType,h.pid,h.cols,h.mask,
          h.xp,h.mp,h.rp,h.dxp,h.dyp] using hdy
      · intro value2
        apply assign_any
        · simpa [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
            safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
            evalOp_unfold,Option.bind,State.setReg,numeric,bop,numericLt,natLt,Region.cast,
            FloatDType.toTileDType,FloatDType.numericDType,h.pid,h.cols,h.mask,
            h.xp,h.mp,h.rp,h.dxp,h.dyp] using hm
        · intro value3
          apply assign_any
          · simpa [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
              safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
              evalOp_unfold,Option.bind,State.setReg,numeric,bop,numericLt,natLt,Region.cast,
              FloatDType.toTileDType,FloatDType.numericDType,h.pid,h.cols,h.mask,
              h.xp,h.mp,h.rp,h.dxp,h.dyp] using hr
          · intro value4
            apply assign_any
            · simp [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
                safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
                evalOp_unfold,Option.bind,State.setReg,numeric,bop,numericLt,natLt,Region.cast,
                FloatDType.toTileDType,FloatDType.numericDType,h.pid,h.cols,h.mask,
                h.xp,h.mp,h.rp,h.dxp,h.dyp]
            · intro value5
              apply assign_any
              · simp [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
                  safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
                  evalOp_unfold,Option.bind,State.setReg,numeric,bop,numericLt,natLt,Region.cast,
                  FloatDType.toTileDType,FloatDType.numericDType,h.pid,h.cols,h.mask,
                  h.xp,h.mp,h.rp,h.dxp,h.dyp]
              · intro value6
                apply assign_any
                · simp [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
                    safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
                    evalOp_unfold,Option.bind,State.setReg,numeric,bop,numericLt,natLt,Region.cast,
                    FloatDType.toTileDType,FloatDType.numericDType,h.pid,h.cols,h.mask,
                    h.xp,h.mp,h.rp,h.dxp,h.dyp]
                · intro value7
                  apply assign_any
                  · simp [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
                      safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
                      evalOp_unfold,Option.bind,State.setReg,numeric,bop,numericLt,natLt,Region.cast,
                      FloatDType.toTileDType,FloatDType.numericDType,h.pid,h.cols,h.mask,
                      h.xp,h.mp,h.rp,h.dxp,h.dyp]
                  · intro value8
                    apply assign_any
                    · simp [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
                        safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
                        evalOp_unfold,Option.bind,State.setReg,numeric,bop,numericLt,natLt,Region.cast,
                        FloatDType.toTileDType,FloatDType.numericDType,h.pid,h.cols,h.mask,
                        h.xp,h.mp,h.rp,h.dxp,h.dyp]
                    · intro value9
                      apply traceSafeList_cons_intro
                      · simpa [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
                          safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
                          evalOp_unfold,Option.bind,State.setReg,numeric,bop,numericLt,natLt,Region.cast,
                          FloatDType.toTileDType,FloatDType.numericDType,h.pid,h.cols,h.mask,
                          h.xp,h.mp,h.rp,h.dxp,h.dyp] using hdx
                      · intro next _
                        apply assign_any
                        · simp [safeExpr_unfold,safeCompute_unfold,safeOp_unfold]
                        · intro tailValue0
                          apply assign_any
                          · simp [safeExpr_unfold,safeCompute_unfold,safeOp_unfold]
                          · intro tailValue1
                            apply assign_any
                            · simp [safeExpr_unfold,safeCompute_unfold,safeOp_unfold]
                            · intro tailValue2
                              apply assign_any
                              · simp [safeExpr_unfold,safeCompute_unfold,safeOp_unfold]
                              · intro tailValue3
                                apply assign_any
                                · simp [safeExpr_unfold,safeCompute_unfold,safeOp_unfold]
                                · intro tailValue4
                                  apply assign_any
                                  · simp [safeExpr_unfold,safeCompute_unfold,safeOp_unfold]
                                  · intro tailValue5
                                    apply assign_any
                                    · simp [safeExpr_unfold,safeCompute_unfold,safeOp_unfold]
                                    · intro tailValue6
                                      exact traceSafeList_nil _ _ _

theorem postlude_safe (M : Algebra α) (R : SumPrimitive α) (p : Parameters)
    (s t : State α) (row : Nat) (a : Acc α p) (h : Ready M p s t row a) (bounds : RegionName → Nat)
    (hdw : ∀ i : Fin p.block,i.val<p.columns → s.pids 0*p.dwStride+i.val<bounds p.dw)
    (hdb : ∀ i : Fin p.block,i.val<p.columns → s.pids 0*p.dbStride+i.val<bounds p.db) :
    traceSafeList (withSum M R) bounds (postlude p) t := by
  have old : LayerNormLiger.Backward.Ready p s t row a := {
    mem := h.mem, pid := h.pid, first := h.first, last := h.last,
    cols := h.cols, mask := h.mask, dw := h.dw, db := h.db,
    xp := h.xp, mp := h.mp, rp := h.rp, dxp := h.dxp, dyp := h.dyp }
  have same : postlude p = LayerNormLiger.Backward.postlude p := by rfl
  rw [same]
  exact SourceSafety.Backward.postlude_safe M R p s t row a old bounds hdw hdb

/-- Source windows for exactly the executed row interval; empty loops require
no X/W/DY/statistics backing. The two final gradient windows remain separate. -/
abbrev Bounds := SourceSafety.Backward.Bounds

theorem range_safe (M : Algebra α) (R : SumPrimitive α) (p : Parameters)
    (s t : State α) (row : Nat) (a : Acc α p) (h : Ready M p s t row a)
    (lower : start p s≤row) (bounds : RegionName → Nat) (hb : Bounds p (s.pids 0) bounds) :
    rangeSafe (withSum M R) bounds "_row" row (stop p s) 1 (body p) t := by
  rw [rangeSafe_unfold,if_neg (by decide)]
  by_cases hi : row<stop p s
  · rw [if_pos hi]
    obtain ⟨u,hu,ready⟩ := body_run M R p s t row a h
    refine ⟨body_safe M R p s t row a h bounds (hb.x row lower hi)
      (hb.weight (Nat.lt_of_le_of_lt lower hi)) (hb.dy row lower hi)
      (hb.mean row lower hi) (hb.rstd row lower hi) (hb.dx row lower hi),?_⟩
    rw [hu]
    exact range_safe M R p s u (row+1) _ ready (by omega) bounds hb
  · rw [if_neg hi]
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
