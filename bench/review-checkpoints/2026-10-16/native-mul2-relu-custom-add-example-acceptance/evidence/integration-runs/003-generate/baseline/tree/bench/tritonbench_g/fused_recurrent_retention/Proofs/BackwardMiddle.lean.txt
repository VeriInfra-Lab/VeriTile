import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardMiddlePrimitives
import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardFirstLoop
import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardSecondLoop

/-! Actual barrier/rebase/zero execution connects the two source loops.
Both register readiness and the memory invariant are derived from the first
loop's result; no externally prepared reverse carry is assumed. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardMiddle
open VeriTile.Triton Primitives MaskedMath BackwardMiddleSource
open ForwardState (keyPointers valuePointers outputPointers)
open BackwardFirstState (keyGradientPointers)
noncomputable section
set_option maxHeartbeats 1200000
set_option linter.unusedSimpArgs false

def state (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (u : BlockState) : BlockState :=
  let u1 := u.setReg "p_q" .ptr [p.BK] (keyPointers p l r.q (p.T-1))
  let u2 := u1.setReg "p_k" .ptr [p.BK] (keyPointers p l r.k (p.T-1))
  let u3 := u2.setReg "p_do" .ptr [p.BV] (valuePointers p l r.do_ (p.T-1))
  let u4 := u3.setReg "p_v" .ptr [p.BV] (valuePointers p l r.v (p.T-1))
  let u5 := u4.setReg "p_dk" .ptr [p.BK] (keyGradientPointers p l r.dk (p.T-1))
  let u6 := u5.setReg "p_dv" .ptr [p.BV] (outputPointers p l r.dv (p.T-1))
  u6.setReg "d_h" .real [p.BK,p.BV] (BackwardSecondState.carryTile p l initial r p.T)

theorem executes (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (hasSeed swapped : Bool) (u : BlockState)
    (hu : BackwardFirstState.Ready p l initial r hasSeed p.T u) :
    stepStmts (BackwardSource.middleCode p r hasSeed swapped) u = some (state p l initial r u) := by
  rw [middle_exact]
  unfold code
  rw [stepStmts.cons_some (BackwardMiddlePrimitives.barrier u)]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (BackwardMiddlePrimitives.pointer p p.BK p.qHead p.DK l.head l.key "i_k" r.q _
      hu.headId hu.keyId))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (BackwardMiddlePrimitives.pointer p p.BK p.qHead p.DK l.head l.key "i_k" r.k _
      (by simp [hu.headId]) (by simp [hu.keyId])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (BackwardMiddlePrimitives.pointer p p.BV p.vHead p.DV l.head l.value "i_v" r.do_ _
      (by simp [hu.headId]) (by simp [hu.valueId])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (BackwardMiddlePrimitives.pointer p p.BV p.vHead p.DV l.head l.value "i_v" r.v _
      (by simp [hu.headId]) (by simp [hu.valueId])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (BackwardMiddlePrimitives.gradient_pointer p l r.dk _
      (by simp [hu.headId]) (by simp [hu.keyId]) (by simp [hu.valueId])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (BackwardMiddlePrimitives.output_pointer p l r.dv _
      (by simp [hu.headId]) (by simp [hu.keyId]) (by simp [hu.valueId])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (BackwardMiddlePrimitives.zero_carry p l initial r _)),
    stepStmts.nil]
  rfl

theorem ready (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (hasSeed : Bool) (u : BlockState)
    (hu : BackwardFirstState.Ready p l initial r hasSeed p.T u) :
    BackwardSecondState.Ready p l initial r p.T (state p l initial r u) := by
  rcases hu with ⟨hk,hv,hh,hb,hkp,hvp,hd,hq,hmk,hmv,hc⟩
  constructor <;> simp [state,hk,hv,hh,hb,hmk,hmv]

theorem memory (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (u : BlockState) : (state p l initial r u).mem = u.mem := rfl

theorem invariant (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (hasSeed : Bool) (u : BlockState)
    (h : BackwardFirstLoop.Holds p l initial r hasSeed p.T u) :
    BackwardSecondLoop.Holds p l initial r hasSeed p.T (state p l initial r u) := by
  refine ⟨ready p l initial r hasSeed u h.ready,?_,?_,?_,?_,?_,?_,?_,?_⟩
  · exact h.qmem
  · exact h.kmem
  · exact h.vmem
  · exact h.domem
  · exact h.output
  · intro s k hs hT hk; omega
  · intro s j hs hT hj; omega
  · intro region off hq hk hv
    exact h.frame region off hq

#axiomsClean executes
#axiomsClean ready
#axiomsClean memory
#axiomsClean invariant
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardMiddle
