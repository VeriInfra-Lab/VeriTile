import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.OutputGrid
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.Logical
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.OutputLogical
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGlaFwd
open scoped BigOperators
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

def inputs (a : Logical.Inputs) (attention : Nat → ℝ) : RealOutputIO.Inputs :=
  ⟨a.query,a.value,a.gate,a.history,attention⟩

theorem formula (c : OutputGrid.Shape) (a : Logical.Inputs) (attention : Nat → ℝ)
    (scale : ℝ) (h : Fin c.heads) (t : Fin c.T) (v : Fin c.V) (hbk : 0<c.BK)
    (ha : ∀ j, j<c.BT → attention (h.val*c.T*c.BT+t.val*c.BT+j)=
      Logical.attention c.T c.K c.BT a scale h.val t.val j) :
    OutputGrid.formula c (inputs a attention) scale h t v=
      Logical.output c.T c.K c.V c.BT a scale h.val t.val v.val := by
  have ht := t.isLt
  have hv := v.isLt
  have hmod := Nat.mod_lt t.val c.bt_pos
  have hcausal (j : Nat) (hj : j ≤ t.val%c.BT) : t.val/c.BT*c.BT+j<c.T := by
    have hdiv := Nat.div_add_mod' t.val c.BT
    omega
  let historyTerm := fun k =>
    a.query (Logical.featureAddress c.T c.K h.val t.val k) * scale *
      Real.exp (a.gate (Logical.featureAddress c.T c.K h.val t.val k)) *
      a.history (Geometry.historyAddress (Geometry.ceilDiv c.T c.BT) c.K c.V h.val (t.val/c.BT) k v.val)
  let padded := fun k => if k<c.K then historyTerm k else 0
  have hsum : (∑ b : Fin (Geometry.ceilDiv c.K c.BK), ∑ e : Fin c.BK, padded (b.val*c.BK+e.val)) =
      ∑ k ∈ Finset.range c.K, historyTerm k := by
    calc
      _ = ∑ b ∈ Finset.range (Geometry.ceilDiv c.K c.BK), ∑ e : Fin c.BK, padded (b*c.BK+e.val) :=
        Fin.sum_univ_eq_sum_range (fun b => ∑ e : Fin c.BK, padded (b*c.BK+e.val)) _
      _ = ∑ k ∈ Finset.range c.K, padded k :=
        SumBlocks.truncated padded c.K c.BK hbk (by intro k hk; simp [padded,Nat.not_lt.mpr hk])
      _ = _ := by apply Finset.sum_congr rfl; intro k hk; simp [padded,Finset.mem_range.mp hk]
  unfold OutputGrid.formula RealOutputIO.formula
  simp only [OutputGrid.owner_launch,OutputGrid.lane,OutputGrid.layout,inputs,Nat.div_add_mod',ht,hv,
    true_and,and_true,ite_true]
  unfold Logical.output
  apply congrArg₂ (·+·)
  · calc
      _ = ∑ b : Fin (Geometry.ceilDiv c.K c.BK), ∑ e : Fin c.BK, padded (b.val*c.BK+e.val) := by
        apply Finset.sum_congr rfl
        intro b _
        apply Finset.sum_congr rfl
        intro e _
        by_cases hk : b.val*c.BK+e.val<c.K
        · simp only [hk,ite_true,padded,historyTerm,Logical.featureAddress]
          rw [Geometry.history_source_address]
        · simp [hk,padded]
      _ = _ := hsum
  · let f := fun j : Nat =>
      (if j ≤ t.val%c.BT then attention (h.val*c.T*c.BT+t.val*c.BT+j) else 0) *
        (if t.val/c.BT*c.BT+j<c.T then
          a.value (h.val*(c.T*c.V)+(t.val/c.BT*c.BT+j)*c.V+v.val) else 0)
    change (∑ j : Fin c.BT, f j.val) = _
    rw [Fin.sum_univ_eq_sum_range f c.BT]
    dsimp only [f]
    calc
      _ = ∑ j ∈ Finset.range (t.val%c.BT+1),
          (if j ≤ t.val%c.BT then attention (h.val*c.T*c.BT+t.val*c.BT+j) else 0) *
            (if t.val/c.BT*c.BT+j<c.T then
              a.value (h.val*(c.T*c.V)+(t.val/c.BT*c.BT+j)*c.V+v.val) else 0) := by
        symm
        apply Finset.sum_subset (Finset.range_mono (by omega))
        intro j _ hj
        simp only [Finset.mem_range,not_lt] at hj
        simp [show ¬j ≤ t.val%c.BT by omega]
      _ = _ := by
        apply Finset.sum_congr rfl
        intro j hj
        have hj' : j ≤ t.val%c.BT := by simpa [Nat.lt_succ_iff] using Finset.mem_range.mp hj
        have hjBT : j<c.BT := by omega
        simp only [hj',ite_true,hcausal j hj']
        rw [ha j hjBT]
        simp only [Geometry.outputAddress,Nat.add_mul,Nat.mul_assoc]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.OutputLogical
