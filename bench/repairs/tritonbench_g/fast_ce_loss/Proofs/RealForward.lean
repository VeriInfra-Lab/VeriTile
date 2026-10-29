import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.RealReduction
import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.ForwardComplete

namespace VeriTile.Bench.Repairs.FastCeLoss.RealProof
open Source Complete Finset
noncomputable section
set_option maxHeartbeats 4000000

def labelCode {V : Nat} : Option (Fin V) → Int
  | none => -100
  | some i => i.val

theorem lse_graph_real (o : Options) (s : Shape) (b : Buffers) (entry : Memory (WithBot ℝ))
    (layout : ForwardLayout s b) (row : Nat) (x : Nat → ℝ)
    (inputs : ∀ col<s.vocab, entry (addr b.logits (row*s.vocab+col))=some (x col)) :
    lseGraph arithmetic o s b entry row=
      some (Math.lse o.softcap o.scaling (fun j : Fin s.vocab => x j.val)) := by
  by_cases single : s.chunks=1
  · rw [lseGraph,if_pos single]
    exact unchunked_lse o s b entry row x layout.positive (layout.singleCover single) inputs
  · let f : Fin s.vocab → ℝ := fun j => Math.transformed o.softcap o.scaling (x j.val)
    have lanes : (fun c : Fin s.chunks => arithmetic.load (forwardLSE arithmetic o s b entry (row,c.val)))=
        (fun c : Fin s.chunks => some (Real.log (Math.chunkMass 65536 c.val f))) := by
      funext c
      simpa only [arithmetic,id_eq,f] using chunked_lse o s b entry row c.val x
        (layout.chunkBlock single) c.isLt inputs
    rw [lseGraph,if_neg single,lanes]
    have positive : 0<s.chunks := Geometry.ceil_positive _ _ (by decide) layout.positive
    rw [host_lse_finite _ positive]
    have covered : ∀ j : Fin s.vocab, j.val/65536<s.chunks := by
      intro j
      exact Geometry.owner_bound (by decide) j.isLt
    have nonempty : ∀ c<s.chunks, ∃ j : Fin s.vocab, j.val/65536=c := by
      intro c hc
      refine ⟨⟨c*65536,Geometry.chunk_starts (by decide) hc⟩,?_⟩
      simp
    have combine := Math.chunk_lse_combination 65536 s.chunks f covered nonempty
    have sums : (∑ c : Fin s.chunks,Real.exp (Real.log (Math.chunkMass 65536 c.val f)))=
        ∑ c ∈ range s.chunks,Real.exp (Real.log (Math.chunkMass 65536 c f)) := by
      exact Fin.sum_univ_eq_sum_range (fun c : Nat => Real.exp (Real.log (Math.chunkMass 65536 c f))) s.chunks
    rw [sums,combine]
    rfl

theorem target_real (o : Options) (s : Shape) (b : Buffers) (labels : Nat → Int)
    (entry : Memory (WithBot ℝ)) (row : Nat) (x : Nat → ℝ) (k : Fin s.vocab)
    (label : labels row = (k.val : Int))
    (inputs : ∀ col<s.vocab, entry (addr b.logits (row*s.vocab+col))=some (x col)) :
    target arithmetic o s b labels entry row=some (Math.transformed o.softcap o.scaling (x k.val)) := by
  unfold target
  rw [label]
  simp only [Int.toNat_natCast,inputs _ k.isLt,arithmetic,id_eq,ite_self]
  exact transform_finite o _

theorem loss_graph_real (o : Options) (s : Shape) (b : Buffers) (labels : Nat → Int)
    (entry : Memory (WithBot ℝ)) (layout : ForwardLayout s b) (row : Nat) (x : Nat → ℝ)
    (k : Option (Fin s.vocab)) (label : labels row=labelCode k)
    (inputs : ∀ col<s.vocab, entry (addr b.logits (row*s.vocab+col))=some (x col)) :
    lossGraph arithmetic o s b labels entry row=
      some (Math.rowLoss o.softcap o.scaling k (fun j : Fin s.vocab => x j.val)) := by
  cases k with
  | none =>
    have ignored : labels row = -100 := label
    by_cases single : s.chunks=1
    all_goals simp [lossGraph,forwardLoss,ignored,single,arithmetic,Math.rowLoss]
  | some k =>
    have actual : labels row = (k.val : Int) := label
    have notIgnored : labels row ≠ -100 := by rw [actual]; omega
    have ht := target_real o s b labels entry row x k actual inputs
    have hl := lse_graph_real o s b entry layout row x inputs
    by_cases single : s.chunks=1
    · rw [lseGraph,if_pos single] at hl
      simp only [lossGraph,if_pos single,forwardLoss,if_neg notIgnored,ht,hl]
      rfl
    · simp only [lossGraph,if_neg single,if_neg notIgnored,forwardLoss,ht,hl]
      change some ((-1)*Math.transformed o.softcap o.scaling (x k.val)+Math.lse o.softcap o.scaling (fun j : Fin s.vocab => x j.val)) = _
      unfold Math.rowLoss
      congr 1
      ring

theorem denominator_real (s : Shape) (labels : Nat → Int) (items : Option ℝ) :
    denominator arithmetic s labels (items.map (fun x => (some x : WithBot ℝ)))=
      some (items.getD (count s labels : ℝ)) := by
  cases items <;> rfl

theorem scalar_graph_real (o : Options) (s : Shape) (b : Buffers) (labels : Nat → Int)
    (entry : Memory (WithBot ℝ)) (layout : ForwardLayout s b) (x : Nat → Nat → ℝ)
    (logical : Nat → Option (Fin s.vocab)) (items : Option ℝ)
    (codes : ∀ row<s.rows, labels row=labelCode (logical row))
    (inputs : ∀ row<s.rows, ∀ col<s.vocab, entry (addr b.logits (row*s.vocab+col))=some (x row col)) :
    scalarGraph arithmetic o s b labels (items.map (fun x => (some x : WithBot ℝ))) entry=
      some (Math.loss o.softcap o.scaling (items.getD (count s labels : ℝ))
        (fun row : Fin s.rows => logical row.val) (fun row col => x row.val col.val)) := by
  unfold scalarGraph
  rw [denominator_real]
  have values : (fun row : Fin s.rows => arithmetic.load (lossGraph arithmetic o s b labels entry row.val))=
      (fun row : Fin s.rows => (some (Math.rowLoss o.softcap o.scaling (logical row.val)
        (fun col : Fin s.vocab => x row.val col.val)) : WithBot ℝ)) := by
    funext row
    exact loss_graph_real o s b labels entry layout row.val (x row.val) (logical row.val)
      (codes row.val row.isLt) (inputs row.val row.isLt)
  rw [values]
  have summed : arithmetic.hostSum (fun row : Fin s.rows =>
      (some (Math.rowLoss o.softcap o.scaling (logical row.val)
        (fun col : Fin s.vocab => x row.val col.val)) : WithBot ℝ)) =
      some (∑ row : Fin s.rows,Math.rowLoss o.softcap o.scaling (logical row.val)
        (fun col : Fin s.vocab => x row.val col.val)) := by
    exact (WithBot.coe_sum (Finset.univ : Finset (Fin s.rows)) (fun row => Math.rowLoss o.softcap o.scaling (logical row.val) (fun col : Fin s.vocab => x row.val col.val))).symm
  rw [summed]
  rfl

#axiomsClean lse_graph_real
#axiomsClean loss_graph_real
#axiomsClean scalar_graph_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.RealProof
