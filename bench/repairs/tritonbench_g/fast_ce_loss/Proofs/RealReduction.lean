import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.RealLocal
import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.Geometry

namespace VeriTile.Bench.Repairs.FastCeLoss.RealProof
open Source Finset
noncomputable section
set_option maxHeartbeats 3000000

theorem lane_sum (V B chunk : Nat) (positive : 0<B) (f : Nat → ℝ) :
    (∑ lane ∈ (univ : Finset (Fin B)).filter (fun i => chunk*B+i.val<V),
      Real.exp (f (chunk*B+lane.val))) =
    Math.chunkMass B chunk (fun j : Fin V => f j.val) := by
  unfold Math.chunkMass
  rw [← Finset.sum_filter]
  refine Finset.sum_bij (fun i hi => (⟨chunk*B+i.val,(mem_filter.mp hi).2⟩ : Fin V)) ?_ ?_ ?_ ?_
  · intro i hi
    simp only [mem_filter,mem_univ,true_and]
    apply Geometry.owner_unique positive
    · omega
    · rw [Nat.add_mul,Nat.one_mul]
      exact Nat.add_lt_add_left i.isLt _
  · intro i hi j hj equal
    apply Fin.ext
    have h := congrArg Fin.val equal
    simpa only [Fin.val_mk,Nat.add_left_cancel_iff] using h
  · intro j hj
    have div : j.val/B=chunk := (mem_filter.mp hj).2
    have value : chunk*B+j.val%B=j.val := by
      simpa only [div,Nat.mul_comm] using Nat.div_add_mod j.val B
    refine ⟨⟨j.val%B,Nat.mod_lt _ positive⟩,?_,?_⟩
    · simp only [mem_filter,mem_univ,true_and,value]
      exact j.isLt
    · apply Fin.ext
      exact value
  · intro i hi
    rfl

theorem unchunked_lse (o : Options) (s : Shape) (b : Buffers) (m : Memory (WithBot ℝ))
    (row : Nat) (x : Nat → ℝ) (vocabPositive : 0<s.vocab) (covers : s.vocab≤s.block)
    (inputs : ∀ col<s.vocab, m (addr b.logits (row*s.vocab+col))=some (x col)) :
    forwardLSE arithmetic o s b m (row,0)=
      some (Math.lse o.softcap o.scaling (fun j : Fin s.vocab => x j.val)) := by
  have blockPositive := lt_of_lt_of_le vocabPositive covers
  rw [forward_lse o s b m (row,0) x blockPositive (by simpa using vocabPositive) inputs]
  simp only [Prod.snd]
  rw [lane_sum s.vocab s.block 0 blockPositive (fun i => Math.transformed o.softcap o.scaling (x i))]
  unfold Math.lse
  congr 2
  unfold Math.chunkMass
  apply Finset.sum_congr rfl
  intro j _
  have bound : j.val<s.block := lt_of_lt_of_le j.isLt covers
  simp [Nat.div_eq_of_lt bound]

theorem chunked_lse (o : Options) (s : Shape) (b : Buffers) (m : Memory (WithBot ℝ))
    (row chunk : Nat) (x : Nat → ℝ) (block : s.block=65536) (active : chunk<s.chunks)
    (inputs : ∀ col<s.vocab, m (addr b.logits (row*s.vocab+col))=some (x col)) :
    forwardLSE arithmetic o s b m (row,chunk)=some (Real.log
      (Math.chunkMass 65536 chunk (fun j : Fin s.vocab => Math.transformed o.softcap o.scaling (x j.val)))) := by
  have bp : 0<s.block := by omega
  have starts : chunk*s.block<s.vocab := by
    rw [block]
    exact Geometry.chunk_starts (by decide) active
  rw [forward_lse o s b m (row,chunk) x bp starts inputs]
  simp only [Prod.snd]
  rw [lane_sum s.vocab s.block chunk bp (fun i => Math.transformed o.softcap o.scaling (x i))]
  rw [block]

#axiomsClean lane_sum
#axiomsClean unchunked_lse
#axiomsClean chunked_lse
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.RealProof
