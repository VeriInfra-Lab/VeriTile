import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.Grid
import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.AddressArithmetic

/-! The explicit int64 loop start/bound/step and widened program-base addition
refine the natural control indices. Representability follows from ordinary
allocated element counts; no i32 safe-domain precondition is introduced. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.NativeControl
open Address

theorem count_representable {count cells : Nat} (hcell : 1<cells)
    (halloc : count*cells≤2^63) : count<2^63 := by
  by_cases hz : count=0
  · subst count; norm_num
  · have hp : 0<count := by omega
    have hl : count<count*cells := by nlinarith
    exact lt_of_lt_of_le hl halloc

theorem linear_exact {base lane total : Nat} (ha : base+lane<total)
    (halloc : total≤2^63) :
    (word lane + word base).toInt=(base+lane : Nat) := by
  rw [show word lane+word base=word (base+lane) by
    simp [word,←BitVec.ofNat_add,Nat.add_comm]]
  exact word_exact (lt_of_lt_of_le ha halloc)

/-- All three source range arguments are wide. Each final increment is still
representable, including the increment that reaches the excluded stop. -/
theorem wide_range (NT : Nat) (hNT : NT<2^63) :
    (word 0).toInt=0 ∧ (word NT).toInt=(NT : Int) ∧ (word 1).toInt=1 ∧
    (∀ i, i≤NT → ((word i).toInt<(word NT).toInt ↔ i<NT)) ∧
    (∀ i, i<NT → (word i+word 1).toInt=(i+1 : Nat)) := by
  refine ⟨word_exact (by norm_num),word_exact hNT,word_exact (by norm_num),?_,?_⟩
  · intro i hi
    rw [word_exact (lt_of_le_of_lt hi hNT),word_exact hNT]
    omega
  · intro i hi
    rw [show word i+word 1=word (i+1) by simp [word,←BitVec.ofNat_add]]
    exact word_exact (lt_of_le_of_lt (by omega) hNT)

/-- Nonempty recurrent histories contain many cells per chunk (BK,BV≥16).
An ordinary allocated history therefore represents the wide loop bound. -/
theorem history_range (heads NT K V : Nat) (hheads : 0<heads) (hk : 1<K) (hv : 0<V)
    (halloc : heads*NT*K*V≤2^63) : NT<2^63 := by
  apply count_representable (cells := heads*K*V)
  · have hh : 1≤heads := hheads
    have hvv : 1≤V := hv
    have hm : K≤heads*K := by nlinarith
    nlinarith
  · simpa [Nat.mul_comm,Nat.mul_left_comm,Nat.mul_assoc] using halloc

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkGatedAttention.NativeControl
