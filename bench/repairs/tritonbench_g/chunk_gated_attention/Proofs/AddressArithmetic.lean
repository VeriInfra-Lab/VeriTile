import bench.repairs.tritonbench_g.chunk_gated_attention.Kernels
import Mathlib.Tactic
import VeriTile.Meta.StatementAudit

/-! Unchanged finite-address arithmetic, split from Address so the FP API can reuse it without importing Real execution. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.Address
open VeriTile.Triton
noncomputable section
theorem flatten2_bounded {m n i j : Nat} (hi : i < m) (hj : j < n) :
    i*n+j < m*n := by
  calc
    i*n+j < i*n+n := Nat.add_lt_add_left hj _
    _ = (i+1)*n := by ring
    _ ≤ m*n := Nat.mul_le_mul_right n (Nat.succ_le_iff.mpr hi)

theorem flatten3_bounded {heads times features head time feature : Nat}
    (hh : head < heads) (ht : time < times) (hf : feature < features) :
    head*(times*features)+time*features+feature < heads*times*features := by
  have h := flatten2_bounded (flatten2_bounded hh ht) hf
  nlinarith

theorem history_bounded {heads chunks keys values head chunk key value : Nat}
    (hh : head < heads) (hc : chunk < chunks) (hk : key < keys) (hv : value < values) :
    head*(chunks*keys*values)+chunk*keys*values+key*values+value < heads*chunks*keys*values := by
  have h := flatten2_bounded (flatten2_bounded (flatten2_bounded hh hc) hk) hv
  nlinarith

def word (n : Nat) : BitVec 64 := BitVec.ofNat 64 n

theorem word_exact {n : Nat} (h : n < 2^63) : (word n).toInt = (n : Int) := by
  have hn : (word n).toNat = n := Nat.mod_eq_of_lt (by omega)
  rw [BitVec.toInt_eq_toNat_of_lt (by rw [hn]; omega),hn]

/-- The actual arithmetic nodes, not just a final integer cast. -/
def historyWord (head chunk keys values chunks key value : Nat) : BitVec 64 :=
  word head * word (chunks*keys*values) + word chunk * word keys * word values +
    word key * word values + word value

theorem history_word_eq (head chunk keys values chunks key value : Nat) :
    historyWord head chunk keys values chunks key value =
      word (head*(chunks*keys*values)+chunk*keys*values+key*values+value) := by
  simp [historyWord,word,← BitVec.ofNat_mul,← BitVec.ofNat_add]

/-- Allocation-supported histories cannot wrap, including the previous 2^31
counterexample. Each partial additive address is nonnegative and no larger
than this final cell. -/
theorem history_native_exact {heads chunks keys values head chunk key value : Nat}
    (hh : head < heads) (hc : chunk < chunks) (hk : key < keys) (hv : value < values)
    (allocated : heads*chunks*keys*values ≤ 2^63) :
    (historyWord head chunk keys values chunks key value).toInt =
      (head*(chunks*keys*values)+chunk*keys*values+key*values+value : Nat) := by
  rw [history_word_eq]
  exact word_exact (lt_of_lt_of_le (history_bounded hh hc hk hv) allocated)

def matrixWord (head time feature times features : Nat) : BitVec 64 :=
  word head * word (times*features) + word time * word features + word feature

theorem matrix_native_exact {heads times features head time feature : Nat}
    (hh : head < heads) (ht : time < times) (hf : feature < features)
    (allocated : heads*times*features ≤ 2^63) :
    (matrixWord head time feature times features).toInt =
      (head*(times*features)+time*features+feature : Nat) := by
  have he : matrixWord head time feature times features =
      word (head*(times*features)+time*features+feature) := by
    simp [matrixWord,word,← BitVec.ofNat_mul,← BitVec.ofNat_add]
  rw [he]
  exact word_exact (lt_of_lt_of_le (flatten3_bounded hh ht hf) allocated)


#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.Address
