import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.Bounds
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.NativeHost
namespace VeriTile.Bench.Repairs.LayerNormWelfold.Native
open VeriTile.Triton FP.Structural Reuse
noncomputable section
set_option maxHeartbeats 1000000

/-- Exact domain of the selected signed-i32 source index ABI. -/
structure IndexABI (S D : Nat) : Prop where
  domain : NativeIndexDomain S D

def word32 (n : Nat) : BitVec 32 := BitVec.ofNat 32 n
def rowWord (D row : Nat) : BitVec 32 := word32 D * word32 row
def offsetWord (B k : Nat) : BitVec 32 := word32 k * word32 B
def columnWord (B k i : Nat) : BitVec 32 := offsetWord B k + word32 i
def flatWord (D B row k i : Nat) : BitVec 32 := rowWord D row + columnWord B k i
def nextWord (B k : Nat) : BitVec 32 := offsetWord B k + word32 B

theorem word32_exact (n : Nat) (h : n<2^31) : (word32 n).toInt=(n : Int) := by
  have hn : n<2^32 := by omega
  have ht : (word32 n).toNat=n := by
    change n%2^32=n
    exact Nat.mod_eq_of_lt hn
  rw [BitVec.toInt_eq_toNat_of_lt (by rw [ht]; omega),ht]

theorem word_operations (D B row k i : Nat) :
    rowWord D row=word32 (D*row) ∧ offsetWord B k=word32 (k*B) ∧
    columnWord B k i=word32 (k*B+i) ∧ flatWord D B row k i=word32 (k*B+i+D*row) ∧
    nextWord B k=word32 ((k+1)*B) := by
  simp [rowWord,offsetWord,columnWord,flatWord,nextWord,word32,
    ←BitVec.ofNat_mul,←BitVec.ofNat_add,Nat.add_mul,Nat.add_comm]

/-- Source scalar parameters, row products, all padded offsets, masks, final
loop exit and signed widening to FP32/BF16 byte displacements are exact.
No numeric output or compiler graph equality is a premise. -/
structure SourceExact (S D B : Nat) : Prop where
  parameters : (word32 S).toInt=(S : Int) ∧ (word32 D).toInt=(D : Int) ∧ (word32 B).toInt=(B : Int)
  row : ∀ r, r<S → (word32 r).toInt=(r : Int) ∧ (rowWord D r).toInt=(D*r : Nat)
  loop : ∀ k, k≤blockCount D B →
    (offsetWord B k).toInt=(k*B : Nat) ∧
    ((offsetWord B k).toInt<(word32 D).toInt ↔ k<blockCount D B)
  exit : ¬(offsetWord B (blockCount D B)).toInt<(word32 D).toInt
  padded : ∀ r, r<S → ∀ k, k<blockCount D B → ∀ i : Fin B,
    (columnWord B k i.val).toInt=(k*B+i.val : Nat) ∧
    (flatWord D B r k i.val).toInt=(k*B+i.val+D*r : Nat) ∧
    (nextWord B k).toInt=((k+1)*B : Nat) ∧
    ((columnWord B k i.val).toInt<(word32 D).toInt ↔ k*B+i.val<D) ∧
    ((word32 r).toInt<(word32 S).toInt ↔ r<S)
  byte_displacement : ∀ r, r<S → ∀ k, k<blockCount D B → ∀ i : Fin B,
    (flatWord D B r k i.val).toInt*4=(4*(k*B+i.val+D*r) : Nat) ∧
    (flatWord D B r k i.val).toInt*2=(2*(k*B+i.val+D*r) : Nat) ∧
    4*(k*B+i.val+D*r)<2^63 ∧ 4*r<2^63

theorem source_exact (S D B : Nat) (abi : IndexABI S D) (allowed : B=1024 ∨ B=2048) : SourceExact S D B := by
  have hD : D<2^31 := by have := abi.domain.2.1; omega
  have hS : S<2^31 := by
    have hSD : S≤S*D := by nlinarith [abi.domain.1]
    have := abi.domain.2.2; omega
  have hB : 0<B := by rcases allowed with h|h <;> omega
  have hBu : B≤2048 := by rcases allowed with h|h <;> omega
  have hBi : B<2^31 := by omega
  have ho : ∀ k, k≤blockCount D B → (offsetWord B k).toInt=(k*B : Nat) := by
    intro k hk
    have hu := rounded_width_upper D B hB
    have hm := Nat.mul_le_mul_right B hk
    have hd := abi.domain.2.1
    rw [(word_operations D B 0 k 0).2.1]
    exact word32_exact _ (by omega)
  have hp : ∀ r, r<S → ∀ k, k<blockCount D B → ∀ i : Fin B,
      (columnWord B k i.val).toInt=(k*B+i.val : Nat) ∧
      (flatWord D B r k i.val).toInt=(k*B+i.val+D*r : Nat) ∧
      (nextWord B k).toInt=((k+1)*B : Nat) := by
    intro r hr k hk i
    have hf := native_indices_fit S D B r k i abi.domain allowed hr hk
    have hw := word_operations D B r k i.val
    exact ⟨by rw [hw.2.2.1]; exact word32_exact _ hf.1,
      by rw [hw.2.2.2.1]; exact word32_exact _ hf.2.1,
      by rw [hw.2.2.2.2]; exact word32_exact _ hf.2.2.1⟩
  refine ⟨⟨word32_exact S hS,word32_exact D hD,word32_exact B hBi⟩,?_,?_,?_,?_,?_⟩
  · intro r hr
    have hm : D*r<S*D := by nlinarith [abi.domain.1]
    have hs := abi.domain.2.2
    exact ⟨word32_exact r (by omega),by rw [(word_operations D B r 0 0).1]; exact word32_exact _ (by omega)⟩
  · intro k hk
    refine ⟨ho k hk,?_⟩
    rw [ho k hk,word32_exact D hD]
    exact_mod_cast (range_offset_iff D B k hB).symm
  · rw [ho _ (by rfl),word32_exact D hD]
    exact_mod_cast (Nat.not_lt_of_ge (block_count_covers D B hB))
  · intro r hr k hk i
    have hx := hp r hr k hk i
    refine ⟨hx.1,hx.2.1,hx.2.2,?_,?_⟩
    · rw [hx.1,word32_exact D hD]; omega
    · rw [word32_exact r (by omega),word32_exact S hS]; omega
  · intro r hr k hk i
    have hf := native_indices_fit S D B r k i abi.domain allowed hr hk
    rw [(hp r hr k hk i).2.1]
    constructor
    · push_cast; omega
    constructor
    · push_cast; omega
    constructor <;> omega

/-- The accepted native entry carries the ABI as a public premise and returns
fixed-width source exactness for both trial and selected configurations, joined
to the unchanged complete returned graph theorem. -/
theorem host_native_i32_complete {α : Type} [Inhabited α] (M : Algebra α) (eq32 : α→α→Bool)
    (p : Primitives α) (bindings : SourceBindings M p) (s : State α) (X W Bias : Tensor)
    (a : Allocations) (S D : Nat) (domain : InputContract s X W Bias S D) (abi : IndexABI S D)
    (live : RegionName→Prop) (hX : live X.region) (hW : live W.region) (hBias : live Bias.region)
    (fresh : a.Fresh live) (trials : List Configuration) (last : Configuration)
    (program : CompiledCall S D last)
    (laws : ∀ port, LocalLaws p (inputs M s X.region W.region Bias.region D port.row)
      (fun step => step∈(program.plan port).steps)) :
    (∀ c : Configuration, SourceExact S D c.block) ∧
    ∃ ret, executeHost M eq32 s X W Bias a S D trials last=some ret ∧
      ret.tensors=(executeNative M p s X W Bias a S D last program).tensors ∧
      ret.state.mem=(executeNative M p s X W Bias a S D last program).state.mem ∧
      ReturnedCorrect M eq32 s X W Bias a S D last (executeNative M p s X W Bias a S D last program) ∧
      (∀ q, live q→∀ off,(executeNative M p s X W Bias a S D last program).state.mem q off=s.mem q off) := by
  exact ⟨fun c => source_exact S D c.block abi c.allowed,
    host_native_complete M eq32 p bindings s X W Bias a S D domain live hX hW hBias fresh trials last program laws⟩

theorem saved_profile_admissible : IndexABI 2 2049 := by
  exact ⟨by norm_num [NativeIndexDomain]⟩

theorem review_counterexample_excluded : ¬IndexABI (2^21+1) 1024 := by
  intro h
  have hs := h.domain.2.2
  norm_num at hs

#axiomsClean word32_exact
#axiomsClean word_operations
#axiomsClean source_exact
#axiomsClean host_native_i32_complete
#axiomsClean saved_profile_admissible
#axiomsClean review_counterexample_excluded
end
end VeriTile.Bench.Repairs.LayerNormWelfold.Native
