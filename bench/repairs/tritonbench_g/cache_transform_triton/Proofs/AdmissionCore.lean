import bench.repairs.tritonbench_g.cache_transform_triton.Contract

namespace VeriTile.Bench.Repairs.CacheTransform
open VeriTile.Triton FP.Structural

theorem checkInputs_iff (r : Request) : checkInputs r=Bool.true ↔ InputContract r := by
  simp only [checkInputs,Bool.and_eq_true,decide_eq_true_eq,InputContract,and_assoc]

theorem admitted_lengths (r : Request) (h : InputContract r) :
    ValidLengths r.mode r.naturalLengths r.cosRows ∧ ValidLengths r.mode r.naturalLengths r.sinRows := by
  have valid : ∀ n∈r.naturalLengths,n ≤ r.cosRows ∧ n ≤ r.sinRows ∧ (r.mode=.decode → 0<n) := by
    intro n hn
    obtain ⟨v,hv,rfl⟩ := List.mem_map.mp hn
    obtain ⟨lo,hc,hs⟩ := h.2.1 v hv
    have nonneg : 0 ≤ v := by cases hm : r.mode <;> simp [hm] at lo <;> omega
    refine ⟨by omega,by omega,?_⟩
    intro hm
    simp [hm] at lo
    omega
  exact ⟨fun n hn => ⟨(valid n hn).1,(valid n hn).2.2⟩,
    fun n hn => ⟨(valid n hn).2.1,(valid n hn).2.2⟩⟩

theorem admitted_signed (r : Request) (h : InputContract r) (v : Int) (hv : v∈r.values) :
    (v.toNat : Int)=v := by
  have lo := (h.2.1 v hv).1
  apply Int.toNat_of_nonneg
  cases hm : r.mode <;> simp [hm] at lo <;> omega

theorem outputRows_eq (r : Request) : r.outputRows=(preparedRows r.mode r.naturalLengths).length := by
  rw [preparedRows_length]
  cases hm : r.mode <;> simp [Request.outputRows,Request.naturalLengths,hm]

/-- The inherited general scheduling lemma specializes to precisely one
launch: START=0, BLOCK=256, and unmodified program ID on every active tile. -/
theorem single_launch (r : Request) (h : InputContract r) (k : Nat) (hk : k<blocks r.count) :
    chunkStart k=0 ∧ localPid k=k := by
  have bound : k<gridLimit := by
    have hb := h.2.2.2.2.2.1
    unfold gridLimit
    omega
  rw [chunkStart_eq,localPid_eq,Nat.div_eq_of_lt bound,Nat.mod_eq_of_lt bound]
  simp


end VeriTile.Bench.Repairs.CacheTransform
