import bench.repairs.tritonbench_g.max_reduction.Proofs.Contract

namespace VeriTile.Bench.Repairs.MaxReduction.Grid
noncomputable section

/-- Exact two-dimensional launch and masked row stores. Empty outer dimensions
produce no jobs; every retained BLOCK_M choice uses the same complete axis. -/
def programs (M K BM : Nat) : List (Nat × Nat) :=
  (List.range (Address.count M BM)).flatMap fun p => (List.range K).map fun k => (p,k)

def lanes (M BM : Nat) (p : Nat × Nat) : List Nat :=
  (List.range BM).filter fun i => p.1*BM+i<M

def jobs (M K BM : Nat) : List Nat :=
  (programs M K BM).flatMap fun p => (lanes M BM p).map fun i => (p.1*BM+i)*K+p.2

theorem mem_programs (M K BM p k : Nat) :
    (p,k)∈programs M K BM ↔ p<Address.count M BM ∧ k<K := by simp [programs]

theorem mem_jobs (M K BM j : Nat) (hb : 0<BM) : j∈jobs M K BM ↔ j<M*K := by
  simp only [jobs,List.mem_flatMap,List.mem_map]
  constructor
  · rintro ⟨⟨p,k⟩,hp,i,hi,rfl⟩
    have hg := (mem_programs M K BM p k).mp hp
    have hl : p*BM+i<M := (List.mem_filter.mp hi).2 |> of_decide_eq_true
    exact Address.output_bound M K _ _ hl hg.2
  · intro hj
    have hk : 0<K := by nlinarith
    have hm := (Nat.div_lt_iff_lt_mul hk).mpr hj
    obtain ⟨hp,hl,he⟩ := Address.segment_owner M BM (j/K) hb hm
    simp only [Address.start] at he
    refine ⟨(j/K/BM,j%K),(mem_programs M K BM _ _).mpr ⟨hp,Nat.mod_lt _ hk⟩,
      j/K%BM,?_,?_⟩
    · simp only [lanes,List.mem_filter,List.mem_range,decide_eq_true_eq]
      exact ⟨Nat.mod_lt _ hb,by rw [he]; exact hm⟩
    · change (j/K/BM*BM+j/K%BM)*K+j%K=j
      rw [he]
      simpa [Nat.mul_comm] using Nat.div_add_mod j K

theorem empty_left (K BM : Nat) : jobs 0 K BM=[] := by simp [jobs,lanes]
theorem empty_right (M BM : Nat) : jobs M 0 BM=[] := by simp [jobs,programs]

/-- Different active program/lane coordinates never race on a value/index
slot. The same proof applies to every autotuning probe and selected launch. -/
theorem unique_owner (M K BM p k i q l j : Nat) (hb : 0<BM)
    (hp : (p,k)∈programs M K BM) (hq : (q,l)∈programs M K BM)
    (hi : i∈lanes M BM (p,k)) (hj : j∈lanes M BM (q,l))
    (same : (p*BM+i)*K+k=(q*BM+j)*K+l) : p=q ∧ k=l ∧ i=j := by
  have ho := Address.output_injective K _ _ _ _ ((mem_programs _ _ _ _ _).mp hp).2
    ((mem_programs _ _ _ _ _).mp hq).2 same
  have hr := Address.owner_unique BM p i q j hb
    (List.mem_range.mp (List.mem_filter.mp hi).1) (List.mem_range.mp (List.mem_filter.mp hj).1) ho.1
  exact ⟨hr.1,ho.2,hr.2⟩

#axiomsClean mem_jobs
#axiomsClean unique_owner
end
end VeriTile.Bench.Repairs.MaxReduction.Grid
