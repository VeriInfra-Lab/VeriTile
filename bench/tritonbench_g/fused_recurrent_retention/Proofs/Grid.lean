import bench.tritonbench_g.fused_recurrent_retention.Proofs.Domain
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention
noncomputable section
set_option maxHeartbeats 4000000

/- The actual loop-carried tile. Forward stores follow the update; the source
[BV,BK] versus [BK,BV] orientation only changes the coordinate projection. -/
theorem programForwardCarry_at (a : Arithmetic α) (s : Shape) (x : Inputs α) (seed : Bool)
    (p : ProgramId) (n lk lv : Nat) :
    programForwardCarry a s x seed p n lk lv=state a s x seed p.head (p.key s lk) (p.value s lv) n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    change a.fma _ (programForwardCarry a s x seed p n lk lv) _=_
    rw [ih]
    rfl

/- Reverse carry starts with F, before any decay, and is decayed only after
the dK/dV store values have been computed. -/
theorem programReverseCarry_at (a : Arithmetic α) (s : Shape) (x : Inputs α) (final : Bool)
    (p : ProgramId) (n lk lv : Nat) :
    programReverseCarry a s x final p n lk lv=reverseCarry a s x final p.head (p.key s lk) (p.value s lv) n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    change a.mul (a.fma _ _ (programReverseCarry a s x final p n lk lv)) _=_
    rw [ih]
    rfl

theorem programAdjoint_at (a : Arithmetic α) (s : Shape) (x : Inputs α) (final : Bool)
    (p : ProgramId) (t lk lv : Nat) :
    programAdjoint a s x final p t lk lv=adjoint a s x final p.head t (p.key s lk) (p.value s lv) := by
  simp only [programAdjoint,programReverseCarry_at,adjoint]

theorem programO_partial (a : Arithmetic α) (s : Shape) (plans : Plans) (x : Inputs α) (seed : Bool)
    (p : ProgramId) (t lv : Nat) :
    programO a s plans x seed p t lv=partialO a s plans x seed p.head t p.ik (p.value s lv) := by
  simp only [programO,programForwardCarry_at,partialO,ProgramId.key]

theorem programQ_partial (a : Arithmetic α) (s : Shape) (plans : Plans) (x : Inputs α) (seed : Bool)
    (p : ProgramId) (t lk : Nat) :
    programQ a s plans x seed p t lk=partialQ a s plans x seed p.head t p.iv (p.key s lk) := by
  simp only [programQ,programForwardCarry_at,partialQ,ProgramId.value]

theorem programK_partial (a : Arithmetic α) (s : Shape) (plans : Plans) (x : Inputs α) (final : Bool)
    (p : ProgramId) (t lk : Nat) :
    programK a s plans x final p t lk=partialK a s plans x final p.head t p.iv (p.key s lk) := by
  simp only [programK,programAdjoint_at,partialK,ProgramId.value]

theorem programV_partial (a : Arithmetic α) (s : Shape) (plans : Plans) (x : Inputs α) (final : Bool)
    (p : ProgramId) (t lv : Nat) :
    programV a s plans x final p t lv=partialV a s plans x final p.head t p.ik (p.value s lv) := by
  simp only [programV,programAdjoint_at,partialV,ProgramId.key]

/- Every active output has an actual grid owner, including nonmultiple edge
tiles. No T>0, B>0 or H>0 condition is imposed globally. -/
theorem output_owner (s : Shape) (domain : s.Admitted) (split h v : Nat)
    (hs : split<s.NK) (hh : h<s.BH) (hv : v<s.DV) :
    let p : ProgramId := ⟨v/s.BV,split,h⟩
    p.Valid s ∧ v%s.BV<s.BV ∧ p.value s (v%s.BV)=v := by
  exact ⟨⟨owner_bound s.DV s.BV v hv,hs,hh⟩,Nat.mod_lt _ (block_positive s domain).2,
    owner_reconstruct s.BV v⟩

theorem gradient_owner (s : Shape) (domain : s.Admitted) (split h k : Nat)
    (hs : split<s.NV) (hh : h<s.BH) (hk : k<s.DK) :
    let p : ProgramId := ⟨split,k/s.BK,h⟩
    p.Valid s ∧ k%s.BK<s.BK ∧ p.key s (k%s.BK)=k := by
  exact ⟨⟨hs,owner_bound s.DK s.BK k hk,hh⟩,Nat.mod_lt _ (block_positive s domain).1,
    owner_reconstruct s.BK k⟩

theorem final_owner (s : Shape) (domain : s.Admitted) (h k v : Nat)
    (hh : h<s.BH) (hk : k<s.DK) (hv : v<s.DV) :
    let p : ProgramId := ⟨v/s.BV,k/s.BK,h⟩
    p.Valid s ∧ k%s.BK<s.BK ∧ v%s.BV<s.BV ∧
      p.key s (k%s.BK)=k ∧ p.value s (v%s.BV)=v := by
  exact ⟨⟨owner_bound s.DV s.BV v hv,owner_bound s.DK s.BK k hk,hh⟩,
    Nat.mod_lt _ (block_positive s domain).1,Nat.mod_lt _ (block_positive s domain).2,
    owner_reconstruct s.BK k,owner_reconstruct s.BV v⟩

theorem output_writer_unique (s : Shape) (p p' : ProgramId) (t t' lv lv' : Nat)
    (hp : p.Valid s) (hp' : p'.Valid s) (ht : t<s.T) (ht' : t'<s.T)
    (hl : lv<s.BV) (hl' : lv'<s.BV) (hv : p.value s lv<s.DV) (hv' : p'.value s lv'<s.DV)
    (he : partialOffset s s.DV p.ik p.head t (p.value s lv)=
      partialOffset s s.DV p'.ik p'.head t' (p'.value s lv')) : p=p' ∧ t=t' ∧ lv=lv' := by
  obtain ⟨hik,hh,htt,hvv⟩ := partialOffset_injective s s.DV _ _ _ _ _ _ _ _ hp.2.2 hp'.2.2 ht ht' hv hv' he
  obtain ⟨hiv,hll⟩ := pair_index_injective s.BV _ _ _ _ hl hl' hvv
  exact ⟨by cases p; cases p'; simp_all,htt,hll⟩

theorem gradient_writer_unique (s : Shape) (p p' : ProgramId) (t t' lk lk' : Nat)
    (hp : p.Valid s) (hp' : p'.Valid s) (ht : t<s.T) (ht' : t'<s.T)
    (hl : lk<s.BK) (hl' : lk'<s.BK) (hk : p.key s lk<s.DK) (hk' : p'.key s lk'<s.DK)
    (he : partialOffset s s.DK p.iv p.head t (p.key s lk)=
      partialOffset s s.DK p'.iv p'.head t' (p'.key s lk')) : p=p' ∧ t=t' ∧ lk=lk' := by
  obtain ⟨hiv,hh,htt,hkk⟩ := partialOffset_injective s s.DK _ _ _ _ _ _ _ _ hp.2.2 hp'.2.2 ht ht' hk hk' he
  obtain ⟨hik,hll⟩ := pair_index_injective s.BK _ _ _ _ hl hl' hkk
  exact ⟨by cases p; cases p'; simp_all,htt,hll⟩

theorem final_writer_unique (s : Shape) (p p' : ProgramId) (lk lk' lv lv' : Nat)
    (hlk : lk<s.BK) (hlk' : lk'<s.BK) (hlv : lv<s.BV) (hlv' : lv'<s.BV)
    (hk : p.key s lk<s.DK) (hk' : p'.key s lk'<s.DK)
    (hv : p.value s lv<s.DV) (hv' : p'.value s lv'<s.DV)
    (he : stateOffset s p.head (p.key s lk) (p.value s lv)=
      stateOffset s p'.head (p'.key s lk') (p'.value s lv')) : p=p' ∧ lk=lk' ∧ lv=lv' := by
  obtain ⟨hh,hkk,hvv⟩ := stateOffset_injective s _ _ _ _ _ _ hk hk' hv hv' he
  obtain ⟨hik,hkl⟩ := pair_index_injective s.BK _ _ _ _ hlk hlk' hkk
  obtain ⟨hiv,hvl⟩ := pair_index_injective s.BV _ _ _ _ hlv hlv' hvv
  exact ⟨by cases p; cases p'; simp_all,hkl,hvl⟩

#axiomsClean programForwardCarry_at
#axiomsClean programReverseCarry_at
#axiomsClean programO_partial
#axiomsClean programQ_partial
#axiomsClean programK_partial
#axiomsClean programV_partial
#axiomsClean output_owner
#axiomsClean gradient_owner
#axiomsClean final_owner
#axiomsClean output_writer_unique
#axiomsClean gradient_writer_unique
#axiomsClean final_writer_unique
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention
