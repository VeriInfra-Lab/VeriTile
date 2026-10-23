import bench.repairs.tritonbench_g.kcache_copy_triton.Contract
import bench.repairs.tritonbench_g.kv_cache_copy.Proofs.Contract
namespace VeriTile.Bench.Repairs.KcacheCopyTriton
noncomputable section
set_option maxHeartbeats 4000000

theorem checkInputs_iff (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables) :
    checkInputs m n b l t=.ok () ↔ InputContract m n b l t := by
  have fields : InputContract m n b l t ↔ (0<n ∧ n<2^31) ∧
      (∀ r,(m r).layout=.strided) ∧ Types m ∧ (∀ r,(m r).device.kind="cuda") ∧
      (∀ r,(m r).device=(m .key).device) ∧ Ranks m ∧
      (∀ r,(m r).isContiguous=Bool.true) ∧ (∀ r,(m r).isNeg=Bool.false) ∧
      Shapes m (config m n b) ∧ CacheShape m (config m n b) ∧ Geometry (config m n b) ∧
      ((config m n b).count<2^31 ∧ (config m n b).heads<2^16 ∧ (config m n b).splits<2^16 ∧
        (config m n b).count*(config m n b).heads*(config m n b).splits<2^31) ∧
      ((∀ r,(m r).numel<2^63) ∧ (config m n b).blockSize<2^63 ∧ (config m n b).columns<2^63) ∧
      (config m n b).width≤2^20 ∧ (∀ r,0<(m r).numel → base m r%4=0) ∧
      Separate m ∧ Bounds (config m n b) l t ∧ Unique (config m n b) l t := by
    constructor
    · intro h
      exact ⟨h.nBound,h.layout,h.dtype,h.cuda,h.device,h.rank,h.contiguous,h.noNeg,h.shapes,h.cacheShape,
        h.geometry,h.grid,h.extent,h.tile,h.alignment,h.separate,h.bounds,h.unique⟩
    · rintro ⟨a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r⟩
      exact ⟨a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r⟩
  rw [fields]
  simp [checkInputs,ite_eq_iff]

theorem checkArguments_iff (m : Option Inputs) (n : Option Int) (b : Option Bool) (l : Lengths) (t : Tables) :
    checkArguments m n b l t=.ok () ↔ Admitted m n b l t := by
  cases m with
  | none => simp [checkArguments,Admitted]
  | some m =>
    cases n with
    | none => simp [checkArguments,Admitted]
    | some n =>
      by_cases hn : 0<n ∧ n<2^31
      · simp only [checkArguments,if_neg (not_not.mpr hn)]
        cases b <;> simp [Admitted,checkInputs_iff,hn.1]
      · simp only [checkArguments,if_pos hn]
        cases b with
        | none => simp [Admitted]
        | some b =>
          constructor
          · intro bad;cases bad
          · rintro ⟨positive,contract⟩
            have bound := contract.nBound
            exact False.elim (hn (by omega))

theorem normalized_rank (xs : List Nat) (rank : xs.length=3 ∨ xs.length=4) : (normalized xs).length=3 := by
  rcases rank with rank|rank <;> simp [normalized,rank]

theorem normalized_prod (xs : List Nat) (rank : xs.length=3 ∨ xs.length=4) : (normalized xs).prod=xs.prod := by
  rcases rank with rank|rank
  · simp [normalized,rank]
  · rcases xs with _|⟨a,xs⟩ <;> simp at rank
    rcases xs with _|⟨b,xs⟩ <;> simp at rank
    rcases xs with _|⟨c,xs⟩ <;> simp at rank
    rcases xs with _|⟨d,xs⟩ <;> simp at rank
    have empty : xs=[] := by simpa using rank
    subst xs
    simp [normalized,Nat.mul_assoc]

theorem input_shape (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m n b l t) :
    normalized (m .key).shape=[(config m n b).count,(config m n b).heads,(config m n b).dims] :=
  KvCacheCopy.triple _ (normalized_rank _ h.rank.1)

theorem split_geometry (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m n b l t) :
    0<(config m n b).width ∧ (config m n b).splits*(config m n b).width=(config m n b).dims := by
  obtain ⟨e,he⟩ := h.geometry.2.2.2
  refine ⟨by rw [he];positivity,?_⟩
  cases b with
  | «false» => simp [config]
  | «true» => exact h.cacheShape.2

theorem numel (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m n b l t) :
    (m .key).numel=(config m n b).count*(config m n b).heads*(config m n b).dims ∧
    (m .cache).numel=(config m n b).blocks*(config m n b).heads*(config m n b).blockSize*(config m n b).dims ∧
    (m .lengths).numel=(config m n b).count/n ∧
    (m .table).numel=(config m n b).count/n*(config m n b).columns := by
  refine ⟨?_,?_,?_,?_⟩
  · change (m .key).shape.prod=_
    rw [←normalized_prod _ h.rank.1,input_shape m n b l t h]
    simp [Nat.mul_assoc]
  · change (m .cache).shape.prod=_
    cases b with
    | «false» => rw [h.cacheShape];simp [Nat.mul_assoc]
    | «true» =>
      rw [h.cacheShape.1]
      simp only [List.prod_cons,List.prod_nil,Nat.mul_one]
      rw [←(split_geometry m n Bool.true l t h).2]
      ring
  · change (m .lengths).shape.prod=_
    rw [h.shapes.2.1]
    simp [config]
  · change (m .table).shape.prod=_
    rw [h.shapes.2.2]
    simp [config]

theorem sequence_bound (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m n b l t)
    (s : Nat) (hs : s<(config m n b).count) : s/n<(config m n b).count/n := by
  apply (Nat.div_lt_iff_lt_mul h.nBound.1).mpr
  have divisible : (config m n b).count%n=0 := h.shapes.1
  rw [Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero divisible)]
  exact hs

theorem metadata_bounds (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m n b l t)
    (s : Nat) (hs : s<(config m n b).count) :
    past (config m n b) l s<2^31 ∧ column (config m n b) l s<(config m n b).columns ∧
    slot (config m n b) l s<(config m n b).blockSize ∧ blockId (config m n b) l t s<(config m n b).blocks := by
  have seq := sequence_bound m n b l t h s hs
  have bounds := h.bounds.1 _ seq
  have block := h.bounds.2 s hs
  have lane := Nat.mod_lt s h.nBound.1
  have positive := h.nBound.1
  have upper := (l (s/n)).toInt_lt
  have pastBound : past (config m n b) l s<(config m n b).blockSize*(config m n b).columns := by
    change (l (s/n)).toInt.toNat-n+s%n<_
    change (n:Int)≤(l (s/n)).toInt ∧ _ at bounds
    omega
  refine ⟨?_,?_,Nat.mod_lt _ h.geometry.2.2.1,?_⟩
  · change (l (s/n)).toInt.toNat-n+s%n<_
    change (n:Int)≤(l (s/n)).toInt ∧ _ at bounds
    omega
  · exact (Nat.div_lt_iff_lt_mul h.geometry.2.2.1).mpr (by simpa [Nat.mul_comm] using pastBound)
  · unfold blockId
    omega

#axiomsClean checkArguments_iff
#axiomsClean numel
#axiomsClean metadata_bounds
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KcacheCopyTriton
