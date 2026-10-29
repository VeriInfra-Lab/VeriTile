import bench.repairs.tritonbench_g.kv_cache_copy.Contract
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic

namespace VeriTile.Bench.Repairs.KvCacheCopy
noncomputable section
set_option maxHeartbeats 4000000

theorem checkInputs_iff (m : Inputs) (b : Bool) (l : Lengths) (t : Tables) :
    checkInputs m b l t=.ok () ↔ InputContract m b l t := by
  have fields : InputContract m b l t ↔
      (∀ r,(m r).layout=.strided) ∧ Types m ∧ (∀ r,(m r).device.kind="cuda") ∧
      (∀ r,(m r).device=(m .key).device) ∧ Ranks m ∧
      (∀ r,(m r).isContiguous=Bool.true) ∧ (∀ r,(m r).isNeg=Bool.false) ∧
      Shapes m (config m b) ∧ KeyShape m (config m b) ∧ Geometry (config m b) ∧
      ((config m b).count<2^31 ∧ (config m b).heads<2^16 ∧ (config m b).count*(config m b).heads<2^31) ∧
      ((∀ r,(m r).numel<2^31) ∧ (config m b).dims<2^31 ∧ (config m b).blockSize<2^31 ∧ (config m b).columns<2^31) ∧
      (config m b).width≤2^20 ∧ (∀ r,0<(m r).numel → base m r%4=0) ∧
      Separate m ∧ Bounds (config m b) l t ∧ Unique (config m b) l t := by
    constructor
    · intro h
      exact ⟨h.layout,h.dtype,h.cuda,h.device,h.rank,h.contiguous,h.noNeg,h.shapes,h.keyShape,h.geometry,
        h.grid,h.extent,h.tile,h.alignment,h.separate,h.bounds,h.unique⟩
    · rintro ⟨a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q⟩
      exact ⟨a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q⟩
  rw [fields]
  simp [checkInputs,ite_eq_iff]

theorem checkArguments_iff (m : Option Inputs) (b : Option Bool) (l : Lengths) (t : Tables) :
    checkArguments m b l t=.ok () ↔ Admitted m b l t := by
  cases m <;> cases b <;> simp [checkArguments,Admitted,checkInputs_iff]

theorem triple (xs : List Nat) (h : xs.length=3) :
    xs=[xs.getD 0 0,xs.getD 1 0,xs.getD 2 0] := by
  rcases xs with _|⟨a,xs⟩ <;> simp at h ⊢
  rcases xs with _|⟨b,xs⟩ <;> simp at h ⊢
  rcases xs with _|⟨c,xs⟩ <;> simp at h ⊢
  simpa using h

theorem normalized_rank (xs : List Nat) (h : InputRank xs) : (normalized xs).length=3 := by
  rcases h with h|⟨h,_⟩ <;> simp [normalized,h]

theorem normalized_prod (xs : List Nat) (h : InputRank xs) : (normalized xs).prod=xs.prod := by
  rcases h with h|⟨h,one⟩
  · simp [normalized,h]
  · rcases xs with _|⟨a,xs⟩ <;> simp at h
    rcases xs with _|⟨b,xs⟩ <;> simp at h
    rcases xs with _|⟨c,xs⟩ <;> simp at h
    rcases xs with _|⟨d,xs⟩ <;> simp at h
    have empty : xs=[] := by simpa using h
    subst xs
    simpa [normalized] using (show a*(c*d)=a*(b*(c*d)) by simpa using congrArg (fun n => a*(n*(c*d))) one.symm)

theorem input_shape (m : Inputs) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m b l t) :
    normalized (m .key).shape=[(config m b).count,(config m b).heads,(config m b).dims] :=
  triple _ (normalized_rank _ h.rank.1)

theorem split_geometry (m : Inputs) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m b l t) :
    0<(config m b).width ∧ (config m b).splits*(config m b).width=(config m b).dims := by
  obtain ⟨e,he⟩ := h.geometry.2.2.2
  refine ⟨by rw [he]; positivity,?_⟩
  cases b with
  | «false» => simp [config]
  | «true» => exact h.keyShape.2

theorem numel (m : Inputs) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m b l t) :
    (m .key).numel=(config m b).count*(config m b).heads*(config m b).dims ∧
    (m .value).numel=(m .key).numel ∧
    (m .keyCache).numel=(config m b).blocks*(config m b).heads*(config m b).blockSize*(config m b).dims ∧
    (m .valueCache).numel=(m .keyCache).numel ∧
    (m .lengths).numel=(config m b).count ∧
    (m .table).numel=(config m b).count*(config m b).columns := by
  have ki : (m .key).numel=(config m b).count*(config m b).heads*(config m b).dims := by
    change (m .key).shape.prod=_
    rw [←normalized_prod _ h.rank.1,input_shape m b l t h]
    simp [Nat.mul_assoc]
  have vi : (m .value).numel=(m .key).numel := by
    change (m .value).shape.prod=(m .key).shape.prod
    rw [←normalized_prod _ h.rank.2,←normalized_prod _ h.rank.1,h.shapes.1]
  have vc : (m .valueCache).numel=(config m b).blocks*(config m b).heads*(config m b).blockSize*(config m b).dims := by
    change (m .valueCache).shape.prod=_
    rw [h.shapes.2.1]
    simp [Nat.mul_assoc]
  have kc : (m .keyCache).numel=(m .valueCache).numel := by
    change (m .keyCache).shape.prod=(m .valueCache).shape.prod
    cases b with
    | «false» => rw [h.keyShape]
    | «true» =>
      rw [h.keyShape.1,h.shapes.2.1]
      simp only [List.prod_cons,List.prod_nil,Nat.mul_one]
      rw [←(split_geometry m Bool.true l t h).2]
      ring
  refine ⟨ki,vi,kc.trans vc,kc.symm,?_,?_⟩
  · change (m .lengths).shape.prod=_
    rw [h.shapes.2.2.1]
    simp
  · change (m .table).shape.prod=_
    rw [h.shapes.2.2.2]
    simp

theorem metadata_bounds (m : Inputs) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m b l t)
    (s : Nat) (hs : s<(config m b).count) :
    past l s<2^31 ∧ column (config m b) l s<(config m b).columns ∧
    slot (config m b) l s<(config m b).blockSize ∧ blockId (config m b) l t s<(config m b).blocks := by
  have bounds := h.bounds s hs
  have upper := (l s).toInt_lt
  have pastBound : past l s<(config m b).blockSize*(config m b).columns := by unfold past; omega
  refine ⟨by unfold past; omega,?_,Nat.mod_lt _ h.geometry.2.2.1,?_⟩
  · exact (Nat.div_lt_iff_lt_mul h.geometry.2.2.1).mpr (by simpa [Nat.mul_comm] using pastBound)
  · unfold blockId
    omega

#axiomsClean checkArguments_iff
#axiomsClean numel
#axiomsClean metadata_bounds
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KvCacheCopy
