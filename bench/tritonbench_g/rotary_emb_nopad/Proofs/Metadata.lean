import bench.tritonbench_g.rotary_emb_nopad.Proofs.Physical

namespace VeriTile.Bench.Repairs.RotaryEmbNopad.Physical
open Checked Native

def MetadataAgreement (r : Request) (a b : Indices) : Prop := r.cached=true →
  (∀ t,t<r.tokens → a (r.lengthBase+2*t)=b (r.lengthBase+2*t)) ∧
  (∀ j,j<r.tokens*r.tableColumns → a (r.tableBase+2*j)=b (r.tableBase+2*j))

theorem agreement_refl (r : Request) (a : Indices) : MetadataAgreement r a a := by
  intro _
  exact ⟨fun _ _ => rfl,fun _ _ => rfl⟩

theorem agreement_trans (r : Request) (a b d : Indices)
    (first : MetadataAgreement r a b) (second : MetadataAgreement r b d) : MetadataAgreement r a d := by
  intro cached
  exact ⟨fun t ht => ((first cached).1 t ht).trans ((second cached).1 t ht),
    fun j hj => ((first cached).2 j hj).trans ((second cached).2 j hj)⟩

theorem metadata_word_frame (p : Instructions) (c : Call) (index : Indices) (accepted : check c index=true)
    (jobs : List Work) (active : ∀ w∈jobs,c.request.Active w) (m : Bytes)
    (cached : c.request.cached=true) (i : Fin 7) (which : i=5 ∨ i=6) (offset : Nat)
    (inside : offset<wordExtent c.request i) :
    load (byteExecute p c.request index jobs m) (.other c.request.device) (base c.request i+offset)=
      load m (.other c.request.device) (base c.request i+offset) := by
  rw [execute_load]
  apply word_frame
  intro w hw
  have pi : present c.request i=true := by simp [present,cached]
  have readonly : ∀ key : Bool,destinationRole c.request key≠i := by
    intro key
    rcases which with rfl|rfl <;> cases key <;> simp [destinationRole,role,cached]
  rintro ⟨_,same|same⟩
  · exact readonly_role c index accepted i pi readonly w (active w hw) false offset inside same.symm
  · exact readonly_role c index accepted i pi readonly w (active w hw) true offset inside same.symm

theorem metadata_frame (p : Instructions) (c : Call) (index : Indices) (accepted : check c index=true)
    (jobs : List Work) (active : ∀ w∈jobs,c.request.Active w) (m : Bytes) :
    MetadataAgreement c.request (indices c.request (byteExecute p c.request index jobs m)) (indices c.request m) := by
  intro cached
  constructor
  · intro t ht
    have low := metadata_word_frame p c index accepted jobs active m cached 6 (Or.inr rfl) (2*t) (by
      simp [wordExtent,expectedShape]; omega)
    have high := metadata_word_frame p c index accepted jobs active m cached 6 (Or.inr rfl) (2*t+1) (by
      simp [wordExtent,expectedShape]; omega)
    change load (byteExecute p c.request index jobs m) (.other c.request.device) (c.request.lengthBase+2*t)=
      load m (.other c.request.device) (c.request.lengthBase+2*t) at low
    change load (byteExecute p c.request index jobs m) (.other c.request.device) (c.request.lengthBase+(2*t+1))=
      load m (.other c.request.device) (c.request.lengthBase+(2*t+1)) at high
    simp only [indices,Nat.add_assoc,low,high]
  · intro j hj
    have low := metadata_word_frame p c index accepted jobs active m cached 5 (Or.inl rfl) (2*j) (by
      simp [wordExtent,expectedShape]; omega)
    have high := metadata_word_frame p c index accepted jobs active m cached 5 (Or.inl rfl) (2*j+1) (by
      simp [wordExtent,expectedShape]; omega)
    change load (byteExecute p c.request index jobs m) (.other c.request.device) (c.request.tableBase+2*j)=
      load m (.other c.request.device) (c.request.tableBase+2*j) at low
    change load (byteExecute p c.request index jobs m) (.other c.request.device) (c.request.tableBase+(2*j+1))=
      load m (.other c.request.device) (c.request.tableBase+(2*j+1)) at high
    simp only [indices,Nat.add_assoc,low,high]

theorem destination_metadata (c : Call) (fixed current : Indices) (accepted : check c fixed=true)
    (same : MetadataAgreement c.request current fixed) (w : Work) (active : c.request.Active w) (odd : Bool) :
    destination c.request current w odd=destination c.request fixed w odd := by
  by_cases cached : c.request.cached=true
  · have length := (same cached).1 w.token active.1
    have past : c.request.past current w.token=c.request.past fixed w.token := by simp [Request.past,length]
    have good := metadata_checked c fixed accepted w active
    simp only [metadataCheck,cached,↓reduceIte,Bool.and_eq_true,decide_eq_true_eq] at good
    have page := (good.1 ⟨w.token,active.1⟩).2.2.1
    have bound := pair_bound _ _ _ _ active.1 page
    have table := (same cached).2 (w.token*c.request.tableColumns+c.request.past fixed w.token/c.request.blockSize) bound
    simp [destination,Request.destination,Request.block,Request.slot,past,table]
  · simp [destination,Request.destination,cached]

def dynamicStep (p : Instructions) (r : Request) (w : Work) (m : Bytes) := byteStep p r (indices r m) w m

def dynamicExecute (p : Instructions) (r : Request) : List Work → Bytes → Bytes
  | [],m => m
  | w::rest,m => dynamicExecute p r rest (dynamicStep p r w m)

theorem dynamic_step_eq (p : Instructions) (c : Call) (fixed : Indices) (accepted : check c fixed=true)
    (w : Work) (active : c.request.Active w) (m : Bytes)
    (same : MetadataAgreement c.request (indices c.request m) fixed) :
    dynamicStep p c.request w m=byteStep p c.request fixed w m := by
  unfold dynamicStep byteStep
  rw [destination_metadata c fixed (indices c.request m) accepted same w active false,
    destination_metadata c fixed (indices c.request m) accepted same w active true]

theorem dynamic_eq (p : Instructions) (c : Call) (fixed : Indices) (accepted : check c fixed=true)
    (jobs : List Work) (active : ∀ w∈jobs,c.request.Active w) (m : Bytes)
    (same : MetadataAgreement c.request (indices c.request m) fixed) :
    dynamicExecute p c.request jobs m=byteExecute p c.request fixed jobs m := by
  induction jobs generalizing m with
  | nil => rfl
  | cons w rest ih =>
    have hw := active w (by simp)
    simp only [dynamicExecute,byteExecute,dynamic_step_eq p c fixed accepted w hw m same]
    apply ih (fun v hv => active v (by simp [hv]))
    apply agreement_trans c.request _ (indices c.request m) fixed _ same
    have frame := metadata_frame p c fixed accepted [w] (by intro v hv; simp only [List.mem_singleton] at hv; subst v; exact hw) m
    exact frame

#axiomsClean metadata_frame
#axiomsClean destination_metadata
#axiomsClean dynamic_eq
end VeriTile.Bench.Repairs.RotaryEmbNopad.Physical
