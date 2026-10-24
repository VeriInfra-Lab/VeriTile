import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.Native
import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.Order

namespace VeriTile.Bench.Repairs.KvCacheFilling.Tensor
open TensorInputs

def output (t : TensorInputs) : Role → TensorDescriptor
  | .key => t.keyCache
  | .value => t.valueCache
  | .keyMeta => t.keyMeta.getD t.keyCache
  | .valueMeta => t.valueMeta.getD t.valueCache

def input (t : TensorInputs) (value : Bool) : TensorDescriptor := if value then t.value else t.key

/-- Ordinary native-contiguous Tensor observation, at the queried data_ptr.
The host checks no unresolved conjugation/negation. This observation reads the
same backing bytes in every memory state; it contains no kernel result. -/
def byte (t : TensorDescriptor) (m : Nat → Byte) (i b : Nat) : Byte :=
  m (t.metadata.dataPtr.toNat+i*bytes t+b)

/-- Ordinary storage binding for the actual Tensor arguments. Layout capacity
is Torch's storage bound, while disjointness is the checked allocation guard.
Empty windows have no pointer equation because no load/store uses them. -/
structure Bindings (t : TensorInputs) (r : Request) (l : Layout r) (ins : InputLayout r l) : Prop where
  outputPointer : ∀ role,activeRole r role → 0<r.pages*r.block*r.heads*r.width role →
    l.base role+l.offset role=(output t role).metadata.dataPtr.toNat
  inputPointer : ∀ v,0<r.tokens*r.heads*stateWidth r v →
    ins.base v+ins.offset v=(input t v).metadata.dataPtr.toNat
  outputStorage : ∀ role,activeRole r role → 0<r.pages*r.block*r.heads*r.width role →
    l.base role=(output t role).storageBase ∧ l.size role=(output t role).storageBytes
  inputStorage : ∀ v,0<r.tokens*r.heads*stateWidth r v →
    ins.base v=(input t v).storageBase ∧ ins.size v=(input t v).storageBytes
  addressSpace : Native.StorageFits r l ins

theorem output_width (t : TensorInputs) (r : Request) (h : InputContract t r)
    (role : Role) (ha : activeRole r role) : bytes (output t role)=elementBytes r role := by
  have types := h.2.1
  have scales := h.2.2.1
  cases role with
  | key =>
    by_cases hp : r.policy=.copy <;> simp [output,bytes,types.2.2.1,elementBytes,hp]
  | value =>
    by_cases hp : r.policy=.copy <;> simp [output,bytes,types.2.2.2.1,elementBytes,hp]
  | keyMeta =>
    have hp : r.policy ≠ .copy := by simpa [activeRole] using ha
    simp only [metadataScales,hp,↓reduceIte] at scales
    cases hk : t.keyMeta <;> cases hv : t.valueMeta <;> simp only [hk,hv] at scales
    all_goals try contradiction
    simp [output,hk,bytes,scales.2.2.1,elementBytes]
  | valueMeta =>
    have hp : r.policy ≠ .copy := by simpa [activeRole] using ha
    simp only [metadataScales,hp,↓reduceIte] at scales
    cases hk : t.keyMeta <;> cases hv : t.valueMeta <;> simp only [hk,hv] at scales
    all_goals try contradiction
    simp [output,hv,bytes,scales.2.2.2,elementBytes]

theorem input_width (t : TensorInputs) (r : Request) (h : InputContract t r)
    (v : Bool) : bytes (input t v)=4 := by
  cases v <;> simp [input,bytes,h.2.1.1,h.2.1.2.1]

theorem output_facts (t : TensorInputs) (r : Request) (h : InputContract t r)
    (role : Role) (ha : activeRole r role) :
    output t role∈dataRoles t r.policy ∧
    (output t role).metadata.numel=r.pages*r.block*r.heads*r.width role := by
  have scales := h.2.2.1
  cases role with
  | key => simp [output,dataRoles,VeriTile.Torch.TensorMetadata.numel,h.1.2.2.1,Request.width,Nat.mul_assoc]
  | value => simp [output,dataRoles,VeriTile.Torch.TensorMetadata.numel,h.1.2.2.2.1,Request.width,Nat.mul_assoc]
  | keyMeta =>
    have hp : r.policy ≠ .copy := by simpa [activeRole] using ha
    simp only [metadataScales,hp,↓reduceIte] at scales
    cases hk : t.keyMeta <;> cases hv : t.valueMeta <;> simp only [hk,hv] at scales
    simp [output,dataRoles,hp,hk,hv,VeriTile.Torch.TensorMetadata.numel,scales.1,Request.width,Nat.mul_assoc]
  | valueMeta =>
    have hp : r.policy ≠ .copy := by simpa [activeRole] using ha
    simp only [metadataScales,hp,↓reduceIte] at scales
    cases hk : t.keyMeta <;> cases hv : t.valueMeta <;> simp only [hk,hv] at scales
    simp [output,dataRoles,hp,hk,hv,VeriTile.Torch.TensorMetadata.numel,scales.2.1,Request.width,Nat.mul_assoc]

/-- Every formed store index, including scale/zero metadata, is signed-int64
exact. Padded pointer lanes use padIndex and therefore obey the same bound. -/
theorem output_index (t : TensorInputs) (r : Request) (h : InputContract t r)
    (a : Address) (b : Nat) (valid : ValidByte r a b) :
    (a.row*r.width a.role+a.column)*elementBytes r a.role+b<2^63 := by
  have facts := output_facts t r h a.role valid.2.2.1
  have limit := h.2.2.2.2.2.1.2.2.2 (output t a.role) facts.1
  rw [facts.2,output_width t r h a.role valid.2.2.1] at limit
  exact lt_trans (flat_bound _ _ _ _ (flat_bound _ _ _ _ valid.1 valid.2.1) valid.2.2.2) limit

theorem output_alignment (t : TensorInputs) (r : Request) (h : InputContract t r)
    (l : Layout r) (ins : InputLayout r l) (binding : Bindings t r l ins)
    (a : Address) (valid : ValidByte r a 0) :
    bytePtr r l a 0 % elementBytes r a.role=0 := by
  have facts := output_facts t r h a.role valid.2.2.1
  have checked := (h.2.2.2.1.2.2 (output t a.role) facts.1).2.2.2
  have positive : 0<(output t a.role).metadata.numel := by
    rw [facts.2]
    exact Nat.zero_lt_of_lt (flat_bound _ _ _ _ valid.1 valid.2.1)
  have aligned := checked.resolve_left (by omega)
  rw [output_width t r h a.role valid.2.2.1] at aligned
  unfold bytePtr
  rw [binding.outputPointer a.role valid.2.2.1 (by omega)]
  simp [Nat.add_mod,aligned]

theorem output_read (t : TensorInputs) (r : Request) (h : InputContract t r)
    (l : Layout r) (ins : InputLayout r l) (binding : Bindings t r l ins)
    (m : Nat → Byte) (a : Address) (b : Nat) (valid : ValidByte r a b) :
    m (bytePtr r l a b)=byte (output t a.role) m (a.row*r.width a.role+a.column) b := by
  have positive : 0<r.pages*r.block*r.heads*r.width a.role :=
    Nat.zero_lt_of_lt (flat_bound _ _ _ _ valid.1 valid.2.1)
  simp only [bytePtr,byte,output_width t r h a.role valid.2.2.1,
    binding.outputPointer a.role valid.2.2.1 positive]

theorem input_read (t : TensorInputs) (r : Request) (h : InputContract t r)
    (l : Layout r) (ins : InputLayout r l) (binding : Bindings t r l ins)
    (m : Nat → Byte) (v : Bool) (row col b : Nat)
    (hr : row<r.tokens*r.heads) (hc : col<stateWidth r v) :
    m (inputPtr r ins v row col b)=byte (input t v) m (row*stateWidth r v+col) b := by
  have positive : 0<r.tokens*r.heads*stateWidth r v :=
    Nat.zero_lt_of_lt (flat_bound _ _ _ _ hr hc)
  simp only [inputPtr,byte,input_width t r h v,binding.inputPointer v positive]

theorem input_alignment (t : TensorInputs) (r : Request) (h : InputContract t r)
    (l : Layout r) (ins : InputLayout r l) (binding : Bindings t r l ins)
    (v : Bool) (row col : Nat) (hr : row<r.tokens*r.heads) (hc : col<stateWidth r v) :
    inputPtr r ins v row col 0 % 4=0 := by
  have member : input t v∈dataRoles t r.policy := by cases v <;> simp [input,dataRoles]
  have size : (input t v).metadata.numel=r.tokens*r.heads*stateWidth r v := by
    cases v <;> simp [input,VeriTile.Torch.TensorMetadata.numel,h.1.1,h.1.2.1,stateWidth,Nat.mul_assoc]
  have positive : 0<r.tokens*r.heads*stateWidth r v :=
    Nat.zero_lt_of_lt (flat_bound _ _ _ _ hr hc)
  have checked := (h.2.2.2.1.2.2 (input t v) member).2.2.2
  have aligned := checked.resolve_left (by omega)
  rw [input_width t r h v] at aligned
  unfold inputPtr
  rw [binding.inputPointer v positive]
  simp [Nat.add_mod,aligned]

def value {α : Type} (decode : (Fin 4 → Byte) → α) (t : TensorInputs) (r : Request)
    (m : Nat → Byte) (v : Bool) (row col : Nat) : α :=
  decode (fun b => byte (input t v) m (row*stateWidth r v+col) b)

theorem readInput_eq {α : Type} (t : TensorInputs) (r : Request) (h : InputContract t r)
    (l : Layout r) (ins : InputLayout r l) (binding : Bindings t r l ins)
    (decode : (Fin 4 → Byte) → α) (zero : α) (m : Nat → Byte)
    (v : Bool) (row col : Nat) (hr : row<r.tokens*r.heads) (hc : col<stateWidth r v) :
    readInput r ins decode zero (fun a => m a.row) v row col=value decode t r m v row col := by
  simp only [readInput,hr,hc,and_self,↓reduceIte,value,physical]
  congr 1
  funext b
  exact input_read t r h l ins binding m v row col b hr hc

def values {α : Type} (decode : (Fin 4 → Byte) → α) (zero : α) (t : TensorInputs)
    (r : Request) (m : Nat → Byte) (v : Bool) (row col : Nat) : α :=
  if row<r.tokens*r.heads ∧ col<stateWidth r v then value decode t r m v row col else zero

theorem readInputs_eq {α : Type} (t : TensorInputs) (r : Request) (h : InputContract t r)
    (l : Layout r) (ins : InputLayout r l) (binding : Bindings t r l ins)
    (decode : (Fin 4 → Byte) → α) (zero : α) (m : Nat → Byte) :
    readInput r ins decode zero (fun a => m a.row)=values decode zero t r m := by
  funext v row col
  by_cases valid : row<r.tokens*r.heads ∧ col<stateWidth r v
  · simpa only [values,valid,↓reduceIte] using
      readInput_eq t r h l ins binding decode zero m v row col valid.1 valid.2
  · simp [readInput,values,valid]

/-- Python returns None and keeps every supplied Tensor object. The CUDA
device context is restored on normal return. No payload allocation occurs. -/
structure Result where
  memory : Nat → Byte
  value : Option Unit
  objects : List Nat
  device : VeriTile.Torch.Device

def run {α : Type} (r : Request) (l : Layout r) (ins : InputLayout r l)
    (ops : Operations α Byte) (kp : Plan r.dk) (vp : Plan r.dv)
    (kb : SourceBinding kp r.policy r.ck) (vb : SourceBinding vp r.policy r.cv)
    (encode : α → Nat → Byte) (decode : (Fin 4 → Byte) → α) (zero : α)
    (m : Nat → Byte) (objects : List Nat) (incoming : VeriTile.Torch.Device) : Result :=
  ⟨fun p => (call r l ins ops kp vp kb vb encode decode zero (fun a => m a.row)).1 (physical p),
    none,objects,incoming⟩

/-- Complete guarded source interpretation through the caller's Tensor
coordinates. Arithmetic operations are retained as source primitives. -/
theorem whole_tensor_source {α : Type} (t : TensorInputs) (r : Request)
    (checked : checkInputs t r = Bool.true) (l : Layout r) (ins : InputLayout r l)
    (binding : Bindings t r l ins) (ops : Operations α Byte)
    (kp : Plan r.dk) (vp : Plan r.dv)
    (kb : SourceBinding kp r.policy r.ck) (vb : SourceBinding vp r.policy r.cv)
    (encode : α → Nat → Byte) (decode : (Fin 4 → Byte) → α) (zero : α)
    (m : Nat → Byte) (objects : List Nat) (incoming : VeriTile.Torch.Device) :
    let result := run r l ins ops kp vp kb vb encode decode zero m objects incoming
    (∀ j∈jobs r, ∀ a∈rowAddresses r j, ∀ b,b<elementBytes r a.role →
      byte (output t a.role) result.memory (a.row*r.width a.role+a.column) b=
        cellByte encode (taskValue r ops kp vp (values decode zero t r m Bool.false)
          (values decode zero t r m Bool.true) j a) b) ∧
    (∀ p,(∀ j∈jobs r, ∀ a∈rowAddresses r j, ∀ b,b<elementBytes r a.role →
      bytePtr r l a b≠p) → result.memory p=m p) ∧
    result.value=none ∧ result.objects=objects ∧ result.device=incoming := by
  have h := (checkInputs_iff t r).mp checked
  have d := Native.domain h
  have exec := call_execution r d l ins ops kp vp kb vb encode decode zero m
  dsimp only
  refine ⟨?_,?_,rfl,rfl,rfl⟩
  · intro j hj a ha b hb
    have ai := (mem_rowAddresses r j a).mp ha
    rw [←output_read t r h l ins binding _ a b
      ⟨(row_address_bound r d j hj a ha).1,ai.2.1,ai.2.2,hb⟩]
    change (call r l ins ops kp vp kb vb encode decode zero (fun a=>m a.row)).1
      (physical (bytePtr r l a b))=_
    have ep := congrFun exec (bytePtr r l a b)
    simp only [Function.comp_apply] at ep
    rw [ep,whole_bytes_readback r d l ops kp vp _ _ encode m j hj a ha b hb]
    rw [readInputs_eq t r h l ins binding decode zero m]
  · intro p outside
    exact (congrFun exec p).trans (whole_bytes_frame r l ops kp vp _ _ encode m p outside)

#axiomsClean output_read
#axiomsClean input_read
#axiomsClean readInput_eq
#axiomsClean output_index
#axiomsClean output_alignment
#axiomsClean input_alignment
#axiomsClean whole_tensor_source
#specNonCircular values avoiding [call,run,runSourceJobs,executeBytes]
end VeriTile.Bench.Repairs.KvCacheFilling.Tensor
