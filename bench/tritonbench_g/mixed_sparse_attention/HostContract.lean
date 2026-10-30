import bench.tritonbench_g.mixed_sparse_attention.Proofs.Common

namespace VeriTile.Bench.Repairs.MixedSparseAttention.Checked
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.MixedSparseAttention.Whole002 Source Whole
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000

inductive Role where
  | query | key | value | sequence | blocks | offsets | counts | columns | output
  deriving DecidableEq, Fintype
def Role.region (c : Config) : Role → RegionName
  | .query=>c.q | .key=>c.k | .value=>c.v | .sequence=>Region.cast c.seq
  | .blocks=>Region.cast c.blocks | .offsets=>Region.cast c.offsets
  | .counts=>Region.cast c.counts | .columns=>Region.cast c.columns | .output=>c.out
def Role.shape (c : Config) : Role → List Nat
  | .query | .key | .value | .output=>[c.Z,c.H,c.T,c.BD]
  | .sequence=>[c.Z]
  | .blocks | .counts=>[c.Z,c.H,c.NR]
  | .offsets=>[c.Z,c.H,c.NR,c.NS]
  | .columns=>[c.Z,c.H,c.NR,c.NV]
def Role.size (c : Config) : Role → Nat
  | .query | .key | .value | .output=>tensorSize c
  | .sequence=>c.Z | .blocks | .counts=>c.Z*c.H*c.NR
  | .offsets=>c.Z*c.H*c.NR*c.NS | .columns=>c.Z*c.H*c.NR*c.NV
def Role.floating : Role → Bool
  | .query | .key | .value | .output=>Bool.true | _=>Bool.false

structure Tensor where
  object : Nat
  shape : List Nat
  strides : List Nat
  floating : Bool
  cuda : Bool
  device : Nat
  contiguous : Bool
  strided : Bool
  negative : Bool
  conjugate : Bool
  requiresGrad : Bool
  pointer : Nat
  storageBase : Nat
  storageBytes : Nat
  storageOffset : Nat

structure Request where
  tensor : Role → Tensor
  scaleWord : BitVec 32

def denseStrides (c : Config) := [c.H*c.T*c.BD,c.T*c.BD,c.BD,1]

structure Schema (c : Config) (request : Request) : Prop where
  selected : c.storage=.fp16 ∧ c.BM=64 ∧ c.BN=64
  heads : 0<c.H
  dim : c.BD∈[16,32,64,128]
  rows : c.NR=(c.T+c.BM-1)/c.BM
  scale : (Float32Bits.mk request.scaleWord).decodeReal=some c.scale
  header : ∀ role,let t:=request.tensor role
    t.shape=role.shape c ∧ t.floating=role.floating ∧ t.cuda=Bool.true ∧
    t.device=(request.tensor .query).device ∧ t.contiguous=Bool.true ∧
    t.negative=Bool.false ∧ t.conjugate=Bool.false ∧ t.strided=Bool.true
  inference : ∀ r∈[Role.query,.key,.value],(request.tensor r).requiresGrad=Bool.false
  dimensions : ∀ n∈[c.Z,c.H,c.T,c.BD,c.NR,c.NS,c.NV],n<2^31
  queryStrides : [c.sqz,c.sqh,c.sqm,c.sqk]=(request.tensor .query).strides
  keyStrides : [c.skz,c.skh,c.skn,c.skk]=(request.tensor .key).strides
  valueStrides : [c.svz,c.svh,c.svn,c.svk]=(request.tensor .value).strides
  outputStrides : [c.soz,c.soh,c.som,c.sok]=(request.tensor .output).strides

abbrev Observed := Role → Nat → Int

/-- Exactly the signed integer arrays inspected by the checked wrapper.
No floating input or output value occurs in the guard. -/
structure MetadataGuard (c : Config) (values : Observed) : Prop where
  sequence : ∀ b<c.Z,0≤values .sequence b ∧ values .sequence b≤c.T
  blocks : ∀ hz<c.Z*c.H,∀ row<c.NR,
    0≤values .blocks (hz*c.NR+row) ∧ values .blocks (hz*c.NR+row)≤c.NS
  counts : ∀ hz<c.Z*c.H,∀ row<c.NR,
    0≤values .counts (hz*c.NR+row) ∧ values .counts (hz*c.NR+row)≤c.NV
  offsets : ∀ hz<c.Z*c.H,∀ row<c.NR,∀ j,
    j<(values .blocks (hz*c.NR+row)).toNat →
    0≤values .offsets ((hz*c.NR+row)*c.NS+j) ∧
    values .offsets ((hz*c.NR+row)*c.NS+j)<values .sequence (hz/c.H)
  columns : ∀ hz<c.Z*c.H,∀ row<c.NR,∀ j,
    j<(values .counts (hz*c.NR+row)).toNat →
    0≤values .columns ((hz*c.NR+row)*c.NV+j) ∧
    values .columns ((hz*c.NR+row)*c.NV+j)<values .sequence (hz/c.H)

/-- Local input interpretation of the inspected signed metadata. Negative
words are excluded by MetadataGuard before the source's natural semantics. -/
structure MetadataBinding {α : Type} [Inhabited α] (c : Config) (values : Observed) (s : State α) : Prop where
  read : ∀ r∈[Role.sequence,.blocks,.offsets,.counts,.columns],∀ i,
    (s.mem (r.region c) i).read .nat=(values r i).toNat

theorem metadata {α : Type} [Inhabited α] (c : Config) (values : Observed) (s : State α)
    (guard : MetadataGuard c values) (binding : MetadataBinding c values s) : Metadata c s := by
  have hs:=binding.read .sequence (by simp)
  have hb:=binding.read .blocks (by simp)
  have hc:=binding.read .counts (by simp)
  have ho:=binding.read .offsets (by simp)
  have hi:=binding.read .columns (by simp)
  simp only [Role.region] at hs hb hc ho hi
  constructor
  · intro b hbound
    rw [hs]
    have hb:=hbound
    have h:=guard.sequence b hb
    simp only [Value] at *
    omega
  · intro hz hh row hr
    rw [hb]
    have h:=guard.blocks hz hh row hr
    simp only [Value] at *
    omega
  · intro hz hh row hr
    rw [hc]
    have h:=guard.counts hz hh row hr
    simp only [Value] at *
    omega
  · intro hz hh row hr j hj
    rw [hb] at hj
    rw [ho,hs]
    have h:=guard.offsets hz hh row hr j hj
    simp only [Value] at *
    omega
  · intro hz hh row hr j hj
    rw [hc] at hj
    rw [hi,hs]
    have h:=guard.columns hz hh row hr j hj
    simp only [Value] at *
    omega

def inputRoles : List Role := [.query,.key,.value,.sequence,.blocks,.offsets,.counts,.columns]

structure Admission (c : Config) (request : Request) (values : Observed) : Prop where
  schema : Schema c request
  nonempty : 0<c.Z ∧ 0<c.T
  separated : Separated c
  metadata : MetadataGuard c values
  grid : c.Z*c.H≤65535
  strides : ∀ r∈[Role.query,.key,.value,.output],(request.tensor r).strides=denseStrides c
  envelope : Machine.envelope c<2147483648
  pointers : ∀ r,(request.tensor r).pointer%4=0 ∧
    (request.tensor r).pointer+4*Machine.envelope c<18446744073709551616
  storage : ∀ r,(request.tensor r).pointer=(request.tensor r).storageBase+4*(request.tensor r).storageOffset ∧
    4*((request.tensor r).storageOffset+r.size c)≤(request.tensor r).storageBytes ∧
    (request.tensor r).storageBase+(request.tensor r).storageBytes≤2^64
  fresh : ∀ r∈inputRoles,
    (request.tensor r).storageBase+(request.tensor r).storageBytes≤(request.tensor .output).pointer ∨
    (request.tensor .output).pointer+4*tensorSize c≤(request.tensor r).storageBase

theorem Admission.contiguous (h : Admission c request values) : Contiguous c := by
  have hq:=h.schema.queryStrides.trans (h.strides .query (by simp))
  have hk:=h.schema.keyStrides.trans (h.strides .key (by simp))
  have hv:=h.schema.valueStrides.trans (h.strides .value (by simp))
  have ho:=h.schema.outputStrides.trans (h.strides .output (by simp))
  simp only [denseStrides,List.cons.injEq,and_true] at hq hk hv ho
  exact ⟨hq.1,hq.2.1,hq.2.2.1,hq.2.2.2,hk.1,hk.2.1,hk.2.2.1,hk.2.2.2,
    hv.1,hv.2.1,hv.2.2.1,hv.2.2.2,ho.1,ho.2.1,ho.2.2.1,ho.2.2.2⟩

theorem Admission.shape (h : Admission c request values) : Shape c :=
  ⟨h.nonempty.1,h.schema.heads,by simp [h.schema.selected.2.1],
    by simp [h.schema.selected.2.2],h.schema.dim,h.schema.rows,h.contiguous⟩

/-- Role reification binds only initial capacities, header addresses and
metadata. It may represent shared physical read-only backing without copying. -/
structure InputBinding {α : Type} [Inhabited α] (c : Config) (request : Request)
    (values : Observed) (before : Physical.Store α) (base : RegionName → Nat) : Prop where
  capacities : ∀ r∈inputRoles,before.capacity (r.region c)=r.size c
  fresh : before.capacity c.out=0
  bases : ∀ r,base (r.region c)=(request.tensor r).pointer
  metadata : MetadataBinding c values before.extend

variable {α : Type} [Inhabited α] {c : Config} {request : Request} {values : Observed}
  {before : Physical.Store α} {base : RegionName → Nat}

theorem machine (accepted : Admission c request values) (binding : InputBinding c request values before base) :
    Machine.Domain c base := by
  refine ⟨accepted.envelope,?_⟩
  intro r width hw
  simp only [Machine.bindings,List.mem_cons,List.not_mem_nil,or_false,Prod.mk.injEq] at hw
  rcases hw with h|h|h|h|h|h|h|h|h
  all_goals rcases h with ⟨rfl,rfl⟩
  all_goals first
    | simpa only [←binding.bases .query] using accepted.pointers .query
    | simpa only [←binding.bases .key] using accepted.pointers .key
    | simpa only [←binding.bases .value] using accepted.pointers .value
    | simpa only [←binding.bases .output] using accepted.pointers .output
    | simpa only [←binding.bases .sequence] using accepted.pointers .sequence
    | simpa only [←binding.bases .blocks] using accepted.pointers .blocks
    | simpa only [←binding.bases .offsets] using accepted.pointers .offsets
    | simpa only [←binding.bases .counts] using accepted.pointers .counts
    | simpa only [←binding.bases .columns] using accepted.pointers .columns

theorem storage (accepted : Admission c request values) (binding : InputBinding c request values before base) :
    Access.Storage c (Physical.allocatedCapacity c before) := by
  have cap (r : Role) (hr : r∈inputRoles) :
      Physical.allocatedCapacity c before (r.region c)=r.size c := by
    have apart : r.region c≠c.out := accepted.separated.inputs _ (by
      simp only [inputRoles,List.mem_cons,List.not_mem_nil,or_false] at hr
      rcases hr with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> simp [Role.region])
    simp only [Physical.allocatedCapacity,if_neg apart,binding.capacities r hr]
  constructor
  · intro r hr
    simp only [List.mem_cons,List.not_mem_nil,or_false] at hr
    rcases hr with rfl|rfl|rfl|rfl
    · exact le_of_eq (cap .query (by simp [inputRoles])).symm
    · exact le_of_eq (cap .key (by simp [inputRoles])).symm
    · exact le_of_eq (cap .value (by simp [inputRoles])).symm
    · simp [Physical.allocatedCapacity]
  · exact le_of_eq (cap .sequence (by simp [inputRoles])).symm
  · exact le_of_eq (cap .blocks (by simp [inputRoles])).symm
  · exact le_of_eq (cap .counts (by simp [inputRoles])).symm
  · exact le_of_eq (cap .offsets (by simp [inputRoles])).symm
  · exact le_of_eq (cap .columns (by simp [inputRoles])).symm

theorem input_byte_separate (accepted : Admission c request values)
    (binding : InputBinding c request values before base) (r : Role) (hr : r∈inputRoles) :
    base (r.region c)+4*before.capacity (r.region c)≤base c.out ∨
    base c.out+4*tensorSize c≤base (r.region c) := by
  have fresh:=accepted.fresh r hr
  have storage:=accepted.storage r
  rw [binding.capacities r hr,binding.bases r,show base c.out=(request.tensor .output).pointer from binding.bases .output]
  omega

/-- All former shape/metadata/allocation/machine preconditions now follow
from the checked wrapper data and ordinary local input-role interpretation. -/
theorem to_contract (accepted : Admission c request values)
    (binding : InputBinding c request values before base) : Contract c before base := by
  refine ⟨accepted.schema.selected.1,accepted.shape,accepted.separated,binding.fresh,
    metadata c values before.extend accepted.metadata binding.metadata,
    storage accepted binding,machine accepted binding,?_⟩
  intro r width hw apart
  simp only [Machine.bindings,List.mem_cons,List.not_mem_nil,or_false,Prod.mk.injEq] at hw
  rcases hw with h|h|h|h|h|h|h|h|h
  all_goals rcases h with ⟨rfl,rfl⟩
  all_goals first
    | exact False.elim (apart rfl)
    | exact input_byte_separate accepted binding .query (by simp [inputRoles])
    | exact input_byte_separate accepted binding .key (by simp [inputRoles])
    | exact input_byte_separate accepted binding .value (by simp [inputRoles])
    | exact input_byte_separate accepted binding .sequence (by simp [inputRoles])
    | exact input_byte_separate accepted binding .blocks (by simp [inputRoles])
    | exact input_byte_separate accepted binding .offsets (by simp [inputRoles])
    | exact input_byte_separate accepted binding .counts (by simp [inputRoles])
    | exact input_byte_separate accepted binding .columns (by simp [inputRoles])

structure EmptyAdmission (c : Config) (request : Request) : Prop where
  schema : Schema c request
  empty : c.Z=0 ∨ c.T=0

def validate (c : Config) (request : Request) (values : Observed) : Except Unit Bool :=
  if c.Z=0 ∨ c.T=0 then
    if EmptyAdmission c request then .ok Bool.false else .error ()
  else if Admission c request values then .ok Bool.true else .error ()

theorem validate_launch : validate c request values=.ok Bool.true ↔
    ¬(c.Z=0 ∨ c.T=0) ∧ Admission c request values := by
  unfold validate
  split_ifs <;> simp_all

theorem validate_empty : validate c request values=.ok Bool.false ↔ EmptyAdmission c request := by
  unfold validate
  by_cases empty : c.Z=0 ∨ c.T=0
  · simp [empty]
  · have no : ¬EmptyAdmission c request := fun h=>empty h.empty
    simp only [if_neg empty]
    split_ifs <;> simp [no]

#axiomsClean metadata
#axiomsClean machine
#axiomsClean storage
#axiomsClean to_contract
#axiomsClean validate_launch
#axiomsClean validate_empty
#specNonCircular MetadataGuard avoiding [FP.Structural.exec,Whole.invoke,Source.finalValues]
#specNonCircular Admission avoiding [FP.Structural.exec,Whole.invoke,Source.finalValues]
#specNonCircular InputBinding avoiding [FP.Structural.exec,Whole.invoke,Source.finalValues]
end
end VeriTile.Bench.Repairs.MixedSparseAttention.Checked
