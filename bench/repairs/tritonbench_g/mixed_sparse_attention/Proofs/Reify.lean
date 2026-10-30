import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.RawMemory

namespace VeriTile.Bench.Repairs.MixedSparseAttention.Checked.Reify
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.MixedSparseAttention.Whole002 Source Whole
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000

/-- Symbolic region names name argument roles. Physical pointers may alias;
the role map creates no allocation and copies no payload. -/
structure Layout (c : Config) where
  route : RegionName → Option Role
  inputs : ∀ r∈inputRoles,route (r.region c)=some r
  output : route c.out=none

def capacity (c : Config) (layout : Layout c) (r : RegionName) : Nat :=
  match layout.route r with | some role=>role.size c | none=>0

def base (c : Config) (request : Request) (layout : Layout c) (r : RegionName) : Nat :=
  match layout.route r with
  | some role=>(request.tensor role).pointer
  | none=>(request.tensor .output).pointer

def observed (c : Config) (integer : α → Int) (payload : Raw.Payload α) : Observed :=
  fun r i=>if i<r.size c then integer (payload r i) else 0

def cell (encode : α → β) (integer : α → Int) (payload : Raw.Payload α) (r : Role) (i : Nat) : Cell β :=
  if r.floating then .mk .real (encode (payload r i)) else .mk .nat (integer (payload r i)).toNat

def prepare (c : Config) (layout : Layout c) (encode : α → β) (integer : α → Int)
    (zero : β) (payload : Raw.Payload α) : Physical.Store β where
  capacity:=capacity c layout
  cells:=fun r i=>match layout.route r with
    | some role=>cell encode integer payload role i.val
    | none=>.mk .real zero
  outside:=.mk .real zero

theorem capacity_input (layout : Layout c) (r : Role) (hr : r∈inputRoles) :
    capacity c layout (r.region c)=r.size c := by simp [capacity,layout.inputs r hr]

theorem input_cell (layout : Layout c) (encode : α → β) (integer : α → Int)
    (zero : β) (payload : Raw.Payload α) (r : Role) (hr : r∈inputRoles)
    (i : Fin ((prepare c layout encode integer zero payload).capacity (r.region c))) :
    (prepare c layout encode integer zero payload).cells (r.region c) i=cell encode integer payload r i.val := by
  simp [prepare,layout.inputs r hr]

theorem input_extend [Inhabited β] (layout : Layout c) (encode : α → β) (integer : α → Int)
    (zero : β) (payload : Raw.Payload α) (r : Role) (hr : r∈inputRoles) (i : Nat) (hi : i<r.size c) :
    (prepare c layout encode integer zero payload).extend.mem (r.region c) i=cell encode integer payload r i := by
  simp [Physical.Store.extend,prepare,capacity,layout.inputs r hr,hi]

/-- The branch used by every enabled source load reads the current physical
word at the supplied pointer, including mixed-dtype read-only aliases. -/
theorem physical_input [Inhabited β] (api : VeriTile.Bench.Repairs.LayernormFwdTriton.Memory.API α Store Byte)
    (rawZero : α) (layout : Layout c) (encode : α → β) (integer : α → Int) (zero : β)
    (request : Request) (memory : Store) (r : Role) (hr : r∈inputRoles) (i : Nat) (hi : i<r.size c) :
    (prepare c layout encode integer zero (Raw.inputs api rawZero c request memory)).extend.mem (r.region c) i=
      if r.floating then .mk .real (encode (api.read memory ((request.tensor r).pointer+4*i)))
      else .mk .nat (integer (api.read memory ((request.tensor r).pointer+4*i))).toNat := by
  rw [input_extend layout encode integer zero _ r hr i hi]
  have hout : r≠.output := by cases r <;> simp_all [inputRoles]
  simp [cell,Raw.inputs,hout,hi]

theorem binding [Inhabited β] (c : Config) (request : Request) (layout : Layout c)
    (encode : α → β) (integer : α → Int) (zero : β) (payload : Raw.Payload α) :
    InputBinding c request (observed c integer payload) (prepare c layout encode integer zero payload)
      (base c request layout) := by
  constructor
  · intro r hr;exact capacity_input layout r hr
  · simp [prepare,capacity,layout.output]
  · intro r
    by_cases hr : r=.output
    · subst r;simp [base,Role.region,layout.output]
    · have member : r∈inputRoles := by cases r <;> simp_all [inputRoles]
      simp [base,layout.inputs r member]
  · constructor
    intro r hr i
    have member : r∈inputRoles := by
      simp only [List.mem_cons,List.not_mem_nil,or_false] at hr
      rcases hr with rfl|rfl|rfl|rfl|rfl <;> simp [inputRoles]
    have hf : r.floating=Bool.false := by
      simp only [List.mem_cons,List.not_mem_nil,or_false] at hr
      rcases hr with rfl|rfl|rfl|rfl|rfl <;> rfl
    by_cases hi : i<r.size c
    · simp [Physical.Store.extend,prepare,capacity,layout.inputs r member,hi,cell,hf,observed]
    · simp [Physical.Store.extend,prepare,capacity,layout.inputs r member,hi,observed,Cell.read,defaultValue]

theorem allocated_input (layout : Layout c)
    (r : Role) (hr : r∈inputRoles) (encode : α → β) (integer : α → Int) (zero : β) (payload : Raw.Payload α) :
    Physical.allocatedCapacity c (prepare c layout encode integer zero payload) (r.region c)=r.size c := by
  have apart : r.region c≠c.out := by
    intro eq
    have h:=layout.inputs r hr
    rw [eq,layout.output] at h
    contradiction
  simp [Physical.allocatedCapacity,apart,prepare,capacity,layout.inputs r hr]

/-- Every source load family is proved in bounds by Access.Safe. Combining
that bound with the final conjunct identifies its enabled load with a current
raw-memory word, so the total reader's fallback is unreachable. -/
theorem checked_access [Inhabited β]
    (api : VeriTile.Bench.Repairs.LayernormFwdTriton.Memory.API α Store Byte)
    (rawZero : α) (c : Config) (layout : Layout c) (encode : α → β) (integer : α → Int) (zero : β)
    (request : Request) (memory : Store)
    (accepted : Admission c request (observed c integer (Raw.inputs api rawZero c request memory))) :
    let before:=prepare c layout encode integer zero (Raw.inputs api rawZero c request memory)
    Access.Safe c (Physical.allocatedCapacity c before) before.extend ∧
    Machine.Closed c (base c request layout) ∧
    (∀ r∈inputRoles,∀ i,i<Physical.allocatedCapacity c before (r.region c) →
      before.extend.mem (r.region c) i=
        if r.floating then .mk .real (encode (api.read memory ((request.tensor r).pointer+4*i)))
        else .mk .nat (integer (api.read memory ((request.tensor r).pointer+4*i))).toNat) := by
  let before:=prepare c layout encode integer zero (Raw.inputs api rawZero c request memory)
  have h:=to_contract accepted (binding c request layout encode integer zero _)
  refine ⟨Access.safe c h.shape _ h.storage before.extend h.metadata,
    Machine.closed c _ h.machine,?_⟩
  intro r hr i hi
  have bound : i<r.size c := by
    simpa only [before,allocated_input layout r hr] using hi
  exact physical_input api rawZero layout encode integer zero request memory r hr i bound

#axiomsClean binding
#axiomsClean physical_input
#axiomsClean checked_access
#specNonCircular Layout avoiding [FP.Structural.exec,Whole.invoke,Source.finalValues]
#specNonCircular prepare avoiding [FP.Structural.exec,Whole.invoke,Source.finalValues]
end
end VeriTile.Bench.Repairs.MixedSparseAttention.Checked.Reify
