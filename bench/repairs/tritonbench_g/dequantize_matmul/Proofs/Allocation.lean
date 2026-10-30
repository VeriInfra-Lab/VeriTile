import bench.repairs.tritonbench_g.dequantize_matmul.Proofs.Selection

namespace VeriTile.Bench.Repairs.DequantizeMatmul
noncomputable section
set_option maxHeartbeats 4000000

structure HostInputs where
  a : Location
  b : Location
  scale : Location
  device : Nat
  caller : Nat → Prop
  aLive : caller a.storage
  bLive : caller b.storage
  scaleLive : caller scale.storage

structure OutGuard (s : Shape) (x : HostInputs) (out : Descriptor) : Prop where
  rows : out.rows=s.M
  columns : out.columns=s.N
  device : out.device=x.device
  noA : x.a.storage≠out.location.storage
  noB : x.b.storage≠out.location.storage
  noScale : x.scale.storage≠out.location.storage

/-- Successful local torch.empty contract: fresh resolved contiguous FP32
storage on the requested device, with all pre-existing caller cells intact. -/
structure FreshAllocation (α : Type) (rows columns device : Nat) (live : Nat → Prop)
    (before : Memory (Cell α)) where
  tensor : Descriptor
  memory : Memory (Cell α)
  rowsEq : tensor.rows=rows
  columnsEq : tensor.columns=columns
  deviceEq : tensor.device=device
  fresh : ∀ storage,live storage → storage≠tensor.location.storage
  preserved : ∀ p,live p.1 → memory p=before p

abbrev Allocator (α : Type) := ∀ rows columns device live before,
  FreshAllocation α rows columns device live before

structure ChosenOutput (α : Type) (s : Shape) (x : HostInputs)
    (provided : Option Descriptor) (before : Memory (Cell α)) where
  tensor : Descriptor
  memory : Memory (Cell α)
  chosen : selectOutput provided tensor=tensor
  returnedProvided : ∀ out,provided=some out → tensor=out
  guard : OutGuard s x tensor
  fresh : provided=none → ∀ storage,x.caller storage → storage≠tensor.location.storage
  preserved : ∀ p,x.caller p.1 → memory p=before p

def chooseAndAllocate (allocate : Allocator α) (s : Shape) (x : HostInputs)
    (provided : Option Descriptor) (valid : ∀ out,provided=some out → OutGuard s x out)
    (before : Memory (Cell α)) : ChosenOutput α s x provided before := by
  cases provided with
  | some out =>
    refine ⟨out,before,rfl,?_,valid out rfl,?_,?_⟩
    · intro other same
      exact Option.some.inj same
    · intro impossible
      cases impossible
    · intro _ _; rfl
  | none =>
    let allocated := allocate s.M s.N x.device x.caller before
    refine ⟨allocated.tensor,allocated.memory,rfl,?_,?_,?_,allocated.preserved⟩
    · intro out impossible
      cases impossible
    · exact ⟨allocated.rowsEq,allocated.columnsEq,allocated.deviceEq,
        allocated.fresh x.a.storage x.aLive,allocated.fresh x.b.storage x.bLive,
        allocated.fresh x.scale.storage x.scaleLive⟩
    · intro _; exact allocated.fresh

structure Preparation (α : Type) (s : Shape) (provided : Option Descriptor)
    (before : Memory (Cell α)) where
  buffers : Buffers
  fresh : Descriptor
  memory : Memory (Cell α)
  allocation : Allocated s provided fresh buffers before memory

def prepare (allocate : Allocator α) (s : Shape) (x : HostInputs)
    (provided : Option Descriptor) (valid : ∀ out,provided=some out → OutGuard s x out)
    (before : Memory (Cell α)) : Preparation α s provided before := by
  let chosen := chooseAndAllocate allocate s x provided valid before
  let scratch := allocate s.K s.N x.device
    (fun storage => x.caller storage ∨ storage=chosen.tensor.location.storage) chosen.memory
  let buffers : Buffers := ⟨x.a,x.b,x.scale,scratch.tensor.location,chosen.tensor.location⟩
  refine ⟨buffers,chosen.tensor,scratch.memory,?_⟩
  refine {
    caller := x.caller
    selectedLocation := congrArg Descriptor.location chosen.chosen
    selectedRows := (congrArg Descriptor.rows chosen.chosen).trans chosen.guard.rows
    selectedColumns := (congrArg Descriptor.columns chosen.chosen).trans chosen.guard.columns
    scratchFresh := fun storage live => scratch.fresh storage (Or.inl live)
    freshOutput := chosen.fresh
    preserved := ?_
    separate := ?_
    inputA := x.aLive
    inputB := x.bLive
    inputScale := x.scaleLive
    outputA := chosen.guard.noA
    outputB := chosen.guard.noB
    outputScale := chosen.guard.noScale }
  · intro p live
    exact (scratch.preserved p (Or.inl live)).trans (chosen.preserved p live)
  · exact ⟨scratch.fresh x.a.storage (Or.inl x.aLive),
      scratch.fresh x.b.storage (Or.inl x.bLive),scratch.fresh x.scale.storage (Or.inl x.scaleLive),
      chosen.guard.noA,(scratch.fresh chosen.tensor.location.storage (Or.inr rfl)).symm⟩

/-- Exact host control: allocate c only for None; always allocate fresh fp_b;
run dequantization and selected torch.mm plan; return the selected c object. -/
def allocatedHostCall (a : Arithmetic α) (allocate : Allocator α) (s : Shape) (x : HostInputs)
    (provided : Option Descriptor) (valid : ∀ out,provided=some out → OutGuard s x out)
    (tile : Tile) (plan : LibraryPlan s) (before : Memory (Cell α)) : HostReturn α :=
  let prepared := prepare allocate s x provided valid before
  hostRun a s prepared.buffers tile plan provided prepared.fresh prepared.memory

theorem allocated_host_real (allocate : Allocator ℝ) (s : Shape) (x : HostInputs)
    (provided : Option Descriptor) (valid : ∀ out,provided=some out → OutGuard s x out)
    (tile : Tile) (plan : LibraryPlan s) (before : Memory (Cell ℝ)) (input : Input ℝ s)
    (rep : Represents input (prepare allocate s x provided valid before).buffers before)
    (r : Fin s.M) (n : Fin s.N) :
    let result := allocatedHostCall realArithmetic allocate s x provided valid tile plan before
    result.memory (result.tensor.location.addr (r.val*s.N+n.val))=.fp32 (mathematical input r n) := by
  exact host_real tile plan (prepare allocate s x provided valid before).allocation input rep r n

#axiomsClean allocated_host_real
#axiomsClean prepare
end
end VeriTile.Bench.Repairs.DequantizeMatmul
