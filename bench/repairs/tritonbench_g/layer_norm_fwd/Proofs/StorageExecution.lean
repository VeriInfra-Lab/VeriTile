import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.StorageCompiler
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.StorageMemory

/-! Current-memory execution of actual emitted store dependencies. Every
instruction evaluates its source operands, performs its destination conversion,
then writes through the actual-width machine address. Fresh allocations will
justify the expanded dependency evaluation even after earlier stores. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.StorageMachine
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Address PhysicalDomain NativeABI WrapperNative
open Compiler (Destination)
noncomputable section
open scoped Classical
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000

def destinationRole : Destination → Role
  | .y _ => .y | .y1 _ => .y1 | .residualOut _ => .residualOut
  | .mean => .mean | .rstd => .rstd | .mask second _ => if second then .mask1 else .mask

def destinationCol : Destination → Nat
  | .y col | .y1 col | .residualOut col | .mask _ col => col
  | .mean | .rstd => 0

theorem destination_write (destination : Destination) : NativeMachineMemory.writeRole (destinationRole destination) := by
  cases destination with
  | y _ | y1 _ | residualOut _ | mean | rstd => trivial
  | mask second _ => cases second <;> trivial

structure Config (α : Type) (block : Nat) where
  ops : NativeRow.Ops α
  casts : StorageCompiler.StoreCasts α
  profile : StorageCompiler.Profile
  flags : Flags
  strides : Strides
  M : Nat
  N : Nat
  eps : α
  p : α
  allocations : RegionName → Allocation
  accepted : Accepted M N block
  storage : Domain flags preparedRegions strides M N allocations
  geometry : NativeGeometry.Geometry flags preparedRegions strides M N

abbrev Memory {α : Type} (c : Config α block) := TypedPhysical.Physical α (layout c.allocations)
abbrev Instruction {α : Type} (c : Config α block) := Fin c.M × Destination

def snapshot {α : Type} [Inhabited α] (c : Config α block) (memory : Memory c) : NativeRow.Inputs α where
  x := fun i j => (TypedPhysical.read (layout c.allocations) memory (preparedRegions.x,i*c.strides.x+j)).read .real
  x1 := fun i j => (TypedPhysical.read (layout c.allocations) memory (preparedRegions.x1,i*c.strides.x1+j)).read .real
  residual := fun i j => (TypedPhysical.read (layout c.allocations) memory (preparedRegions.residual,i*c.strides.residual+j)).read .real
  weight := fun j => (TypedPhysical.read (layout c.allocations) memory (preparedRegions.weight,j)).read .real
  bias := fun j => (TypedPhysical.read (layout c.allocations) memory (preparedRegions.bias,j)).read .real
  weight1 := fun j => (TypedPhysical.read (layout c.allocations) memory (preparedRegions.weight1,j)).read .real
  bias1 := fun j => (TypedPhysical.read (layout c.allocations) memory (preparedRegions.bias1,j)).read .real
  rowscale := fun i => (TypedPhysical.read (layout c.allocations) memory (preparedRegions.rowscale,i)).read .real
  seeds := fun i => (TypedPhysical.read (layout c.allocations) memory (preparedRegions.seeds,i)).read .int

theorem snapshot_view {α : Type} [Inhabited α] (c : Config α block) (memory : Memory c) (template : State α) :
    snapshot c memory=NativeSchedule.snapshot preparedRegions c.strides
      (TypedPhysical.pull (layout c.allocations) memory template) := rfl

def key {α : Type} (c : Config α block) (instruction : Instruction c) : AliasMemory.Key :=
  (region preparedRegions (destinationRole instruction.2),
    offset c.strides c.M c.N instruction.1.val (destinationCol instruction.2) (destinationRole instruction.2))

theorem store_enabled {α : Type} (c : Config α block)
    (artifact : StorageCompiler.Artifact c.profile c.flags c.N block) (destination : Destination)
    (operand : StorageCompiler.Operand) (h : artifact.stores destination=some operand) :
    active c.flags (destinationRole destination) ∧
    NativeAccess.laneActive c.N (destinationCol destination) (destinationRole destination) := by
  rw [artifact.storeOperands] at h
  cases destination with
  | y col => simpa [StorageCompiler.sourceStore,Compiler.sourceStore,destinationRole,destinationCol,active,NativeAccess.laneActive] using Option.isSome_iff_exists.mpr ⟨operand,h⟩
  | y1 col | residualOut col =>
    simpa [StorageCompiler.sourceStore,Compiler.sourceStore,destinationRole,destinationCol,active,NativeAccess.laneActive] using Option.isSome_iff_exists.mpr ⟨operand,h⟩
  | mean =>
    cases hf : c.flags.rms <;> simp_all [StorageCompiler.sourceStore,Compiler.sourceStore,destinationRole,destinationCol,active,NativeAccess.laneActive]
  | rstd => exact ⟨trivial,trivial⟩
  | mask second col =>
    cases second <;> simp_all [StorageCompiler.sourceStore,Compiler.sourceStore,destinationRole,destinationCol,active,NativeAccess.laneActive]

theorem store_column {α : Type} (c : Config α block) (destination : Destination)
    (h : NativeAccess.laneActive c.N (destinationCol destination) (destinationRole destination)) :
    destinationCol destination<block := by
  cases destination with
  | mean | rstd => exact c.accepted.blockPositive
  | y col | y1 col | residualOut col => exact lt_of_lt_of_le h c.accepted.width
  | mask second col => cases second <;> exact lt_of_lt_of_le h c.accepted.width

theorem store_written {α : Type} (c : Config α block)
    (artifact : StorageCompiler.Artifact c.profile c.flags c.N block) (instruction : Instruction c)
    (operand : StorageCompiler.Operand) (h : artifact.stores instruction.2=some operand) :
    AliasMemory.Writes c.flags preparedRegions c.strides c.M c.N (key c instruction) := by
  obtain ⟨ha,hl⟩ := store_enabled c artifact instruction.2 operand h
  exact NativeMachineMemory.role_written c.flags preparedRegions c.strides c.M c.N
    instruction.1.val (destinationCol instruction.2) instruction.1.isLt (destinationRole instruction.2)
    (destination_write instruction.2) ha hl

def step {α : Type} [Inhabited α] (c : Config α block)
    (artifact : StorageCompiler.Artifact c.profile c.flags c.N block) (memory : Memory c)
    (instruction : Instruction c) : Memory c :=
  match artifact.stores instruction.2 with
  | none => memory
  | some operand => store c.allocations preparedRegions c.strides c.M c.N instruction.1.val
      (destinationCol instruction.2) c.flags (destinationRole instruction.2) memory
      (operand.eval c.ops c.casts (snapshot c memory) c.M instruction.1.val c.eps c.p)

def event {α : Type} (c : Config α block)
    (artifact : StorageCompiler.Artifact c.profile c.flags c.N block) (inputs : NativeRow.Inputs α)
    (instruction : Instruction c) : Option (TypedPhysical.Event α) :=
  (artifact.stores instruction.2).map (fun operand =>
    ⟨key c instruction,operand.eval c.ops c.casts inputs c.M instruction.1.val c.eps c.p⟩)

theorem step_exact {α : Type} [Inhabited α] (c : Config α block)
    (artifact : StorageCompiler.Artifact c.profile c.flags c.N block) (memory : Memory c)
    (instruction : Instruction c) :
    step c artifact memory instruction=
      match event c artifact (snapshot c memory) instruction with
      | none => memory
      | some e => TypedPhysical.write (layout c.allocations) memory e.key e.value := by
  cases he : artifact.stores instruction.2 with
  | none => simp [step,event,he]
  | some operand =>
    have hh := store_enabled c artifact instruction.2 operand he
    simp only [step,event,he,Option.map_some]
    rw [store_exact c.flags preparedRegions c.strides c.M c.N block c.allocations c.accepted c.storage
      instruction.1.val (destinationCol instruction.2) instruction.1.isLt (store_column c instruction.2 hh.2)
      (destinationRole instruction.2),if_pos hh]
    rfl

def run {α : Type} [Inhabited α] (c : Config α block)
    (artifact : StorageCompiler.Artifact c.profile c.flags c.N block) : List (Instruction c) → Memory c → Memory c
  | [],memory => memory
  | instruction::rest,memory => run c artifact rest (step c artifact memory instruction)

def rowDestinations (block : Nat) : List Destination :=
  (List.range block).map (.mask Bool.false)++(List.range block).map (.mask Bool.true)++
    (List.range block).map .residualOut++[.mean,.rstd]++
    (List.range block).map .y++(List.range block).map .y1

/-- Source-order stores retain scalar mean/rstd stores for N=0. -/
def program {α : Type} (c : Config α block) : List (Instruction c) :=
  (List.finRange c.M).flatMap (fun row => (rowDestinations block).map (fun destination => (row,destination)))

def execute {α : Type} [Inhabited α] (c : Config α block)
    (artifact : StorageCompiler.Artifact c.profile c.flags c.N block) (memory : Memory c) : Memory c :=
  run c artifact (program c) memory

#axiomsClean destination_write
#axiomsClean snapshot_view
#axiomsClean store_enabled
#axiomsClean store_column
#axiomsClean store_written
#axiomsClean step_exact
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.StorageMachine
