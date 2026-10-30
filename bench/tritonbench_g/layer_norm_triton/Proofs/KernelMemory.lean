import bench.tritonbench_g.layer_norm_triton.Proofs.KernelGraph
import bench.tritonbench_g.layer_norm_triton.Contract

/-! Kernel-only Tensor and memory interface. No allocator, stream, host call,
saved-version receipt, or assumed successful execution is part of this module.
Input/InputGuard describe the actual contiguous Tensor view, including its
backing, storage offset, logical coordinates and observed data pointer. -/
namespace VeriTile.Bench.Repairs.LayerNormTriton.KernelMemory
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm hiding Ops Plan
open Storage
noncomputable section
open scoped Classical

structure Input where
  dense : Dense
  metadata : VeriTile.Torch.TensorMetadata
  shape : metadata.shape=dense.tensor.view.shape
  device : metadata.device.index=dense.tensor.view.device
  negative : metadata.isNeg=dense.tensor.view.negative
  pointer : 0<metadata.numel → metadata.dataPtr.toNat=dense.address 0

/-- The selected Python guard is unchanged. No alignment of the original
storage base, no copy, and no input disjointness premise is introduced. -/
def InputGuard (x : Input) : Prop := Contract.TensorContract x.metadata

theorem input_read (api : Memory.API α Store Byte) (neg : α → α)
    (x : Input) (m : World Store) (i : Nat) (hi : i<x.dense.count) :
    api.read (m x.dense.owner) (x.dense.address i)=logical api neg m x.dense.tensor i :=
  (dense_logical api neg m x.dense i hi).symm

theorem input_pointer (x : Input) (i : Nat) (hi : i<x.dense.count) :
    x.dense.address i=x.metadata.dataPtr.toNat+4*i := by
  have hm : 0<x.metadata.numel := by
    rw [VeriTile.Torch.TensorMetadata.numel,x.shape,←x.dense.countEq]
    omega
  rw [x.pointer hm]
  simp [Dense.address]

/-- Logical input on every valid Tensor coordinate; the extension outside the
Tensor is irrelevant to masked accesses and keeps the load equation total. -/
def inputValue (api : Memory.API α Store Byte) (neg : α → α)
    (x : Input) (m : World Store) (i : Nat) : α :=
  if i<x.dense.count then logical api neg m x.dense.tensor i
  else api.read (m x.dense.owner) (x.dense.address i)

theorem inputValue_load (api : Memory.API α Store Byte) (neg : α → α)
    (x : Input) (m : World Store) (i : Nat) :
    inputValue api neg x m i=api.read (m x.dense.owner) (x.dense.address i) := by
  unfold inputValue
  split
  · exact dense_logical api neg m x.dense i ‹_›
  · rfl

theorem inputValue_logical (api : Memory.API α Store Byte) (neg : α → α)
    (x : Input) (m : World Store) (i : Nat) (hi : i<x.dense.count) :
    inputValue api neg x m i=logical api neg m x.dense.tensor i := by
  simp only [inputValue,hi,if_true]

/-- Canonical complete enumeration of all writes. Coverage is constructed,
never supplied as an execution or scheduler-success premise. -/
def schedule (p : Execution.Buffers K inputs) : Execution.E.Schedule K where
  objectIdentity := 0
  owner := p.owner
  stream := 0
  grid := []
  pointers := []
  order := (List.finRange K).flatMap fun k => (List.range (p.output k).count).map (fun i => (k,i))
  status := .ok ()

theorem schedule_covered (p : Execution.Buffers K inputs) (k : Fin K) (i : Nat)
    (hi : i<(p.output k).count) : (k,i)∈(schedule p).order := by
  simp only [schedule,List.mem_flatMap,List.mem_finRange,true_and,List.mem_map,List.mem_range]
  exact ⟨k,i,hi,rfl⟩

def writes (p : Execution.Buffers K inputs) (values : ∀ k,Fin (p.output k).count → α) :=
  VeriTile.Bench.Repairs.FastLayernorm.Execution.scheduledWrites (p.trace values) (schedule p)

def run (p : Execution.Buffers K inputs) (api : Memory.API α Store Byte)
    (values : ∀ k,Fin (p.output k).count → α) (m : World Store) : World Store :=
  p.run api values (schedule p) m

/-- A total independent graph interpreter: storage is supplied as valid Dense
views and every listed write is constructed from a bounded output coordinate. -/
def exec (p : Execution.Buffers K inputs) (api : Memory.API α Store Byte)
    (values : ∀ k,Fin (p.output k).count → α) (m : World Store) : Except Unit (World Store) :=
  .ok (run p api values m)

def Outside (p : Execution.Buffers K inputs) (key : Key) : Prop :=
  key.1≠p.owner ∨ ∀ k i,i<(p.output k).count →
    key.2<(p.output k).address i ∨ (p.output k).address i+4≤key.2

theorem run_read (p : Execution.Buffers K inputs) (api : Memory.API α Store Byte)
    (neg : α → α) (values : ∀ k,Fin (p.output k).count → α) (m : World Store)
    (k : Fin K) (i : Fin (p.output k).count) :
    logical api neg (run p api values m) (p.output k).tensor i.val=values k i := by
  rw [dense_logical api neg _ _ _ i.isLt,p.owned k]
  exact Execution.run_readback p api values (schedule p) (schedule_covered p) m k i

theorem run_outside (p : Execution.Buffers K inputs) (api : Memory.API α Store Byte)
    (values : ∀ k,Fin (p.output k).count → α) (m : World Store) (key : Key) (outside : Outside p key) :
    observe api (run p api values m) key=observe api m key := by
  apply runWorld_frame
  intro owner e he
  obtain ⟨k,i,rfl⟩ := (VeriTile.Bench.Repairs.FastLayernorm.Execution.trace_mem (p.trace values) e).mp
    (VeriTile.Bench.Repairs.FastLayernorm.Execution.scheduled_subset (p.trace values) (schedule p) e he)
  rcases outside with h|h
  · exact False.elim (h owner)
  · exact h k i.val i.isLt

theorem run_frame (p : Execution.Buffers K inputs) (api : Memory.API α Store Byte)
    (values : ∀ k,Fin (p.output k).count → α) (m : World Store) (key : Key) (hk : caller inputs key) :
    observe api (run p api values m) key=observe api m key :=
  Execution.run_frame p api values (schedule p) m key hk

/-- All output Tensor values, exact complete memory, every exterior byte, and
complete caller backing storage. No source-execution premise occurs here. -/
theorem executes (p : Execution.Buffers K inputs) (api : Memory.API α Store Byte)
    (neg : α → α) (values : ∀ k,Fin (p.output k).count → α) (m : World Store) :
    ∃ final,exec p api values m=.ok final ∧
      final=runWorld api p.owner (writes p values) m ∧
      (∀ k i,logical api neg final (p.output k).tensor i.val=values k i) ∧
      (∀ key,Outside p key → observe api final key=observe api m key) ∧
      (∀ key,caller inputs key → observe api final key=observe api m key) := by
  exact ⟨run p api values m,rfl,rfl,run_read p api neg values m,
    run_outside p api values m,run_frame p api values m⟩

#axiomsClean input_read
#axiomsClean input_pointer
#axiomsClean schedule_covered
#axiomsClean executes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.KernelMemory
