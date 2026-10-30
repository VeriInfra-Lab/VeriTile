import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.PhysicalAdapter

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory (API Apart Write)
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 1000000
namespace PhysicalKernel
variable {α : Type} [Inhabited α]

/-- An internal input-memory invariant, containing only typed physical reads. -/
def Stable (access : Access α Store Byte) (zero : α) (buffers : Physical.Buffers)
    (initial current : Store) : Prop :=
  (∀ r i, Physical.read access.toAPI zero (buffers.input r) current i=
    Physical.read access.toAPI zero (buffers.input r) initial i) ∧
  (∀ r i, integer access (buffers.input r) current i=integer access (buffers.input r) initial i)

theorem Stable.refl (access : Access α Store Byte) (zero : α)
    (buffers : Physical.Buffers) (m : Store) : Stable access zero buffers m m :=
  ⟨fun _ _ => rfl,fun _ _ => rfl⟩

theorem stable_views (access : Access α Store Byte) (zero : α)
    (buffers : Physical.Buffers) (initial current : Store)
    (stable : Stable access zero buffers initial current) (id : Program) :
    Views (state access zero buffers current id) (Physical.inputs access.toAPI zero buffers initial)
      (metadata access (buffers.input .rows) initial) (metadata access (buffers.input .cols) initial) := by
  constructor
  all_goals intro address
  all_goals simp [state,Physical.inputs,metadata,stable.1,stable.2]

def Bounded (buffer : Physical.Buffer) (events : List (Write α)) : Prop :=
  ∀ e∈events,buffer.base≤e.address ∧ e.address+4≤buffer.base+buffer.storage.bytes

theorem bounded_apart (buffers : Physical.Buffers) (events : List (Write α))
    (bound : Bounded buffers.output events)
    (separate : ∀ r,buffers.output.base+buffers.output.storage.bytes≤(buffers.input r).base ∨
      (buffers.input r).base+(buffers.input r).storage.bytes≤buffers.output.base)
    (e : Write α) (he : e∈events) (r : Physical.Input) (i : Nat)
    (live : 4*((buffers.input r).storage.offset+i+1)≤(buffers.input r).storage.bytes) :
    Apart e.address (Physical.inputAddress (buffers.input r) i) := by
  have b := bound e he
  have s := separate r
  unfold Apart Physical.inputAddress
  omega

theorem read32_run_frame (access : Access α Store Byte) (events : List (Write α))
    (m : Store) (address : Nat) (apart : ∀ e∈events,Apart e.address address) :
    access.read32 (LayernormFwdTriton.Memory.run access.toAPI events m) address=access.read32 m address := by
  induction events generalizing m with
  | nil => rfl
  | cons e es ih =>
    change access.read32 (LayernormFwdTriton.Memory.run access.toAPI es
      (access.write m e.address e.value)) address=_
    rw [ih _ (fun q hq => apart q (List.mem_cons_of_mem _ hq))]
    exact access.read32_apart m _ _ _ (apart e (List.mem_cons_self ..))

theorem run_stable (access : Access α Store Byte) (zero : α) (buffers : Physical.Buffers)
    (events : List (Write α)) (bound : Bounded buffers.output events)
    (separate : ∀ r,buffers.output.base+buffers.output.storage.bytes≤(buffers.input r).base ∨
      (buffers.input r).base+(buffers.input r).storage.bytes≤buffers.output.base)
    (initial current : Store) (stable : Stable access zero buffers initial current) :
    Stable access zero buffers initial (LayernormFwdTriton.Memory.run access.toAPI events current) := by
  constructor
  · intro r i
    rw [← stable.1 r i]
    unfold Physical.read
    split_ifs with live
    · exact LayernormFwdTriton.Memory.run_read_frame access.toAPI events current _
        (fun e he => bounded_apart buffers events bound separate e he r i live)
    · rfl
  · intro r i
    rw [← stable.2 r i]
    unfold integer
    split_ifs with live
    · rw [read32_run_frame access events current _
        (fun e he => bounded_apart buffers events bound separate e he r i live)]
    · rfl

end PhysicalKernel
#axiomsClean PhysicalKernel.stable_views
#axiomsClean PhysicalKernel.bounded_apart
#axiomsClean PhysicalKernel.read32_run_frame
#axiomsClean PhysicalKernel.run_stable
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
