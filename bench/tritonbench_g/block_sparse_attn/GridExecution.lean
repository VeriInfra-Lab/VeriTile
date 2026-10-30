import bench.tritonbench_g.block_sparse_attn.StoreTrace
import bench.tritonbench_g.block_sparse_attn.PhysicalGeometry
import bench.tritonbench_g.block_sparse_attn.MemoryFrame

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory (API Apart Write)
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 3000000
set_option linter.unusedSectionVars false
set_option linter.unusedSimpArgs false
namespace PhysicalKernel
variable {α : Type} [Inhabited α]

theorem output_stable (access : Access α Store Byte) (a : Arithmetic α) (native : Bool)
    (p : Parameters) (buffers : Physical.Buffers) (initial current : Store)
    (stable : Stable access a.zero buffers initial current) (id : Program) :
    output a native p (state access a.zero buffers current id)=
      output a native p (state access a.zero buffers initial id) := by
  unfold output accumulation visitCount first last csrColumn update scoreAccumulation scoreStep query key value
  simp [qPointer,kPointer,vPointer,rowPointer,batch,head,kv,
    layout,row,channel,state,stable.1,stable.2]

def events (access : Access α Store Byte) (a : Arithmetic α) (native : Bool)
    (p : Parameters) (buffers : Physical.Buffers) (initial : Store) (id : Program) :=
  valueEvents p buffers (state access a.zero buffers initial id)
    (output a native p (state access a.zero buffers initial id))

theorem trace_stable (access : Access α Store Byte) (a : Arithmetic α) (native : Bool)
    (p : Parameters) (buffers : Physical.Buffers) (initial current : Store)
    (stable : Stable access a.zero buffers initial current) (id : Program) (stride : 16≤p.om) :
    programTrace access a native p buffers current id=some (events access a native p buffers initial id) := by
  rw [programTrace_complete access a native p buffers current id stride,
    output_stable access a native p buffers initial current stable id]
  rfl

theorem events_bounded (access : Access α Store Byte) (a : Arithmetic α) (native : Bool)
    (p : Parameters) (batches : Nat) (buffers : Physical.Buffers) (initial : Store)
    (q k v out : Host.Tensor) (rows cols : Nat)
    (abi : ABI p batches access buffers initial q k v out rows cols)
    (id : Program) (valid : id.Valid p batches) :
    Bounded buffers.output (events access a native p buffers initial id) := by
  intro e member
  change e∈(lanes p (state access a.zero buffers initial id)).map _ at member
  obtain ⟨i,hi,rfl⟩ := List.mem_map.mp member
  have live : id.row*16+i.1.val<p.length := of_decide_eq_true (List.mem_filter.mp hi).2
  have active := coordinate_active p batches id valid (show 0<p.heads from abi.header.shape.2.1) i live
  have bound := output_window p batches access buffers initial q k v out rows cols abi
    (coordinate p batches id i) active
  simpa only [output_offset access a.zero buffers initial p batches id i abi.parameters] using bound

/-- Every program executes the complete quote (or the selected local native
lowering), then materializes only its actual enabled store cells. -/
def grid (access : Access α Store Byte) (a : Arithmetic α) (native : Bool)
    (p : Parameters) (buffers : Physical.Buffers) : List Program → Store → Option Store
  | [],m => some m
  | id::ids,m => (program access a native p buffers m id).bind (grid access a native p buffers ids)

def gridEvents (access : Access α Store Byte) (a : Arithmetic α) (native : Bool)
    (p : Parameters) (buffers : Physical.Buffers) (initial : Store) (ids : List Program) :=
  ids.flatMap (events access a native p buffers initial)

theorem grid_static (access : Access α Store Byte) (a : Arithmetic α) (native : Bool)
    (p : Parameters) (batches : Nat) (buffers : Physical.Buffers) (initial : Store)
    (q k v out : Host.Tensor) (rows cols : Nat)
    (abi : ABI p batches access buffers initial q k v out rows cols)
    (ids : List Program) (valid : ∀ id∈ids,id.Valid p batches)
    (current : Store) (stable : Stable access a.zero buffers initial current) :
    grid access a native p buffers ids current=some
      (LayernormFwdTriton.Memory.run access.toAPI (gridEvents access a native p buffers initial ids) current) := by
  induction ids generalizing current with
  | nil => rfl
  | cons id ids ih =>
    have hv := valid id (List.mem_cons_self ..)
    have bound := events_bounded access a native p batches buffers initial q k v out rows cols abi id hv
    have hs := run_stable access a.zero buffers _ bound abi.separate initial current stable
    simp only [grid,program,trace_stable access a native p buffers initial current stable id
      (abi_stride p batches access buffers initial q k v out rows cols abi),Option.map_some,Option.bind_some]
    rw [ih (fun j hj => valid j (List.mem_cons_of_mem _ hj)) _ hs]
    simp only [gridEvents,List.flatMap_cons,LayernormFwdTriton.Memory.run,List.foldl_append]

theorem grid_success (access : Access α Store Byte) (a : Arithmetic α) (native : Bool)
    (p : Parameters) (batches : Nat) (buffers : Physical.Buffers) (initial : Store)
    (q k v out : Host.Tensor) (rows cols : Nat)
    (abi : ABI p batches access buffers initial q k v out rows cols) :
    grid access a native p buffers (programs p batches) initial=some
      (LayernormFwdTriton.Memory.run access.toAPI
        (gridEvents access a native p buffers initial (programs p batches)) initial) :=
  grid_static access a native p batches buffers initial q k v out rows cols abi _
    (fun id member => (programs_mem p batches id).mp member) initial (Stable.refl ..)

end PhysicalKernel
#axiomsClean PhysicalKernel.output_stable
#axiomsClean PhysicalKernel.trace_stable
#axiomsClean PhysicalKernel.events_bounded
#axiomsClean PhysicalKernel.grid_static
#axiomsClean PhysicalKernel.grid_success
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
