import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Contract
import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Readback

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory (API Apart Write)
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 1000000
namespace PhysicalKernel
variable {α : Type} [Inhabited α]

def lanes (p : Parameters) (s : State (Replicas α)) : List (TileIndex [16,16]) :=
  (TileShape.allIndices [16,16]).filter (fun i => decide (row s i.1<p.length))

/-- Materialize the actual source store cells at their physical FP32 addresses.
The final state comes from `execution`, and the sole mask is the source mask. -/
def storedEvents (p : Parameters) (buffers : Physical.Buffers)
    (before final : State (Replicas α)) : List (Write α) :=
  (lanes p before).map (fun i =>
    ⟨Physical.inputAddress buffers.output (outputOffset p before i),
      (final.mem "out" (outputOffset p before i)).read .real (owner i.2.1.val)⟩)

def valueEvents (p : Parameters) (buffers : Physical.Buffers)
    (before : State (Replicas α)) (values : Matrix α) : List (Write α) :=
  (lanes p before).map (fun i =>
    ⟨Physical.inputAddress buffers.output (outputOffset p before i),values i (owner i.2.1.val)⟩)

def programTrace (access : Access α Store Byte) (a : Arithmetic α) (native : Bool)
    (p : Parameters) (buffers : Physical.Buffers) (m : Store) (program : Program) :
    Option (List (Write α)) :=
  let before := state access a.zero buffers m program
  (execution a native p before).map (storedEvents p buffers before)

def program (access : Access α Store Byte) (a : Arithmetic α) (native : Bool)
    (p : Parameters) (buffers : Physical.Buffers) (m : Store) (id : Program) : Option Store :=
  (programTrace access a native p buffers m id).map (fun events =>
    LayernormFwdTriton.Memory.run access.toAPI events m)

theorem source_store_materialization (p : Parameters) (buffers : Physical.Buffers)
    (before final : State (Replicas α)) (values : Matrix α) (stride : 16≤p.om)
    (stored : final.mem=outputMemory p before values) :
    storedEvents p buffers before final=valueEvents p buffers before values := by
  apply List.map_congr_left
  intro i member
  have live : row before i.1<p.length := of_decide_eq_true (List.mem_filter.mp member).2
  simp only [stored,outputMemory_readback p before values stride i live,Cell.read_same]

theorem programTrace_complete (access : Access α Store Byte) (a : Arithmetic α) (native : Bool)
    (p : Parameters) (buffers : Physical.Buffers) (m : Store) (id : Program) (stride : 16≤p.om) :
    programTrace access a native p buffers m id=
      some (valueEvents p buffers (state access a.zero buffers m id)
        (output a native p (state access a.zero buffers m id))) := by
  obtain ⟨final,success,stored⟩ := complete a native p (state access a.zero buffers m id)
    (state_integers access a.zero buffers m id)
  simp only [programTrace,success,Option.map_some,
    source_store_materialization p buffers _ final _ stride stored]

end PhysicalKernel
#axiomsClean PhysicalKernel.source_store_materialization
#axiomsClean PhysicalKernel.programTrace_complete
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
