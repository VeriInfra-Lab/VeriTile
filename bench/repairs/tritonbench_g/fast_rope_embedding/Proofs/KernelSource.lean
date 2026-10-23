import bench.repairs.tritonbench_g.fast_rope_embedding.Contract
import bench.repairs.tritonbench_g.fast_rope_embedding.Proofs.KernelInterface

namespace VeriTile.Bench.Repairs.FastRopeEmbedding.KernelContract
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RopeEmbedding
noncomputable section
set_option maxHeartbeats 5000000

/-- Active Tensor indexing justifies the packed-address interpretation, while
execution retains the actual stride argument, including arbitrary singleton
strides. The entire original head loop succeeds. -/
theorem executes {α : Type} [Inhabited α] (op : Operations α) (c : Call) (admit : Admitted c)
    (s : State α) (row group : Nat) (launch : c.geometry.Launched row group) :
    ∃ final,FP.Structural.exec op.backend c.source (programState s row group)=some final ∧
      Kernel.Post c.geometry row group s final (fun h col=>scalarResult op c.geometry c.backward
        (fun o=>(s.mem "Q" o).read .real) (fun o=>(s.mem "C" o).read .real)
        (fun o=>(s.mem "S" o).read .real) row h col) := by
  have gh := group_head c.geometry group launch.2.2
  have dim : 0<c.geometry.dimension := by
    have half := launch.1
    change 0<c.geometry.dimension/2 at half
    omega
  have hp : 0<c.geometry.heads*c.geometry.dimension := Nat.mul_pos (by omega) dim
  exact KernelInterface.executes op c.geometry c.qStride c.cosineStride c.sineStride
    c.backward s row group launch (active_stride c admit row launch.2.1 hp)
    (admit.cosineStride launch.1) (admit.sineStride launch.1)

#axiomsClean executes
end
end VeriTile.Bench.Repairs.FastRopeEmbedding.KernelContract
