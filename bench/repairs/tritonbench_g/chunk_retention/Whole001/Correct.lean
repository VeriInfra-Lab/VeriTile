import bench.repairs.tritonbench_g.chunk_retention.Whole001.Proofs.Calls
namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001
noncomputable section

/-- Whole independent dense real retention, including actual returned handles,
separate later backward memory, detached loss differential and raw caller frames.
The forward premise has no future cotangent or backward-allocation binding. -/
specification correct (c : Shape) (_ok : HostABI.Accepted c) (_runtime : HostABI.RuntimeContract c)
    (scale : ℝ) (tree : SumTree) (complete : tree.Complete c.keyBlocks)
    (args : CallMeta c) (alloc : PhaseAPI.ForwardBuffers caller) (fb : Binding c)
    (bound : PhaseAPI.ForwardBinding args alloc fb) (before : Memory ℝ) :
    Calls.RealPost c scale tree args alloc fb before :=
  Calls.real_post c scale tree complete args alloc fb bound before

#axiomsClean correct
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001
