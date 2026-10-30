import bench.repairs.tritonbench_g.chunk_retention.Whole001.Proofs.Calls
namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001
noncomputable section
open Numeric

/-- Whole software FP32 graph, with independently timed forward/backward calls. -/
specification reference_fp (u : UnaryModels) (c : Shape) (_ok : HostABI.Accepted c)
    (_runtime : HostABI.RuntimeContract c) (tree : SumTree) (complete : tree.Complete c.keyBlocks)
    (args : CallMeta c) (alloc : PhaseAPI.ForwardBuffers caller) (fb : Binding c)
    (bound : PhaseAPI.ForwardBinding args alloc fb) (before : Memory Word) :
    Calls.Post (wordOps (referenceISA u)) (wordOps (referenceISA u)) c
      (observe (u.hostScale (BitVec.ofNat 32 c.K))) (observe (u.hostScale (BitVec.ofNat 32 c.K)))
      tree args alloc fb before :=
  Calls.post _ c _ tree complete args alloc fb bound before

/-- Exactly eight fixed-width scalar leaves. Both actual physical calls equal
the whole software graphs, with returned handles, saved context and own frames. -/
specification fp_equiv (u : UnaryModels) (ports : ScalarPorts) (laws : ScalarLaws u ports)
    (c : Shape) (_ok : HostABI.Accepted c) (_runtime : HostABI.RuntimeContract c)
    (tree : SumTree) (complete : tree.Complete c.keyBlocks)
    (args : CallMeta c) (alloc : PhaseAPI.ForwardBuffers caller) (fb : Binding c)
    (bound : PhaseAPI.ForwardBinding args alloc fb) (before : Memory Word) :
    Calls.Post (wordOps (nativeISA ports)) (wordOps (referenceISA u)) c
      (observe (ports.hostScale (BitVec.ofNat 32 c.K))) (observe (u.hostScale (BitVec.ofNat 32 c.K)))
      tree args alloc fb before := by
  rw [isa_eq u ports laws,laws.hostScale]
  exact Calls.post _ c _ tree complete args alloc fb bound before

#axiomsClean reference_fp
#axiomsClean fp_equiv
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001
