import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.KernelMemory

/-! Kernel launch contract for the unchanged selected guard. Outputs are
ordinary fresh allocation descriptors. No successful host/kernel call is a
premise. Empty dimensions remain admitted and dispatch no kernel. -/
namespace VeriTile.Bench.Repairs.LayerNormTriton.KernelContract
open VeriTile.Bench.Repairs.FastLayernorm.Storage
open KernelMemory
noncomputable section

structure Forward (M N : Nat) where
  (x w bias : Input)
  normalized : List Nat
  epsilon : Float
  admitted : Contract.InputContract x.metadata w.metadata bias.metadata (some normalized) (some epsilon)
  columns : N=normalized.prod
  xCount : x.dense.count=M*N
  wCount : w.dense.count=N
  bCount : bias.dense.count=N
  buffers : Execution.Buffers 3 [x.dense.tensor,w.dense.tensor,bias.dense.tensor]
  owner : buffers.owner=x.dense.owner
  inverseCount : (buffers.output 0).count=M
  meanCount : (buffers.output 1).count=M
  outputCount : (buffers.output 2).count=M*N
  outputShape : (buffers.output 2).tensor.view.shape=x.metadata.shape

structure Backward (M N : Nat) (x w dy r mu : Input) where
  gradient : Contract.GradientContract dy.metadata x.metadata x.metadata.shape
  xCount : x.dense.count=M*N
  dyCount : dy.dense.count=M*N
  wCount : w.dense.count=N
  inverseCount : r.dense.count=M
  meanCount : mu.dense.count=M
  first : Execution.BackwardBuffers M N (Geometry.groups N)
    [x.dense.tensor,w.dense.tensor,dy.dense.tensor,r.dense.tensor,mu.dense.tensor]
  last : Execution.FinalBuffers N
    [x.dense.tensor,w.dense.tensor,dy.dense.tensor,r.dense.tensor,mu.dense.tensor]
  sameOwner : (first.buffers.output 0).owner=last.buffers.owner
  preserveDX : ∀ k,Disjoint (last.buffers.output k).tensor.backing (first.buffers.output 0).tensor.backing
  preserveScratch : ∀ k j,Disjoint (last.buffers.output k).tensor.backing (first.buffers.output j).tensor.backing

/-- Exact source launch condition; N=1 takes the ordinary nonempty branch. -/
def launched (M N : Nat) : Bool := M != 0 && N != 0

@[simp] theorem launched_iff (M N : Nat) : launched M N=Bool.true ↔ 0<M ∧ 0<N := by
  simp [launched]; omega

#axiomsClean launched_iff
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.KernelContract
