import bench.tritonbench_g.mixed_sparse_attention.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.MixedSparseAttention
open VeriTile Triton FP.Structural
open Whole002 Source Whole
open scoped VeriTile.Triton.LogicalDomainIO
noncomputable section

/-- Count-selected key occurrences. Block entries retain the original causal
and sequence filters; explicit columns keep their original arbitrary gather
coordinates. Concatenation preserves duplicates and their weights. -/
def occurrences (c : Config) (xs : KernelIO.LogicalInputs) (hz query : Nat) : List Nat :=
  ((List.range (xs.blockCount c hz query)).flatMap fun block =>
    ((List.range c.BN).map fun j => xs.blockOffset c hz query block+j).filter fun key =>
      decide (key<xs.sequenceLength (hz/c.H) ∧ key≤query)) ++
    (List.range (xs.columnCount c hz query)).map (fun j => xs.column c hz query j)

/-- Independent sparse attention in logical tensor coordinates. The exact
source coefficient 1.44269504 and base-two exponent remain explicit. Empty
support projects to zero through Real division, as in the canonical target. -/
def sparseAttention (c : Config) (xs : KernelIO.LogicalInputs)
    (hz : Fin (c.Z*c.H)) (query : Fin c.T) (channel : Fin c.BD) : ℝ :=
  let keys := occurrences c xs hz.val query.val
  let weight := fun key => pow2 ((c.scale * 1.44269504) *
    ∑ k ∈ Finset.range c.BD, xs.query c hz.val query.val k * xs.key c hz.val key k)
  (keys.map (fun key => weight key * xs.value c hz.val key channel.val)).sum /
    (keys.map weight).sum

/-- Complete original grid and eight full named arrays. The IO retains every
source footprint, active query mask, typed conversion and exterior frame. -/
specification kernel_real (c : Config) (sep : Separated c) (layout : Contiguous c)
    (hbm : 0<c.BM) (hbn : 0<c.BN) (work : List Program)
    (order : work.Perm (programs (grid c))) :
    Spec.Real (RealIO.logicalIO c work ⊨{RealIO.logicalDomain c} sparseAttention c) := by
  apply LogicalIOProof.implements c sep layout hbm hbn work order
  intro xs hz row col
  rfl

#specNonCircular sparseAttention avoiding [FP.Structural.exec,runSchedule,Source.finalValues,State.mem]
#stmtSurfaceSubset kernel_real ⊆
  [Spec.Real,LogicalDomainIO.ImplementsOn,RealIO.logicalIO,RealIO.logicalDomain,sparseAttention,KernelIO.LogicalInputs,Config,Config.BM,Config.BN,Separated,Contiguous,Program,programs,grid,List.Perm,KernelIO.Values,ChannelValues,KernelIO.outputs,Config.Z,Config.H,Config.T,Config.BD,ArrayProgramIO.ImplementsOn,Real.R,RealIO.io]
#axiomsClean kernel_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MixedSparseAttention
