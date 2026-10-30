import bench.tritonbench_g.fused_rwkv6_kernel.Proofs.LogicalCoordinates

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Correct
open VeriTile Triton FP.Structural FusedRwkv6Kernel
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

def kernelIO (c : Shape) (scale : ℝ) := Interface.logicalIO c scale
open scoped BigOperators

/-- Independent per-key-shard RWKV output. The mathematical state obeys
H[0]=H0 (or zero) and H[n+1]=H[n]*exp(w[visit n])+k[visit n]*v[visit n].
`visit` retains either forward or reverse traversal; no host shard sum is claimed. -/
def partialValue (c : Shape) (x : Input ℝ) (scale : ℝ) (z ik n f : Nat) : ℝ :=
  ∑ j : Fin c.BK, if ik*c.BK+j.val<c.K then
    (realState c x z (ik*c.BK+j.val) f n+
      x.k z (visit c n) (ik*c.BK+j.val)*x.v z (visit c n) f*x.u z (ik*c.BK+j.val))*
      (x.q z (visit c n) (ik*c.BK+j.val)*scale) else 0

/-- The two output channels are key-shard partials and the optional final state. -/
def formula (c : Shape) (scale : ℝ) (xs : Interface.TensorInputs ℝ c) :
    Interface.TensorOutputs ℝ c where
  partials := fun ik z t f => partialValue c (Interface.tensorData c xs) scale
    z.val ik.val (visit c t.val) f.val
  finalState := fun z e f => realState c (Interface.tensorData c xs) z.val e.val f.val c.T

/-- Full source grid writes every independent key-shard partial and optional
terminal recurrence. Actual access safety, typed outputs and the raw exterior
frame are included in the shared IO judgment. No host sum is introduced. -/
specification rwkv_real (c : Shape) (hc : Sizes c) (scale : ℝ) :
    Spec.Real (kernelIO c scale ⊨ formula c scale) :=
  Interface.Result.logical_real c hc scale

#stmtSurfaceSubset rwkv_real ⊆ [Shape,Sizes,Spec.Real,LogicalProgramIO.Implements,kernelIO,formula,Interface.TensorInputs,Interface.TensorOutputs]
#specNonCircular formula avoiding [kernel,Source.launch,FP.Structural.exec,FusedRwkv6Kernel.runProgram]
#axiomsClean rwkv_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Correct
