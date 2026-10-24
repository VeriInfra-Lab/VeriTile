import bench.repairs.tritonbench_g.fast_layernorm.Proofs.IONative

namespace VeriTile.Bench.Repairs.FastLayernorm.Equivalence
open VeriTile Triton FP.Structural KernelNative
noncomputable section

/-- The same source IO and masks as Correct. Arbitrary symbolic inputs remain
admitted. Native observations retain output-specific compiler schedules. -/
abbrev forwardIO (R : Reductions) := Interface.forwardIO (algebra R)
abbrev backwardIO (R : Reductions) := Interface.backwardIO (algebra R)
abbrev affineIO (R : Reductions) := Interface.affineIO (algebra R)
abbrev forwardFormula := Interface.nativeForward
abbrev backwardFormula := Interface.nativeBackward
abbrev affineFormula := Interface.nativeAffine

/-- Native observation of the canonical forward source, including saved r/mu.
There is no N≤B restriction here. Primitive FP32 rounding and compiler schedules
are model parameters; this is not an algebraic floating rewrite. -/
theorem forward_kernel_native (R : Reductions) (I : Instructions) (S : OutputSchedule)
    (N B pid : Nat) (eps : ℝ) (bias : Bool) :
    (forwardIO R N B pid eps bias).Observes (Interface.forwardObserver I S N B pid)
      (forwardFormula R I S N B pid eps bias) :=
  Interface.forward_native R I S N B pid eps bias

theorem backward_kernel_native (R : Reductions) (I : Instructions) (S : OutputSchedule)
    (N B pid : Nat) :
    (backwardIO R N B pid).Observes (Interface.backwardObserver I S N B pid)
      (backwardFormula R I S N B pid) :=
  Interface.backward_native R I S N B pid

theorem affine_kernel_native (R : Reductions) (I : Instructions) (S : OutputSchedule)
    (rows N wc bc pid : Nat) (bias : Bool) :
    (affineIO R rows N wc bc pid bias).Observes (Interface.affineObserver I S pid)
      (affineFormula R I S rows N wc bc pid bias) :=
  Interface.affine_native R I S rows N wc bc pid bias

-- Exact whole-memory results and epsilon ABI/compiler-binding specializations
-- remain ordinary theorems in Proofs/RawNative.lean, with their original names.
#stmtSurfaceSubset forward_kernel_native ⊆
  [Reductions,Instructions,OutputSchedule,Term,Word,ProgramIO.Observes,forwardIO,
   Interface.forwardObserver,forwardFormula,KernelNative.instInhabitedTerm]
#stmtSurfaceSubset backward_kernel_native ⊆
  [Reductions,Instructions,OutputSchedule,Term,Word,ProgramIO.Observes,backwardIO,
   Interface.backwardObserver,backwardFormula,KernelNative.instInhabitedTerm]
#stmtSurfaceSubset affine_kernel_native ⊆
  [Reductions,Instructions,OutputSchedule,Term,Word,ProgramIO.Observes,affineIO,
   Interface.affineObserver,affineFormula,KernelNative.instInhabitedTerm]
#specNonCircular forwardFormula avoiding [forwardKernel,backwardKernel,affineKernel,FP.Structural.exec]
#specNonCircular backwardFormula avoiding [forwardKernel,backwardKernel,affineKernel,FP.Structural.exec]
#specNonCircular affineFormula avoiding [forwardKernel,backwardKernel,affineKernel,FP.Structural.exec]
#axiomsClean forward_kernel_native
#axiomsClean backward_kernel_native
#axiomsClean affine_kernel_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.Equivalence
