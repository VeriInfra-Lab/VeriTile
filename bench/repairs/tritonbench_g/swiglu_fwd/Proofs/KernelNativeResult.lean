import bench.repairs.tritonbench_g.swiglu_fwd.Proofs.Kernel
import bench.repairs.tritonbench_g.swiglu_triton.Proofs.KernelNative

namespace VeriTile.Bench.Repairs.SwigluFwd.KernelNativeResult
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluFwd.Whole
open VeriTile.Bench.TritonBenchG.SwigluBackward.Observed (Term graphAlgebra)
open VeriTile.Bench.Repairs.SwigluTriton
open VeriTile.Bench.Repairs.SwigluTriton.Native
noncomputable section

/-- Actual ordered forward native graph, with the encoded sigmoid constants
and instruction ports. No invented arithmetic variant or scalar hypothesis. -/
def fp32Formula (instructions : InstructionSemantics) (x y : Word) : Word :=
  let s := sigmoidGraph instructions x
  instructions.mulRN (instructions.mulRN s x) y

theorem scalar_native (instructions : InstructionSemantics) (x y : Word) :
    interpret (compiledPorts instructions) (scalar graphAlgebra (.input x) (.input y)) =
      fp32Formula instructions x y := by rfl

theorem swiglu_kernel_native (instructions : InstructionSemantics)
    (c : Layout) (B row tile : Nat) (read : RegionName → Nat → Word)
    (scratch : State (Term Word)) :
    ∃ final, exec graphAlgebra (kernel c.X c.Y c.OUT c.sx c.sy c.so c.cols B)
        (tileState graphAlgebra c B row tile (KernelNative.rawState [c.X, c.Y] read scratch)) = some final ∧
      (∀ j : Fin B, tile*B+j.val < c.cols →
        interpret (compiledPorts instructions)
          ((final.mem c.OUT (row*c.so+(tile*B+j.val))).read .real) =
          fp32Formula instructions (read c.X (row*c.sx+(tile*B+j.val)))
            (read c.Y (row*c.sy+(tile*B+j.val)))) ∧
      (∀ r o, Outside c B row tile r o →
        final.mem r o = (KernelNative.rawState [c.X, c.Y] read scratch).mem r o) := by
  obtain ⟨final, he, values, frame⟩ := KernelProof.run graphAlgebra c B row tile (KernelNative.rawState [c.X, c.Y] read scratch)
  refine ⟨final, he, ?_, frame⟩
  intro j hj
  rw [values j hj]
  change interpret (compiledPorts instructions)
    (scalar graphAlgebra
      (((KernelNative.rawState [c.X, c.Y] read scratch).mem c.X (row*c.sx+(tile*B+j.val))).read .real)
      (((KernelNative.rawState [c.X, c.Y] read scratch).mem c.Y (row*c.sy+(tile*B+j.val))).read .real)) = _
  rw [KernelNative.rawState_read [c.X, c.Y] read scratch c.X _ (by simp),
    KernelNative.rawState_read [c.X, c.Y] read scratch c.Y _ (by simp)]
  exact scalar_native instructions _ _

#specNonCircular fp32Formula avoiding [kernel, interpret, scalar]
#axiomsClean swiglu_kernel_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluFwd.KernelNativeResult
