import bench.tritonbench_g.attention_kernel_aligned.Proofs.LogicalReader
import bench.tritonbench_g.attention_kernel_aligned.Proofs.IOSafety
import bench.tritonbench_g.attention_kernel_aligned.Proofs.BF16IOSafety

/-! Finite original strided inputs, complete source-grid execution, all logical
attention outputs and the exact exterior frame. Actual masked load/store safety
is part of the shared IO judgment. Both original storage branches are retained. -/
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.Correct
open VeriTile Triton FP.Structural Full Whole Interface
open scoped BigOperators VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

def attentionScore (D BN biasLast : Nat) (scale : ℝ) (q k bias : Nat → Nat → ℝ) (m j : Nat) : ℝ :=
  (∑ d∈Finset.range D,(q m d*(scale*1.44269504))*k j d) +
    (bias m (j/BN)+bias m (j%BN%biasLast+biasLast))
def attention (N D BN biasLast : Nat) (scale : ℝ)
    (q k v bias : Nat → Nat → ℝ) (m d : Nat) : ℝ :=
  (∑ j∈Finset.range N,pow2 (attentionScore D BN biasLast scale q k bias m j)*v j d) /
    (∑ j∈Finset.range N,pow2 (attentionScore D BN biasLast scale q k bias m j))
specification kernel_correct (p : Parameters) (g : Guard p)
    (separate : Whole.Nonempty p → Separated p)
    (work : List Program) (order : (programs (grid p)).Perm work) :
    Spec.Real (LogicalIO.io p work ⊨ fun xs b h row col =>
      attention p.length p.head p.blockN p.biasLast p.scale
        (xs.query b h) (xs.key b h) (xs.value b h) (xs.bias b h) row.val col.val) :=
  LogicalReader.correct p g separate work order _ rfl

#axiomsClean kernel_correct
#specNonCircular attention avoiding [Source.source,FP.Structural.exec,Whole.runPrograms,Whole.answer]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.Correct

namespace VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Correct
open VeriTile.Bench.Repairs.AttentionKernelAligned.Interface (layout readInputs)
open VeriTile Triton FP.Structural Full Whole Interface
open scoped BigOperators VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

specification kernel_correct (p : Parameters) (g : Guard p)
    (separate : Whole.Nonempty p → Separated p)
    (work : List Program) (order : (programs (grid p)).Perm work) :
    Spec.Real (LogicalIO.io p work ⊨ fun xs b h row col =>
      VeriTile.Bench.Repairs.AttentionKernelAligned.Correct.attention p.length p.head p.blockN p.biasLast p.scale
        (xs.query b h) (xs.key b h) (xs.value b h) (xs.bias b h) row.val col.val) :=
  LogicalReader.correct p g separate work order _ rfl

#axiomsClean kernel_correct
#specNonCircular VeriTile.Bench.Repairs.AttentionKernelAligned.Correct.attention avoiding [Source.source,FP.Structural.exec,Whole.runPrograms,Whole.answer]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Correct

