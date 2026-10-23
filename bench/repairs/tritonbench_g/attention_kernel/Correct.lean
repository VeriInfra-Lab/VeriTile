import bench.repairs.tritonbench_g.attention_kernel.Proofs.LogicalReader
import bench.repairs.tritonbench_g.attention_kernel.Proofs.IOResults

namespace VeriTile.Bench.Repairs.AttentionKernel.Correct
open VeriTile Triton FP.Structural Whole001 Source Whole Machine
open scoped BigOperators VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

def attentionScore (D BN : Nat) (scale : ℝ) (q k bias : Nat→Nat→ℝ) (m n : Nat) : ℝ :=
  (scale*1.44269504)*((Finset.range D).sum (fun d=>q m d*k n d))+
    (bias m (n/BN)+bias m (64+(n%BN)%64))*1.44269504

/-- Independent base-2 softmax weighted sum over the initial logical inputs.
The literal coefficient and both 128-wide bias windows are source-exact. -/
def attention (N D BN : Nat) (scale : ℝ) (q k v bias : Nat→Nat→ℝ) (m d : Nat) : ℝ :=
  ((Finset.range N).sum (fun n=>pow2 (attentionScore D BN scale q k bias m n)*v n d)) /
    ((Finset.range N).sum (fun n=>pow2 (attentionScore D BN scale q k bias m n)))

/-- Complete original grid implements the independent finite attention formula.
The shared judgment proves actual source access safety, successful execution,
every logical output and the complete exterior frame. Original machine/address
closure remains an explicit conclusion. -/
specification kernel_correct (p : Parameters) (bases : Bases) (domain : Domain p bases)
    (hbm : 0<p.bm) (separate : Separated p) (work : List Program)
    (order : (programs (grid p)).Perm work) :
    Spec.Real ((LogicalIO.io p work ⊨ fun xs b h row col =>
      attention p.length p.dim p.bn p.scale (xs.query b h) (xs.key b h)
        (xs.value b h) (xs.bias b h) row.val col.val) ∧ Closed p bases) :=
  ⟨LogicalReader.correct p bases domain hbm separate work order _ (by intros; rfl),Machine.closed p bases domain⟩

#specNonCircular attention avoiding [Source.kernel,FP.Structural.exec,Whole.runPrograms]
#axiomsClean kernel_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionKernel.Correct
