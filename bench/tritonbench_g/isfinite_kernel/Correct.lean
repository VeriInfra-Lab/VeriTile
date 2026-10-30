import bench.tritonbench_g.isfinite_kernel.Kernels
import bench.tritonbench_g.isfinite_kernel.Proofs.FP
import bench.tritonbench_g.isfinite_kernel.Proofs.TypedFlatWitness
import VeriTile.Meta.Specification

/-! Public mathematical bit-interpreter contracts for the complete sources.
`Spec.Real` below wraps a mathematical proposition about typed IEEE bit
patterns and `FP.Structural.exec`. It is not a Real erasure or an all-reals
are-finite interpretation: this source's classifier rejects Real projection.
These contracts establish actual successful execution, independent exponent
classification and a full tagged-cell frame. Companion theorems below prove
complete original/candidate flat execution under explicit allocation bounds
and unspecified-value compatibility, with concrete nonempty witnesses.
Native libdevice refinement remains a separate unfinished obligation. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.Correctness
open VeriTile VeriTile.Triton FP.Structural FP.ProfiledRewrite
open Parametric TypedBits BitClassification
noncomputable section

export TypedFlat (pair32_flat_correct pair64_flat_correct nonempty32_flat_pair nonempty64_flat_pair)

/-- Named typed correctness target: a successful complete execution and the
independent pointwise bit formula with every unwritten tagged cell preserved.
This target uses the caller's actual initial state and input realization. -/
def correct_target {α : Type} [Inhabited α] (model : Algebra α → Algebra α)
    (classify : α → Bool) (kernel : ComputeKernel) (p : Parameters) : Prop :=
  ∀ (rest : Algebra α) (bits : Nat → α) (initial : State α),
    p.input ≠ p.output → 0 < p.outputStride → InputBits p bits initial →
    ∃ final, FP.Structural.exec (model rest) kernel initial = some final ∧
      Post classify p bits initial final

def Contract32 (kernel : ComputeKernel) (p : Parameters) : Prop :=
  correct_target (fun rest => engine (backend32 rest)) finite32 kernel p

def Contract64 (kernel : ComputeKernel) (p : Parameters) : Prop :=
  correct_target (fun rest => engine (backend64 rest)) finite64 kernel p

specification original_fp32_correctness (p : Parameters) :
    Spec.Real (Contract32 (Kernels.original32 p) p) := by
  intro rest bits initial hne hstride hinput
  exact original32_correct rest p bits initial hne hstride hinput

specification original_fp64_correctness (p : Parameters) :
    Spec.Real (Contract64 (Kernels.original64 p) p) := by
  intro rest bits initial hne hstride hinput
  exact original64_correct rest p bits initial hne hstride hinput

specification index_commuted_fp32_correctness (p : Parameters) :
    Spec.Real (Contract32 (Kernels.indexCommuted32 p) p) := by
  intro rest bits initial hne hstride hinput
  exact candidate32_correct rest p bits initial hne hstride hinput

specification index_commuted_fp64_correctness (p : Parameters) :
    Spec.Real (Contract64 (Kernels.indexCommuted64 p) p) := by
  intro rest bits initial hne hstride hinput
  exact candidate64_correct rest p bits initial hne hstride hinput

#axiomsClean pair32_flat_correct
#axiomsClean pair64_flat_correct
#axiomsClean nonempty32_flat_pair
#axiomsClean nonempty64_flat_pair
#axiomsClean original_fp32_correctness
#axiomsClean original_fp64_correctness
#axiomsClean index_commuted_fp32_correctness
#axiomsClean index_commuted_fp64_correctness
end
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.Correctness
