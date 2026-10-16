import bench.tritonbench_g.reversed_cumsum_scalar.Kernels
import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.RealSurface
import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.RealWitness
import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.RealRegression
import VeriTile.Meta.Specification

/-! Complete real source contracts: successful execution from arbitrary
registers, independent padded suffix values, standard tagged-cell realization
and frame, and successful finite-allocation execution. These are linear row
contracts; arbitrary Python tensor-stride layouts are outside this interface. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.Correctness
open VeriTile VeriTile.Triton
noncomputable section

def original (dtype out_dtype : FloatDType) (s o : RegionName) (T BT : Nat) : ComputeKernel :=
  (Kernels.original dtype out_dtype s o T BT).eraseDType

def indexCommuted (dtype out_dtype : FloatDType) (s o : RegionName) (T BT : Nat) : ComputeKernel :=
  (Kernels.indexCommuted dtype out_dtype s o T BT).eraseDType

def Contract (kernel : ComputeKernel) (T BT : Nat) (s o : RegionName) : Prop :=
  (∀ initial : BlockState, s ≠ o → 0 < BT →
    ∃ final, exec kernel.toAlgKernel initial = some final ∧
      RealExecution.Post T BT s o initial final) ∧
  (∀ initial : BlockState, s ≠ o → 0 < BT →
    ComputeCorrect.RealizesFrame_without_Rounding kernel initial
      (RealSurface.writes T (initial.pids 0) o) (RealSurface.expectedCell T BT s initial)) ∧
  (∀ (A : FlatAlloc), A.Disjoint → (∀ r, r ∉ A.regions → A.extent r = 0) →
    ∀ initial : BlockState, s ≠ o → 0 < BT →
      RealSafety.Bounds T (initial.pids 0) s o A.extent →
      initial.undef = (fun _ _ => 0) →
      ∃ final, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState initial) = some final ∧
        RealIO.Post A T s o initial final)

theorem source_contract (swapped : Bool) (dtype out_dtype : FloatDType)
    (s o : RegionName) (T BT : Nat) :
    Contract (TypedSource.source swapped dtype out_dtype s o T BT).eraseDType T BT s o := by
  refine ⟨?_,?_,?_⟩
  · intro initial hSO hBT
    exact RealExecution.source_correct swapped T BT s o dtype out_dtype initial hSO hBT
  · intro initial hSO hBT
    exact RealSurface.source_realizes swapped T BT s o dtype out_dtype initial hSO hBT
  · intro A hd hc initial hSO hBT hb hu
    exact RealIO.source_correct swapped A hd hc T BT s o dtype out_dtype initial hSO hBT hb hu

specification reversed_cumsum_scalar_correctness (dtype out_dtype : FloatDType)
    (s o : RegionName) (T BT : Nat) :
    Spec.Real (Contract (original dtype out_dtype s o T BT) T BT s o) :=
  source_contract Bool.false dtype out_dtype s o T BT

specification index_commuted_correctness (dtype out_dtype : FloatDType)
    (s o : RegionName) (T BT : Nat) :
    Spec.Real (Contract (indexCommuted dtype out_dtype s o T BT) T BT s o) :=
  source_contract Bool.true dtype out_dtype s o T BT

theorem nonempty_instances (dtype out_dtype : FloatDType) :
    (∃ v, exec (RealWitness.allocation.flattenKernel
      (original dtype out_dtype "s" "o" 17 16).toAlgKernel)
      (RealWitness.allocation.flattenState RealWitness.initial) = some v) ∧
    (∃ v, exec (RealWitness.allocation.flattenKernel
      (indexCommuted dtype out_dtype "s" "o" 17 16).toAlgKernel)
      (RealWitness.allocation.flattenState RealWitness.initial) = some v) := by
  obtain ⟨u,hu,_⟩ := RealWitness.real_nonempty Bool.false dtype out_dtype
  obtain ⟨v,hv,_⟩ := RealWitness.real_nonempty Bool.true dtype out_dtype
  exact ⟨⟨u,hu⟩,⟨v,hv⟩⟩

#axiomsClean source_contract
#axiomsClean reversed_cumsum_scalar_correctness
#axiomsClean index_commuted_correctness
#axiomsClean nonempty_instances
end
end VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.Correctness
