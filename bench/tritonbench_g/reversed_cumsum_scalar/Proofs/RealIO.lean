import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.RealSafety
import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.RealFormula
import VeriTile.Triton.Memory.FlattenFrame

/-! Successful finite-allocation execution of the complete scalar source.
The generic flattening theorem requires an explicitly zero unspecified-value
oracle; the region theorem retains arbitrary unspecified padding. The frame
preserves complete tagged cells, including addresses outside the allocation. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealIO
open VeriTile.Triton RealProgram RealValues RealLoop RealMemory
noncomputable section

def Outside (A : FlatAlloc) (T row : Nat) (o region : RegionName) (offset : Nat) : Prop :=
  region ≠ A.flat ∨ ∀ q z, footprint T row o (q,z) → offset ≠ A.addr q z

def Post (A : FlatAlloc) (T : Nat) (s o : RegionName) (initial final : BlockState) : Prop :=
  (∀ t, t < T →
    final.mem A.flat (A.addr o (address T (initial.pids 0) t)) =
      MemCell.real (RealFormula.suffix T (initial.pids 0) (initial.readMem s) t)) ∧
  (∀ region offset, Outside A T (initial.pids 0) o region offset →
    final.mem region offset = (A.flattenState initial).mem region offset)

theorem flatten_post (A : FlatAlloc) (hd : A.Disjoint)
    (hc : ∀ r, r ∉ A.regions → A.extent r = 0)
    (T BT : Nat) (s o : RegionName) (initial final : BlockState) (hBT : 0 < BT)
    (hb : RealSafety.Bounds T (initial.pids 0) s o A.extent)
    (hu : initial.undef = fun _ _ => 0)
    (hp : RealExecution.Post T BT s o initial final) :
    Post A T s o initial (A.flattenState final) := by
  constructor
  · intro t ht
    have ho := hb.output t ht
    have hr : o ∈ A.regions := by
      by_contra hn
      rw [hc o hn] at ho
      omega
    rw [A.flattenState_mem_addr hd final hr ho, hp.1 t ht]
    change MemCell.real (expected T BT (initial.pids 0)
      (initial.readMem s) (initial.undef s) t) = _
    rw [RealFormula.zero_oracle T BT (initial.pids 0) (initial.readMem s)
      (initial.undef s) t hBT ht (by intro offset; rw [hu])]
  · intro region offset hn
    exact A.flattenState_frame initial final (fun q z => footprint T (initial.pids 0) o (q,z))
      (fun q z h => (hp.2 q z h).symm) region offset hn

theorem source_correct (swapped : Bool) (A : FlatAlloc) (hd : A.Disjoint)
    (hc : ∀ r, r ∉ A.regions → A.extent r = 0)
    (T BT : Nat) (s o : RegionName) (dtype out_dtype : FloatDType)
    (initial : BlockState) (hSO : s ≠ o) (hBT : 0 < BT)
    (hb : RealSafety.Bounds T (initial.pids 0) s o A.extent)
    (hu : initial.undef = fun _ _ => 0) :
    ∃ final, exec (A.flattenKernel
      (TypedSource.source swapped dtype out_dtype s o T BT).eraseDType.toAlgKernel)
        (A.flattenState initial) = some final ∧ Post A T s o initial final := by
  obtain ⟨final, he, hp⟩ := RealExecution.source_correct swapped T BT s o dtype out_dtype initial hSO hBT
  refine ⟨A.flattenState final, ?_, flatten_post A hd hc T BT s o initial final hBT hb hu hp⟩
  rw [A.exec_flatten hd hc _ initial
    (RealSafety.source_safe swapped T BT s o dtype out_dtype initial hSO hBT A.extent hb)
    (RealSafety.source_flatten_ok swapped T BT s o dtype out_dtype) hu, he]
  rfl

#axiomsClean flatten_post
#axiomsClean source_correct
end
end VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealIO
