import bench.tritonbench_g.reversed_cumsum.Proofs.RealSafety
import bench.tritonbench_g.reversed_cumsum.Proofs.RealFormula
import VeriTile.Triton.Memory.FlattenFrame

/-! Finite allocation semantics of the complete source. The current generic
flattening bridge requires an explicitly zero unspecified-value oracle. The
region execution theorem retains arbitrary padding values; this specialization
therefore has the conventional suffix formula. The frame preserves complete
tagged cells, including relocated pointer values and unallocated addresses. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsum.RealIO
open VeriTile.Triton RealProgram RealStep RealLoop
noncomputable section

def Outside (A : FlatAlloc) (p : Layout) (l : Launch) (o region : RegionName)
    (offset : Nat) : Prop :=
  region ≠ A.flat ∨ ∀ q z, footprint p l o (q,z) → offset ≠ A.addr q z

def Post (A : FlatAlloc) (p : Layout) (s o : RegionName)
    (initial final : BlockState) : Prop :=
  (∀ (t : Nat) (j : Fin p.BS), active p (RealInit.launch initial) t j.val →
    final.mem A.flat (A.addr o (address p (RealInit.launch initial) t j.val)) =
      MemCell.real (RealFormula.suffix p (RealInit.launch initial) (initial.readMem s) t j)) ∧
  (∀ region offset, Outside A p (RealInit.launch initial) o region offset →
    final.mem region offset = (A.flattenState initial).mem region offset)

theorem flatten_post (A : FlatAlloc) (hd : A.Disjoint)
    (hc : ∀ r, r ∉ A.regions → A.extent r = 0)
    (p : Layout) (s o : RegionName) (initial final : BlockState)
    (hBT : 0 < p.BT)
    (hb : RealSafety.Bounds p (RealInit.launch initial) s o A.extent)
    (hu : initial.undef = fun _ _ => 0)
    (hp : RealExecution.Post p s o initial final) :
    Post A p s o initial (A.flattenState final) := by
  constructor
  · intro t j ha
    have ho := hb.output t j ha
    have hr : o ∈ A.regions := by
      by_contra hn
      rw [hc o hn] at ho
      omega
    rw [A.flattenState_mem_addr hd final hr ho, hp.1 t j ha]
    change MemCell.real (expected p (RealInit.launch initial)
      (initial.readMem s) (initial.undef s) t j) = _
    rw [RealFormula.zero_oracle p (RealInit.launch initial) (initial.readMem s)
      (initial.undef s) t j hBT ha (by intro offset; rw [hu])]
  · intro region offset hn
    exact A.flattenState_frame initial final
      (fun q z => footprint p (RealInit.launch initial) o (q,z))
      (fun q z h => (hp.2 q z h).symm) region offset hn

/-- Both source and candidate run from the finite image of the initial state.
No successful intermediate execution is assumed. -/
theorem source_correct (swapped : Bool) (A : FlatAlloc) (hd : A.Disjoint)
    (hc : ∀ r, r ∉ A.regions → A.extent r = 0)
    (p : Layout) (s o : RegionName) (dtype out_dtype : FloatDType)
    (initial : BlockState) (hSO : s ≠ o) (hBT : 0 < p.BT)
    (hinj : NoCollision p (RealInit.launch initial))
    (hb : RealSafety.Bounds p (RealInit.launch initial) s o A.extent)
    (hu : initial.undef = fun _ _ => 0) :
    ∃ final, exec (A.flattenKernel
      (RealSource.source swapped dtype out_dtype s o p.headStride p.timeStride p.channelStride
        p.T p.S p.BT p.BS).eraseDType.toAlgKernel) (A.flattenState initial) = some final ∧
      Post A p s o initial final := by
  obtain ⟨final, he, hp⟩ := RealExecution.source_correct swapped p s o dtype out_dtype
    initial hSO hBT hinj
  refine ⟨A.flattenState final, ?_, flatten_post A hd hc p s o initial final hBT hb hu hp⟩
  rw [A.exec_flatten hd hc _ initial
    (RealSafety.source_safe swapped p s o dtype out_dtype initial hSO hBT hinj A.extent hb)
    (RealSafety.source_flatten_ok swapped p s o dtype out_dtype) hu, he]
  rfl

#axiomsClean flatten_post
#axiomsClean source_correct
end
end VeriTile.Bench.TritonBenchG.ReversedCumsum.RealIO
