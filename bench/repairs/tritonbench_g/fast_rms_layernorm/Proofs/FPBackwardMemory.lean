import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.FPBackward
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.FPMemory

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.FPBackwardMemory
open VeriTile.Triton FP.Structural FPSource FPMemory
noncomputable section
variable {α : Type} [Inhabited α]

def footprint (h : Call) (u : State α) (q : RegionName) (o : Nat) : Prop :=
  q="dx" ∧ ∃ j : Fin h.cols,o=u.pids 0*h.cols+j.val

theorem complete (M : Algebra α) (h : Call) (u : State α) :
    ∃ v, FP.Structural.exec (FP.ProfiledRewrite.engine M) (Kernels.backward h) u=some v ∧
      (∀ j : Fin h.cols, v.mem "dx" (u.pids 0*h.cols+j.val)=
        FPBackwardReference.output M h (raw h u "x" h.cols) (raw h u "w" 0)
          (raw h u "dx" h.cols) ((u.mem "r" (u.pids 0)).read .real)
          ⟨j.val,j.isLt.trans_le h.covered⟩) ∧
      (∀ q o,¬footprint h u q o → v.mem q o=u.mem q o) := by
  obtain ⟨v,run,mem⟩:=FPBackward.source_memory M h u
  refine ⟨v,run,?_,?_⟩
  · intro j
    rw [mem]
    have readback := commit_readback "dx" (fun i : TileIndex [h.block] => u.pids 0*h.cols+i.1.val)
      (fun i => FPBackwardReference.output M h (raw h u "x" h.cols) (raw h u "w" 0)
        (raw h u "dx" h.cols) ((u.mem "r" (u.pids 0)).read .real) i.1)
      (fun i => i.1.val<h.cols) u (⟨j.val,j.isLt.trans_le h.covered⟩,PUnit.unit) j.isLt (row_injective _ _)
    have weightZero : raw h u "w" 0=(fun k : Fin h.block => (u.mem "w" k.val).read .real) := by
      funext k
      simp [raw]
    rw [weightZero] at readback ⊢
    exact readback
  · intro q o outside
    rw [mem]
    unfold FPBackward.expectedHeap
    apply commit_frame
    by_cases same : q="dx"
    · right
      intro i _ active eq
      exact outside ⟨same,⟨⟨i.1.val,active⟩,eq⟩⟩
    · exact Or.inl same

#axiomsClean complete
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.FPBackwardMemory
