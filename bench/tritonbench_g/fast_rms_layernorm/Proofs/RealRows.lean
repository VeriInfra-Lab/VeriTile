import bench.tritonbench_g.fast_rms_layernorm.Proofs.RealReads
import bench.tritonbench_g.fast_rms_layernorm.Proofs.Mathematical_ForwardRun
import bench.tritonbench_g.fast_rms_layernorm.Proofs.Mathematical_BackwardRun
import bench.tritonbench_g.fast_rms_layernorm.Proofs.Mathematical_BackwardMath

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.RealRows
open VeriTile.Triton RealBridge
open VeriTile.Bench.TritonBenchG.FastRmsLayernorm
noncomputable section

def raw (h : Call) (u : BlockState) (r : RegionName) (stride : Nat) (j : Fin h.cols) : ℝ :=
  u.readMem r (u.pids 0*stride+j.val)

def forwardFootprint (h : Call) (u : BlockState) (q : RegionName) (o : Nat) : Prop :=
  (q="y" ∧ ∃ j : Fin h.cols,o=u.pids 0*h.cols+j.val) ∨ (q="r" ∧ o=u.pids 0)

theorem forward_complete (h : Call) (u : BlockState) :
    ∃ v, exec (Kernels.forward h).eraseDType.toAlgKernel u=some v ∧
      (∀ j : Fin h.cols,v.readMem "y" (u.pids 0*h.cols+j.val)=
        RealReference.output h.gemma h.eps (raw h u "x" h.cols) (raw h u "w" 0) j) ∧
      v.readMem "r" (u.pids 0)=RealReference.inv h.eps (raw h u "x" h.cols) ∧
      (∀ q o,¬forwardFootprint h u q o → v.mem q o=u.mem q o) := by
  obtain ⟨v,he,hy,hr⟩ := ForwardRun.run h.gemma "y" "x" "w" "r" h.cols h.cols 1 1 h.cols h.block h.eps u (by decide)
  have weights (j : Fin h.block) (hj : j.val<h.cols) :
      u.readMem "w" (ForwardRun.weightOffset h.gemma 1 j.val)=raw h u "w" 0 ⟨j.val,hj⟩ := by
    cases h.gemma <;> simp [ForwardRun.weightOffset,raw]
  refine ⟨v,he,?_,?_,?_⟩
  · intro j
    exact (hy (lane h j) j.isLt).trans
      (RealReads.output_agrees h _ _ (raw h u "x" h.cols) (raw h u "w" 0) (fun _ _ => rfl) weights j)
  · simpa using hr.trans (RealReads.inv_agrees h _ (raw h u "x" h.cols) (fun _ _ => rfl))
  · intro q o outside
    apply ForwardRun.frame h.gemma "y" "x" "w" "r" h.cols h.cols 1 1 h.cols h.block h.eps u v he q o
    · intro j hj ⟨same,eq⟩
      exact outside (Or.inl ⟨same.symm,⟨⟨j.val,hj⟩,eq.symm⟩⟩)
    · rintro ⟨same,eq⟩
      exact outside (Or.inr ⟨same.symm,by simpa using eq.symm⟩)

def backwardFootprint (h : Call) (u : BlockState) (q : RegionName) (o : Nat) : Prop :=
  q="dx" ∧ ∃ j : Fin h.cols,o=u.pids 0*h.cols+j.val

theorem backward_values (h : Call) (u : BlockState) :
    ∃ v, exec (Kernels.backward h).eraseDType.toAlgKernel u=some v ∧
      (∀ j : Fin h.cols,v.readMem "dx" (u.pids 0*h.cols+j.val)=
        RealMath.gradient h.gemma h.cols (u.readMem "r" (u.pids 0))
          (fun k => u.readMem "x" (u.pids 0*h.cols+k.val))
          (fun k => u.readMem "dx" (u.pids 0*h.cols+k.val))
          (fun k => u.readMem "w" k.val) (lane h j)) ∧
      (∀ q o,¬backwardFootprint h u q o → v.mem q o=u.mem q o) := by
  obtain ⟨v,he⟩ := BackwardRun.executes h.gemma "dx" "x" "w" "r" "dw" h.cols h.cols 1 1 1 h.cols h.block h.eps u
  have injective : Function.Injective (fun j : Fin h.block => RealExecution.dyOutOffset u h.cols j) := by
    intro a b eq
    exact Fin.ext (Nat.add_left_cancel eq)
  have result (j : Fin h.block) (hj : j.val<h.cols) : v.readMem "dx" (u.pids 0*h.cols+j.val)=
      VeriTile.Bench.TritonBenchG.FastRmsLayernorm.RealMath.gradient h.gemma h.cols (u.readMem "r" (u.pids 0))
        (fun k => u.readMem "x" (u.pids 0*h.cols+k.val)) (fun k => u.readMem "dx" (u.pids 0*h.cols+k.val))
        (fun k => u.readMem "w" k.val) j := by
    cases mode : h.gemma
    · have run := he
      rw [mode] at run
      have hh := RealExecution.rms_layernorm_backward_dy_correct "dx" "x" "w" "r" "dw" h.cols h.cols 1 1 1 h.cols h.block h.eps u v injective run j
      rw [if_pos hj] at hh
      simpa using hh.trans (BackwardMath.plain_gradient u "dx" "x" "w" "r" h.cols h.cols 1 h.cols h.block j hj)
    · have run := he
      rw [mode] at run
      have hh := RealExecution.gemma_rms_layernorm_backward_dy_correct "dx" "x" "w" "r" "dw" h.cols h.cols 1 1 1 h.cols h.block h.eps u v injective run j
      rw [if_pos hj] at hh
      simpa using hh.trans (BackwardMath.gemma_gradient u "dx" "x" "w" "r" h.cols h.cols 1 h.cols h.block j hj)
  refine ⟨v,he,?_,?_⟩
  · intro j
    exact result (lane h j) j.isLt
  · intro q o outside
    apply BackwardRun.frame h.gemma "dx" "x" "w" "r" "dw" h.cols h.cols 1 1 1 h.cols h.block h.eps u v he q o
    intro j hj ⟨same,eq⟩
    exact outside ⟨same.symm,⟨⟨j.val,hj⟩,eq.symm⟩⟩

theorem backward_complete (h : Call) (u : BlockState)
    (saved : u.readMem "r" (u.pids 0)=RealReference.inv h.eps (raw h u "x" h.cols)) :
    ∃ v, exec (Kernels.backward h).eraseDType.toAlgKernel u=some v ∧
      (∀ j : Fin h.cols,v.readMem "dx" (u.pids 0*h.cols+j.val)=
        RealReference.inputGradient h.gemma h.eps (raw h u "x" h.cols) (raw h u "w" 0) (raw h u "dx" h.cols) j) ∧
      (∀ q o,¬backwardFootprint h u q o → v.mem q o=u.mem q o) := by
  obtain ⟨v,he,ho,hf⟩:=backward_values h u
  refine ⟨v,he,?_,hf⟩
  intro j
  have row:=ho j
  rw [saved] at row
  exact row.trans (RealReads.gradient_agrees h _ _ _ (raw h u "x" h.cols) (raw h u "w" 0) (raw h u "dx" h.cols)
    (fun _ _ => rfl) (by intros; simp [raw]) (fun _ _ => rfl) j)

#axiomsClean forward_complete
#axiomsClean backward_values
#axiomsClean backward_complete
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.RealRows
