import bench.tritonbench_g.fused_recurrent_delta.MathematicalIO
import bench.tritonbench_g.fused_recurrent_delta.Proofs.MathAgreement
import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelRealGradients
import bench.tritonbench_g.fused_recurrent_delta.Proofs.IOInput
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.IOSaved
open VeriTile.Triton FP.Structural Interface
noncomputable section

def cutIncoming (s : Shape) (g : Nat → Fin s.BBV → ℝ) (t : Nat) (j : Fin s.BBV) :=
  if t<s.T then g t j else 0

theorem cutoff_saved (s : Shape) (f : Flags) (iv : Nat)
    (a : Fin s.BBV → Math.Input s.BK) (g : Nat → Fin s.BBV → ℝ)
    (terminal : Fin s.BBV → Math.Vec s.BK) (vals : BackwardInputs ℝ s)
    (ready : SavedInputs s f iv a g terminal vals) :
    KernelReal.Saved s (backwardData KernelReal.algebra s f iv vals)
      (fun j => MathAgreement.cutoff s.T (a j)) (cutIncoming s g) terminal := by
  constructor
  · intro t i j
    by_cases ht : t<s.T
    · simpa [MathAgreement.cutoff,ht] using ready.query t ht i j
    · simp [backwardData,MathAgreement.cutoff,ht,KernelReference.zero,KernelReal.algebra]
  · intro t i j
    by_cases ht : t<s.T
    · simpa [MathAgreement.cutoff,ht] using ready.key t ht i j
    · simp [backwardData,MathAgreement.cutoff,ht,KernelReference.zero,KernelReal.algebra]
  · intro t j
    by_cases ht : t<s.T
    · simpa [MathAgreement.cutoff,ht] using ready.beta t ht j
    · simp [backwardData,MathAgreement.cutoff,ht,KernelReference.zero,KernelReal.algebra]
  · intro t ht j
    have state := (MathAgreement.cutoff_agrees s.T (a j)).state t (by omega)
    rw [state]
    simpa only [MathAgreement.cutoff,if_pos ht] using ready.residual t ht j
  · intro t j
    by_cases ht : t<s.T
    · simpa [cutIncoming,ht] using ready.incoming t ht j
    · simp [backwardData,cutIncoming,ht,KernelReference.zero,KernelReal.algebra]
  · exact ready.initial
  · exact ready.terminal

theorem gradient (s : Shape) (scale : ℝ) (a : Fin s.BBV → Math.Input s.BK)
    (g : Nat → Fin s.BBV → ℝ) (terminal : Fin s.BBV → Math.Vec s.BK)
    (t : Nat) (ht : t<s.T) (j : Fin s.BBV) :
    KernelReal.gradient s scale (fun j => MathAgreement.cutoff s.T (a j)) (cutIncoming s g) terminal t j=
      Math.gradient scale (a j) (fun n => g n j) (terminal j) s.T t := by
  exact (MathAgreement.cutoff_agrees s.T (a j)).gradient scale _ _
    (by intro n hn; simp [cutIncoming,hn]) (terminal j) t ht

theorem seed (s : Shape) (scale : ℝ) (a : Fin s.BBV → Math.Input s.BK)
    (g : Nat → Fin s.BBV → ℝ) (terminal : Fin s.BBV → Math.Vec s.BK) (j : Fin s.BBV) :
    Math.carry scale (MathAgreement.cutoff s.T (a j)) (fun n => cutIncoming s g n j) (terminal j) s.T 0=
      Math.carry scale (a j) (fun n => g n j) (terminal j) s.T 0 := by
  unfold Math.carry
  exact (MathAgreement.cutoff_agrees s.T (a j)).reverse scale _ _
    (by intro n hn; simp [cutIncoming,hn]) (terminal j) _ (by omega)

/-- Every previously admitted raw saved-input relation induces the new finite
logical relation. No condition on the original time suffix is added. -/
theorem from_original (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (a : Fin s.BBV → Math.Input s.BK) (g : Nat → Fin s.BBV → ℝ)
    (terminal : Fin s.BBV → Math.Vec s.BK) (vals : BackwardInputs ℝ s) (u : State ℝ)
    (hin : (backwardIO KernelReal.algebra s f scale bh iv).InputBinding vals u)
    (saved : KernelReal.Saved s (KernelBackwardReference.input KernelReal.algebra s f bh iv u) a g terminal) :
    SavedInputs s f iv a g terminal vals := by
  have agree := IOInput.backward_data KernelReal.algebra s f scale bh iv vals u hin
  constructor
  · intro t ht i j; exact (congrFun (agree.query t ht) i).symm.trans (saved.query t i j)
  · intro t ht i j; exact (congrFun (agree.key t ht) i).symm.trans (saved.key t i j)
  · intro t ht j; exact (congrFun (agree.beta t ht) j).symm.trans (saved.beta t j)
  · intro t ht j; exact (congrFun (agree.value t ht) j).symm.trans (saved.residual t ht j)
  · intro t ht j; exact (congrFun (agree.incoming t ht) j).symm.trans (saved.incoming t j)
  · intro i j; exact (congrFun agree.initial (i,j,())).symm.trans (saved.initial i j)
  · intro i j; exact (congrFun agree.terminal (i,j,())).symm.trans (saved.terminal i j)

#axiomsClean cutoff_saved
#axiomsClean gradient
#axiomsClean seed
#axiomsClean from_original
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.IOSaved
