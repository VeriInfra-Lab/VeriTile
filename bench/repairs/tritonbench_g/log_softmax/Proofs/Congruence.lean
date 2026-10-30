import bench.repairs.tritonbench_g.log_softmax.Proofs.Graph
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.LogSoftmax
noncomputable section

/-- Inactive logical-row extensions cannot affect an active result. -/
theorem forward_congr_active (ops : Ops α) (plan : Plan N C) (x y : Nat → α)
    (input : ∀ j,j<N → x j=y j) (i : Nat) (active : i<N) :
    forward ops plan x i=forward ops plan y i := by
  have loads : forwardLoad (C:=C) ops N x=forwardLoad ops N y := by
    funext j
    unfold forwardLoad
    split_ifs with h
    · rw [input j.val h]
    · rfl
  have maximum : rowMax ops plan x=rowMax ops plan y := by
    funext lane
    simp only [rowMax,loads]
  have shifts : shifted ops plan x=shifted ops plan y := by
    funext j
    simp only [shifted,loads,maximum]
  have exps : exponentials ops plan x=exponentials ops plan y := by
    funext j
    simp only [exponentials,shifts]
  simp only [forward,input i active,maximum,denominator,exps]

theorem backward_congr_active (ops : Ops α) (plan : Plan N C) (y g yr gr : Nat → α)
    (inputY : ∀ j,j<N → y j=yr j) (inputG : ∀ j,j<N → g j=gr j)
    (i : Nat) (active : i<N) : backward ops plan y g i=backward ops plan yr gr i := by
  have loads : gradientLoad (C:=C) ops N g=gradientLoad ops N gr := by
    funext j
    unfold gradientLoad
    split_ifs with h
    · rw [inputG j.val h]
    · rfl
  simp only [backward,inputY i active,inputG i active,scale,loads]

#axiomsClean forward_congr_active
#axiomsClean backward_congr_active
end
end VeriTile.Bench.Repairs.LogSoftmax
