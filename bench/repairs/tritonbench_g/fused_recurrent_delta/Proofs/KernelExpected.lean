import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelGeometry
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelCompiler
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelExpected
open VeriTile.Triton FP.Structural KernelProgram KernelCompiler KernelNativeCarrier
noncomputable section

theorem written (platform : Platform) (binding : Binding platform kind s f) (host : UInt64)
    (p : Program kind s) (u : State Node) (r : RegionName) (off : Nat) (word : Word)
    (expected : Expected platform binding host p u r off word) : KernelGeometry.Writes kind s f p r off := by
  cases kind with
  | forward =>
    cases expected with
    | output t j active => exact Or.inl ⟨Or.inr rfl,t.val,j,t.isLt,active,rfl⟩
    | residual t j active => exact Or.inl ⟨Or.inl rfl,t.val,j,t.isLt,active,rfl⟩
    | terminal enabled i j key value => exact Or.inr ⟨enabled,rfl,i,j,key,value,rfl⟩
  | backward =>
    cases expected with
    | query t i active => exact Or.inl ⟨Or.inr rfl,t.val,i,t.isLt,active,rfl⟩
    | key t i active => exact Or.inl ⟨Or.inl rfl,t.val,i,t.isLt,active,rfl⟩
    | value t j active => exact Or.inr (Or.inl ⟨rfl,t.val,j,t.isLt,active,rfl⟩)
    | vectorBeta enabled t j active =>
      exact Or.inr (Or.inr (Or.inl ⟨rfl,by simp only [enabled,ite_true]; exact ⟨t.val,j,t.isLt,active,rfl⟩⟩))
    | scalarBeta enabled t =>
      exact Or.inr (Or.inr (Or.inl ⟨rfl,by simp only [enabled,Bool.false_eq_true,ite_false]; exact ⟨t.val,t.isLt,rfl⟩⟩))
    | seed enabled i j key value => exact Or.inr (Or.inr (Or.inr ⟨enabled,rfl,i,j,key,value,rfl⟩))

theorem exists_word (platform : Platform) (binding : Binding platform kind s f) (host : UInt64)
    (p : Program kind s) (u : State Node) (r : RegionName) (off : Nat)
    (active : KernelGeometry.Writes kind s f p r off) : ∃ word,Expected platform binding host p u r off word := by
  cases kind with
  | forward =>
    rcases active with ⟨role,t,j,ht,hj,rfl⟩|⟨enabled,rfl,i,j,hi,hj,rfl⟩
    · rcases role with rfl|rfl
      · exact ⟨_,KernelNative.ForwardExpected.residual ⟨t,ht⟩ j hj⟩
      · exact ⟨_,KernelNative.ForwardExpected.output ⟨t,ht⟩ j hj⟩
    · exact ⟨_,KernelNative.ForwardExpected.terminal enabled i j hi hj⟩
  | backward =>
    rcases active with ⟨role,t,i,ht,hi,rfl⟩|⟨rfl,t,j,ht,hj,rfl⟩|⟨rfl,beta⟩|⟨enabled,rfl,i,j,hi,hj,rfl⟩
    · rcases role with rfl|rfl
      · exact ⟨_,KernelNative.BackwardExpected.key ⟨t,ht⟩ i hi⟩
      · exact ⟨_,KernelNative.BackwardExpected.query ⟨t,ht⟩ i hi⟩
    · exact ⟨_,KernelNative.BackwardExpected.value ⟨t,ht⟩ j hj⟩
    · cases hb : f.vectorBeta <;> simp only [hb,Bool.false_eq_true,ite_false,ite_true] at beta
      · obtain ⟨t,ht,rfl⟩ := beta
        exact ⟨_,KernelNative.BackwardExpected.scalarBeta hb ⟨t,ht⟩⟩
      · obtain ⟨t,j,ht,hj,rfl⟩ := beta
        exact ⟨_,KernelNative.BackwardExpected.vectorBeta hb ⟨t,ht⟩ j hj⟩
    · exact ⟨_,KernelNative.BackwardExpected.seed enabled i j hi hj⟩

#axiomsClean written
#axiomsClean exists_word
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelExpected
