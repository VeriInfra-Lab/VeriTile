import bench.tritonbench_g.bgmv_shrink_kernel.Target
namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.Target
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry
noncomputable section
variable {α : Type} [Inhabited α]

theorem products_agree (M : Algebra α) (h : Call) (data : Data α) (b slot split c : Nat) :
    FPValues.products M h data b slot split c=products M h data b slot split c := by
  funext i
  simp only [FPValues.products,products,FPValues.loaded,loaded,FPValues.weight,weight,
    VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry.rank,VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry.inputOffset,VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry.weightOffset,Nat.add_assoc]
  rfl

theorem sums_agree (M : Algebra α) (P : Primitives α) (h : Call) (data : Data α)
    (b slot split c : Nat) : FPValues.sums M P h data b slot split c=sums M P h data b slot split c := by
  unfold FPValues.sums sums
  rw [products_agree]
  rfl

theorem accumulator_agree (M : Algebra α) (P : Primitives α) (h : Call) (data : Data α)
    (b slot split c : Nat) :
    FPValues.accumulator M P h data b slot split c=accumulator M P h data b slot split c := by
  induction c with
  | zero => rfl
  | succ c ih =>
    funext n
    simp only [FPValues.accumulator,accumulator,ih,sums_agree]
    rfl

theorem agrees (M : Algebra α) (P : Primitives α) (h : Call) (orders : VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset.Orders h)
    (data : Data α) (b : Fin h.batches) (n : Fin h.N) :
    FPValues.target M P h orders data b n=target M P h orders data b n := by
  have payloadEq : FPValues.payload M P h data=payload M P h data := by
    funext b slot split n
    simp only [FPValues.payload,payload,accumulator_agree]
  simp only [FPValues.target,target,payloadEq,VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry.outputOffset]

theorem real_agrees (h : Call) (u : BlockState) (indices : Nat → Int)
    (b : Fin h.batches) (n : Fin h.N) : VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.CheckedReal.math h u indices b n=real h u indices b n := rfl

#axiomsClean agrees
#axiomsClean real_agrees
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.Target
