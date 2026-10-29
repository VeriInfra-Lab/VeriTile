import bench.repairs.tritonbench_g.matmul_triton_autotune.Proofs.KernelTensor
import bench.repairs.tritonbench_g.matmul_triton_autotune.Proofs.NativeGrid

namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Kernel
open VeriTile.Triton FP.Structural Whole
open VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly
noncomputable section
set_option maxHeartbeats 3000000

/-- Kernel-grid execution of the sole source quotation. The native integer
interpreter uses actual Tensor base addresses and agrees on the complete
executed memory; success is a conclusion, never an admission assumption. -/
def Executed {α : Type} [Inhabited α] (M : Algebra α) (D : FPValue.DotAcc α)
    (r : Whole.Request) (s : State α) (t : TensorAPI r s) (ids : List Nat)
    (final : State α) : Prop :=
  Repair.runPrograms M D r ids s = some final ∧
  ∃ native, Native.runPrograms M D r (Native.bases t) ids s = some native ∧
    native.mem = final.mem

theorem execute {α : Type} [Inhabited α]
    (M : Algebra α) (D : FPValue.DotAcc α) (Q : Numerics α) (law : PrimitiveLaws M D Q)
    (a b : Matrix) (al : TensorBinding a) (bl : TensorBinding b) (h : InputContract a b)
    (activation : String) (selection : Selection) (c : OutputTensor)
    (cl : OutputBinding c a b) (fresh : Fresh a b c)
    (schedule : Schedule (request a b activation selection)) (read : Nat → α) (scratch : State α) :
    ∃ final,
      Executed M D (request a b activation selection) (initial a b read scratch)
        (tensorAPI a b al bl h activation selection c cl fresh read scratch) schedule.order final ∧
      Whole.Result Q (request a b activation selection) (initial a b read scratch) final ∧
      Native.FullAccessSafety (request a b activation selection) (initial a b read scratch)
        (tensorAPI a b al bl h activation selection c cl fresh read scratch) := by
  let r := request a b activation selection
  let s := initial a b read scratch
  let t := tensorAPI a b al bl h activation selection c cl fresh read scratch
  obtain ⟨final,he,result⟩ := candidate_grid_order_correct M D Q law r s schedule.order
    schedule.complete (by change ("c" : RegionName) ≠ "a"; decide)
      (by change ("c" : RegionName) ≠ "b"; decide)
  have valid : ∀ pid ∈ schedule.order, pid < programs r ∧ pid < 2^31 := by
    intro pid hp
    have hb := List.mem_range.mp (schedule.complete.mem_iff.mp hp)
    exact ⟨hb, lt_trans hb schedule.programIdWidth⟩
  obtain ⟨native,typed,hn,ht,hm⟩ := Native.programs_correspondence M D r s t
    schedule.order valid s s rfl
  have same : typed = final := Option.some.inj (ht.symm.trans he)
  refine ⟨final,⟨he,native,hn,?_⟩,result,
    Native.full_access_safety r s t schedule.programIdWidth⟩
  simpa only [same] using hm

#axiomsClean execute
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Kernel
