import bench.repairs.tritonbench_g.softmax_triton3.Proofs.Real

namespace VeriTile.Bench.Repairs.SoftmaxTriton3.KernelRealResult
open VeriTile.Triton
open VeriTile.Bench.Repairs.SoftmaxTriton1 hiding blockSize numWarps
noncomputable section

/-- Independent last-axis softmax of input plus the optional additive mask. -/
def reference : Formula ℝ := fun n _ xs mask c =>
  let logits := fun j => xs j+(mask.getD (fun _ => 0)) j
  Real.exp (logits c)/∑ j ∈ Finset.range n,Real.exp (logits j)

/-- Complete mathematical row kernel over raw state and logical lane arrays.
No Tensor object, metadata, checked wrapper, or output allocation occurs here. -/
theorem softmax_kernel_correct (out inp : RegionName) (mask : Option RegionName)
    (rows n B p : Nat) (positive : 0 < B) (cover : n ≤ B) (valid : p < rows)
    (s : BlockState) (xs : Nat → ℝ) (ms : Option (Nat → ℝ))
    (input : ∀ c, c < n → RealModel.access.read s inp (p*n+c) = xs c)
    (masks : ∀ r, mask = some r → ∀ c, c < n →
      RealModel.access.read s r (p*n+c) = (ms.getD (fun _ => RealModel.access.read s inp 0)) c)
    (matching : mask.isSome = ms.isSome) :
    Spec.Real (∃ t, RealModel.evaluate out inp mask rows n B p s = some t ∧
      (∀ c, c < n → RealModel.access.read t out (p*n+c) = reference n B xs ms c) ∧
      (∀ r i, (r ≠ out ∨ ∀ c, c < n → i ≠ p*n+c) → t.mem r i = s.mem r i)) := by
  obtain ⟨t,run⟩ := RealModel.terminates out inp mask rows n B p positive s
  exact ⟨t,run,RealModel.source.row out inp mask rows n B p s t xs ms
    positive cover valid input masks matching run⟩

#specNonCircular reference avoiding [softmax_kernel,RealModel.evaluate]
#axiomsClean softmax_kernel_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SoftmaxTriton3.KernelRealResult
