import bench.tritonbench_g.softmax_triton3.IO
import bench.tritonbench_g.softmax_triton3.Proofs.RealIO
import bench.tritonbench_g.softmax_triton1.Proofs.KernelMath
namespace VeriTile.Bench.Repairs.SoftmaxTriton3.Logical
open VeriTile.Triton Correct
open VeriTile.Bench.Repairs.SoftmaxTriton1 hiding blockSize numWarps
open scoped VeriTile.Triton.MaskedKernelIO₁ VeriTile.Triton.MaskedKernelIO₂
noncomputable section
theorem judgment_iff (out inp : RegionName) (n B : Nat) (cover : n ≤ B) :
    (softmaxIO out inp n B ⊨ fun xs i =>
      Real.exp (xs i) / ∑ j : Fin n, Real.exp (active cover xs j)) ↔
    (softmaxIO out inp n B ⊨ fun xs i =>
      Real.exp (xs i) / ∑ j : Fin n, Real.exp (xs (Fin.castLE cover j))) := Iff.rfl

theorem masked_judgment_iff (out inp mask : RegionName) (n B : Nat) (cover : n ≤ B) :
    (softmaxMaskedIO out inp mask n B ⊨ fun xs ms i =>
      Real.exp (xs i+ms i) / ∑ j : Fin n, Real.exp (active cover xs j+active cover ms j)) ↔
    (softmaxMaskedIO out inp mask n B ⊨ fun xs ms i =>
      Real.exp (xs i+ms i) / ∑ j : Fin n, Real.exp (xs (Fin.castLE cover j)+ms (Fin.castLE cover j))) := Iff.rfl

theorem unmasked (out inp : RegionName) (n B : Nat) (positive : 0 < B) (cover : n ≤ B) :
    softmaxIO out inp n B ⊨ fun xs i =>
      Real.exp (xs i) / ∑ j : Fin n, Real.exp (active cover xs j) := by
  have old : softmaxIO out inp n B ⊨ fun xs i =>
      Real.exp (xs i) / ∑ j : Fin B, if j.val < n then Real.exp (xs j) else 0 :=
    RealIO.unmasked out inp n B positive cover
  simpa only [active,RealModel.masked_sum_eq_logical n B cover] using old

theorem masked (out inp mask : RegionName) (n B : Nat) (positive : 0 < B) (cover : n ≤ B) :
    softmaxMaskedIO out inp mask n B ⊨ fun xs ms i =>
      Real.exp (xs i+ms i) / ∑ j : Fin n, Real.exp (active cover xs j+active cover ms j) := by
  have old : softmaxMaskedIO out inp mask n B ⊨ fun xs ms i =>
      Real.exp (xs i+ms i) / ∑ j : Fin B, if j.val < n then Real.exp (xs j+ms j) else 0 :=
    RealIO.masked out inp mask n B positive cover
  simpa only [active,RealModel.masked_sum_eq_logical n B cover] using old
end
end VeriTile.Bench.Repairs.SoftmaxTriton3.Logical
