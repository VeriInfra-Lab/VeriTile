import bench.tritonbench_g.softmax_triton1.IO
import bench.tritonbench_g.softmax_triton1.Proofs.RealIO
namespace VeriTile.Bench.Repairs.SoftmaxTriton1.Logical
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedKernelIO₁
noncomputable section
/-- Complete old/new source judgments coincide on the unchanged full B-lane IO. -/
theorem judgment_iff (out inp : RegionName) (ib ob n B : Nat) (cover : n ≤ B) :
    (RealIO.softmaxIO out inp ib ob n B ⊨ fun xs i =>
      Real.exp (xs i) / ∑ j : Fin n, Real.exp (active cover xs j)) ↔
    (RealIO.softmaxIO out inp ib ob n B ⊨ fun xs i =>
      Real.exp (xs i) / ∑ j : Fin n, Real.exp (xs (Fin.castLE cover j))) := Iff.rfl
end
end VeriTile.Bench.Repairs.SoftmaxTriton1.Logical
