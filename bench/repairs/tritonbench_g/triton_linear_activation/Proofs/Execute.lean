import bench.repairs.tritonbench_g.triton_linear_activation.Kernels
import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.Autotune
namespace VeriTile.Bench.Repairs.TritonLinearActivation
variable {α : Type} {r : Request} {caller : Set Nat} {before : Memory α}

theorem execute_eq (h : Allocation r caller before) (a : Arithmetic α) (c : Fin 98)
    (currentDevice : Nat) (trials : List (Fin 98)) (valid : r.Valid) :
    execute h a c currentDevice trials=h.hostResult a c currentDevice := by
  unfold execute
  rw [h.eval_autotunedCode a trials c valid]
  rfl
end VeriTile.Bench.Repairs.TritonLinearActivation
