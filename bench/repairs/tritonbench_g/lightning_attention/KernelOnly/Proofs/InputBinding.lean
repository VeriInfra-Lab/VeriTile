import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.Proofs.Reference
import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.PhaseReal
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.IOProof
open VeriTile Triton FP.Structural
noncomputable section

theorem forward_inputs (d : Host.Dimensions) (x : (forwardIO d).Inputs) (s : State ℝ)
    (bound : (forwardIO d).InputBinding x s) (h : Fin (d.B*d.H)) :
    Real.logicalQ d.parameters h.val s.mem = queryValues d x h ∧
    Real.logicalK d.parameters h.val s.mem = keyValues d x h ∧
    Real.logicalV d.parameters h.val s.mem = valueValues d x h := by
  refine ⟨?_,?_,?_⟩
  · funext t a; exact bound (0 : Fin 4) (h,t,a,()) (by change 0 < 3; omega)
  · funext t a; exact bound (1 : Fin 4) (h,t,a,()) (by change 1 < 3; omega)
  · funext t a; exact bound (2 : Fin 4) (h,t,a,()) (by change 2 < 3; omega)

theorem backward_inputs (d : Host.Dimensions) (x : (backwardIO d).Inputs) (s : State ℝ)
    (bound : (backwardIO d).InputBinding x s) (h : Fin (d.B*d.H)) :
    Real.logicalQ d.parameters h.val s.mem = queryValues d x h ∧
    Real.logicalK d.parameters h.val s.mem = keyValues d x h ∧
    Real.logicalV d.parameters h.val s.mem = valueValues d x h ∧
    Real.logicalG d.parameters h.val s.mem = incomingValues d x h := by
  refine ⟨?_,?_,?_,?_⟩
  · funext t a; exact bound (0 : Fin 4) (h,t,a,()) trivial
  · funext t a; exact bound (1 : Fin 4) (h,t,a,()) trivial
  · funext t a; exact bound (2 : Fin 4) (h,t,a,()) trivial
  · funext t a; exact bound (3 : Fin 4) (h,t,a,()) trivial

end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.IOProof
