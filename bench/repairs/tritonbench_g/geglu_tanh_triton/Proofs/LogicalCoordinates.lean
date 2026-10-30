import bench.repairs.tritonbench_g.geglu_tanh_triton.IO
import bench.repairs.tritonbench_g.geglu_tanh_triton.Proofs.PublicIO

namespace VeriTile.Bench.Repairs.GegluTanhTriton.IO
open VeriTile Triton FP.Structural
noncomputable section

/-- Both directions of the complete existing forward/backward grid judgment. -/
theorem logical_iff (candidate backward : Bool) (count : Nat) (order : List Nat)
    (f : Channel → ℝ → ℝ → ℝ → ℝ) :
    (logicalIO candidate backward count order).Implements
      (fun xs ch i => f ch (xs.a i) (xs.b i) (if backward then xs.incoming i else 0)) ↔
    (Reference.realIO candidate backward count order).Implements (Reference.formula backward count f) := by
  rfl

theorem logical_real (candidate backward : Bool) (count : Nat) (order : List Nat)
    (legal : order.Perm (List.range (Grid.grid count))) :
    (logicalIO candidate backward count order).Implements
      (fun xs ch i => RealExecution.formula ch (xs.a i) (xs.b i) (if backward then xs.incoming i else 0)) :=
  (logical_iff candidate backward count order RealExecution.formula).mpr
    (PublicIO.real candidate backward count order legal)
end
end VeriTile.Bench.Repairs.GegluTanhTriton.IO
