import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.PublicResult
import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.Calls

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.PublicData
open VeriTile.Triton FP.Structural Quoted Numeric
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.ofBits .fp32 0⟩

theorem native_complete (mode : Mode) (u : UnaryModels) (ports : ScalarPorts) (laws : ScalarLaws u ports)
    (s : Shape) (shape : s.Admitted) (p : PlanBinding s)
    (incoming finalGradient : Fin 4 → Nat) (seed final : Bool) :
    (io mode (nativeArithmetic ports) s p incoming finalGradient seed final).Observes
      (fun _ _ n => n.value) (expected (arithmetic u) s p seed final) := by
  rw [nativeArithmetic_eq u ports laws]
  exact complete mode (arithmetic u) s shape p incoming finalGradient seed final

#axiomsClean native_complete
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.PublicData
