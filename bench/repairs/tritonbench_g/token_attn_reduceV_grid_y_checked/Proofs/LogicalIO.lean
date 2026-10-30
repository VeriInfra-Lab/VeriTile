import bench.repairs.tritonbench_g.token_attn_reduceV_grid_y_checked.IO
import bench.repairs.tritonbench_g.token_attn_reduceV_grid_y_checked.Proofs.RealIO
namespace VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked.Logical
open VeriTile.Triton FP.Structural TokenAttnReduceV
noncomputable section
theorem judgment_iff (p : Parameters) (B H : Nat) (counts : Fin B → Nat) (work : List Grid.PID) :
    (io p B H counts work).ImplementsOn (domain p B H counts)
      (fun xs b h d => ∑ n ∈ Finset.range (counts b), xs.probAt b h n * xs.valueAt b h n d) ↔
    (Interface.programIO Real.algebra Real.sum p B H counts work).ImplementsOn
      (Interface.LengthsMatch p B H counts) (Interface.mathematical p B H counts) := by
  have same : (io p B H counts work).lift
      (fun xs b h d => ∑ n ∈ Finset.range (counts b), xs.probAt b h n * xs.valueAt b h n d) =
      Interface.mathematical p B H counts := by
    funext xs port
    fin_cases port
    rfl
  change (Interface.programIO Real.algebra Real.sum p B H counts work).ImplementsOn _ _ ↔ _
  rw [same]
  rfl
end
end VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked.Logical
