import bench.repairs.tritonbench_g.token_attn_mistral_grid_y_checked.IO
import bench.repairs.tritonbench_g.token_attn_mistral_grid_y_checked.Proofs.PublicIO
namespace VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked.Logical
open VeriTile Triton FP.Structural TokenAttnMistral
noncomputable section
theorem judgment_iff (p : Parameters) (work : List Grid.PID) (counts : Fin work.length → Nat) :
    (io p work counts).ImplementsOn (domain p work counts)
      (fun xs job d => ∑ n ∈ Finset.range (xs.attention job),
        if n + max (xs.sequence job-p.window) 0 < xs.sequence job then
          Reference.probAt 0 xs job n * Reference.valueAt 0 xs job n d else 0) ↔
    (IO.real p work counts).ImplementsOn (Ports.Domain p work counts) (Reference.mathematical p work counts) := by
  have same : (io p work counts).lift
      (fun xs job d => ∑ n ∈ Finset.range (xs.attention job),
        if n + max (xs.sequence job-p.window) 0 < xs.sequence job then
          Reference.probAt 0 xs job n * Reference.valueAt 0 xs job n d else 0) =
      Reference.mathematical p work counts := by
    funext xs port
    fin_cases port
    rfl
  change (IO.real p work counts).ImplementsOn _ _ ↔ _
  rw [same]
  rfl
end
end VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked.Logical
