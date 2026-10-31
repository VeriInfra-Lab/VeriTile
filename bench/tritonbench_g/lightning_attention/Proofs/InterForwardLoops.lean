import bench.tritonbench_g.lightning_attention.Proofs.InterQueryBody
import bench.tritonbench_g.lightning_attention.Proofs.InterDeltaBody
namespace VeriTile.Bench.Repairs.LightningAttention
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]
theorem foldRange_id {β : Type} (n : Nat) (x : β) : foldRange (fun _ a => a) 0 n 1 x=x := by
  induction n with
  | zero => exact foldRange_zero _ _
  | succ n ih => rw [foldRange_succ,ih]

theorem querySubFold (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head block : Nat)
    (carry : Matrix α p.E p.D) (m : Memory α) :
    foldRange (fun sub mem => querySubMemory M D p head ((block*p.NC+sub)*p.CB) block carry mem)
      0 p.NC 1 m=queryBlockMemory M D p head block carry m := by
  by_cases hb : block=0
  · simp [querySubMemory,queryBlockMemory,hb,foldRange_id]
  · simp only [querySubMemory,queryBlockMemory,hb,if_false]

end
end VeriTile.Bench.Repairs.LightningAttention
