import bench.tritonbench_g.lightning_attention.Proofs.InterReverseUpdate
import bench.tritonbench_g.lightning_attention.Proofs.InterReverseDelta
import bench.tritonbench_g.lightning_attention.Proofs.InterForwardLoops
namespace VeriTile.Bench.Repairs.LightningAttention
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]
def reverseRow (p : Parameters) (block sub : Nat) := p.NT*p.BT-(block*p.NC+sub)*p.CB

theorem reverseRow_step (p : Parameters) (hbalance : p.NC*p.CB=p.BT)
    (block sub : Nat) (hb : block<p.NT) (hs : sub<p.NC) :
    reverseRow p block (sub+1)+p.CB=reverseRow p block sub := by
  have hcoef : block*p.NC+(sub+1)≤p.NT*p.NC := by
    calc
      _ ≤ (block+1)*p.NC := by rw [Nat.add_mul]; omega
      _ ≤ p.NT*p.NC := Nat.mul_le_mul_right p.NC (by omega)
  have hsize : (block*p.NC+(sub+1))*p.CB≤p.NT*p.BT := by
    calc
      _ ≤ (p.NT*p.NC)*p.CB := Nat.mul_le_mul_right p.CB hcoef
      _ = _ := by rw [Nat.mul_assoc,hbalance]
  have hshift : (block*p.NC+(sub+1))*p.CB=(block*p.NC+sub)*p.CB+p.CB := by ring
  unfold reverseRow
  rw [hshift] at hsize ⊢
  omega

theorem reverseRow_end (p : Parameters) (block : Nat) :
    reverseRow p block p.NC=reverseRow p (block+1) 0 := by simp [reverseRow,Nat.add_mul]

theorem keyValueSubFold (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head block : Nat)
    (carry : Matrix α p.D p.E) (m : Memory α) :
    foldRange (fun sub mem => keyValueSubMemory M D p head (reverseRow p block (sub+1)) block carry mem)
      0 p.NC 1 m=keyValueBlockMemory M D p head block carry m := by
  by_cases hb : block=0
  · simp [keyValueSubMemory,keyValueBlockMemory,hb,foldRange_id]
  · simp only [keyValueSubMemory,keyValueBlockMemory,hb,if_false,reverseRow,Nat.add_assoc]

#axiomsClean reverseRow_step
end
end VeriTile.Bench.Repairs.LightningAttention
