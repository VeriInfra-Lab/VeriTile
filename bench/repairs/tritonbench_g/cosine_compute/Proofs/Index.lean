import bench.repairs.tritonbench_g.cosine_compute.Proofs.Contract
import bench.tritonbench_g.add_example.Proofs.FiniteIndices

namespace VeriTile.Bench.Repairs.CosineCompute

def offset32 (pid lane : Nat) : BitVec 32 :=
  BitVec.ofNat 32 pid * BitVec.ofNat 32 256 + BitVec.ofNat 32 lane
def mask32 (n pid lane : Nat) : Bool := (offset32 pid lane).slt (BitVec.ofNat 32 n)

theorem offset_word (pid lane : Nat) :
    offset32 pid lane = BitVec.ofNat 32 (pid*256+lane) := by
  simp only [offset32, BitVec.ofNat_mul_ofNat, BitVec.ofNat_add_ofNat]

theorem indices (n pid lane : Nat) (hn : n < 2^31)
    (hp : pid < gridCount n) (hl : lane < 256) :
    pid*256+lane < 2^31 ∧
    (offset32 pid lane).toInt = (pid*256+lane : Nat) ∧
    mask32 n pid lane = decide (pid*256+lane < n) := by
  have hg : gridCount n ≤ 2^23 := by unfold gridCount; omega
  have hi : pid*256+lane < 2^31 := by omega
  have ho : (offset32 pid lane).toInt = (pid*256+lane : Nat) := by
    rw [offset_word]
    exact VeriTile.Bench.TritonBenchG.AddExample.Finite.nat32_signed _ hi
  refine ⟨hi, ho, ?_⟩
  rw [mask32, BitVec.slt_eq_decide, ho,
    VeriTile.Bench.TritonBenchG.AddExample.Finite.nat32_signed n hn]
  simp only [Int.ofNat_lt]

theorem host_grid (n : Nat) :
    gridCount n = (n+blockSize-1)/blockSize ∧
    (∀ i, i < n → i/256 < gridCount n ∧ i%256 < 256 ∧ i/256*256+i%256=i) := by
  constructor
  · simp [gridCount,blockSize]
  · intro i hi; unfold gridCount; omega

#axiomsClean indices
#axiomsClean host_grid
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.CosineCompute
