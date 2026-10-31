import bench.tritonbench_g.attention_llama.Proofs.Mathematical_FP8RealLoopFrame

/-! Lift the checked FP8 iteration through the full dynamic range. Byte
requirements refer to the actual visited block starts and active source masks.
The invariant tracks raw memory and pointer positions, not output equations. -/
namespace VeriTile.Bench.TritonBenchG.AttentionLlama.FP8RealLoop
open VeriTile.Triton RealKernel FP8RealMemory FP8RealIteration
set_option maxRecDepth 16000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

/-- The input addresses at the next streaming counter. Numeric accumulator
registers are deliberately unrestricted by this structural invariant. -/
structure Cursor (K V : RegionName) (initial : BlockState)
    (kBase vBase skn skk svk svn BN BD i : Nat) (s : BlockState) : Prop where
  memory : s.mem = initial.mem
  offsets : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val))
  keys : s.regs .ptr [BN, BD] "k_ptrs" = some (alPtrTile K kBase skn skk i BN BD)
  values : s.regs .ptr [BN, BD] "v_ptrs" = some (alPtrTile V vBase svk svn i BN BD)

/-- Finite signed bytes are needed only for reached block starts and active
lanes. Inactive cells are not constrained, including the final partial block. -/
def BytesFrom (R : RegionName) (s : BlockState)
    (base srow scol cur stop N BN BD : Nat) : Prop :=
  ∀ k : Nat, cur + k * BN < stop → ∀ (j : Fin BN) (d : Fin BD),
    cur + k * BN + j.val < N →
      FiniteByte (s.readMemValue .int R
        (base + (cur + k * BN + j.val) * srow + d.val * scol))

private theorem bytes_next (R : RegionName) (s : BlockState)
    (base srow scol cur stop N BN BD : Nat)
    (h : BytesFrom R s base srow scol cur stop N BN BD) :
    BytesFrom R s base srow scol (cur + BN) stop N BN BD := by
  intro k hk j d hj
  have he : cur + BN + k * BN = cur + (k + 1) * BN := by
    simp only [Nat.add_mul, Nat.one_mul]
    omega
  rw [he] at hk hj ⊢
  exact h (k + 1) hk j d hj

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionLlama.FP8RealLoop
