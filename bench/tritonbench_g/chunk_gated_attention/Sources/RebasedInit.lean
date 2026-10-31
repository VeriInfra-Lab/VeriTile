import bench.tritonbench_g.chunk_gated_attention.Sources.RebasedAccess
import bench.tritonbench_g.chunk_gated_attention.Sources.Grid
import bench.tritonbench_g.chunk_gated_attention.Sources.Mathematical_RealHInit

/-! Flattened launch decoder and the actual zero-plus-seed initialization.
The old mathematical starting-state and invariant definitions are reused. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedInit
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealHMath RealHProgram RealHTiles RealHLoads
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def decoded (p : Layout) (base : Nat) (u : BlockState) : Launch :=
  let linear := u.pids 0+base
  ⟨linear % (p.V/p.BV), (linear / (p.V/p.BV)) % (p.K/p.BK),
    linear / ((p.V/p.BV)*(p.K/p.BK))⟩

theorem seed_load (p : Layout) (l : Launch) (r : RegionName) (a : Nat → ℝ) (u : BlockState)
    (hk : l.key*p.BK+p.BK ≤ p.K) (hv : l.value*p.BV+p.BV ≤ p.V)
    (hm : ∀ off, u.readMem r off = a off)
    (hp : u.regs .blockPtr [p.BK,p.BV] "p_h" = some (RebasedAccess.statePointer p l r)) :
    evalOp (load2 p.BK p.BV "p_h") u = some (stateTile p (seedData p l a (fun _ => 0))) := by
  rw [RebasedAccess.matrix_load r u "p_h" _ _ _ _ _ hp]
  congr 1
  apply Tile.ext
  rintro ⟨e,j,⟨⟩⟩
  have he : l.key*p.BK+e.val < p.K := by omega
  have hj : l.value*p.BV+j.val < p.V := by omega
  simp only [stateTile,Tile.mat,RealHLoads.seedData,keyIndex,valueIndex,he,hj,and_self,if_true,hm,
    stateAddress,Nat.mul_one]
  congr 2 <;> ring

theorem add_seed (p : Layout) (l : Launch) (r : RegionName) (a : Nat → ℝ) (u : BlockState)
    (hk : l.key*p.BK+p.BK ≤ p.K) (hv : l.value*p.BV+p.BV ≤ p.V)
    (hm : ∀ off, u.readMem r off = a off)
    (hp : u.regs .blockPtr [p.BK,p.BV] "p_h" = some (RebasedAccess.statePointer p l r))
    (hh : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p (fun _ _ => 0))) :
    evalOp (.add .real (.consSame (.consSame .nil)) (carry p) (load2 p.BK p.BV "p_h")) u =
      some (stateTile p (seedData p l a (fun _ => 0))) := by
  rw [evalOp_add,carry,evalOp_ref,hh,seed_load p l r a u hk hv hm hp]
  simp only [Option.bind_eq_bind,Option.bind_some]
  congr 1
  apply Tile.ext
  rintro ⟨e,j,⟨⟩⟩
  simp [stateTile,Tile.mat,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.add]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedInit
