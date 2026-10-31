import bench.tritonbench_g.chunk_cumsum_vector.Sources.Mathematical_RealLoop
import VeriTile.Triton.Semantics.ProgramRewrite

/-! Source initialization and complete region-memory real correctness. -/
namespace VeriTile.Bench.TritonBenchG.ChunkCumsumVector.RealInit
open VeriTile.Triton RealProgram RealStep RealLoop
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def launch (u : BlockState) : Launch := ⟨u.pids 0, u.pids 1⟩

def initial (p : Layout) (u : BlockState) : BlockState :=
  let u1 := u.setReg "i_s" .nat [] (Tile.scalar (u.pids 0))
  let u2 := u1.setReg "i_bh" .nat [] (Tile.scalar (u.pids 1))
  let u3 := u2.setReg "o_i" .nat [p.BT] ⟨fun i => i.1.val⟩
  let u4 := u3.setReg "m_s" .real [p.BT, p.BT] (lowerTriTile p.BT)
  u4.setReg "b_z" .real [p.BS] (Tile.vec fun _ => some 0)

theorem triangular_eval (p : Layout) (u : BlockState)
    (h : u.regs .nat [p.BT] "o_i" = some ⟨fun i => i.1.val⟩) :
    evalOp (.where (.ge .nat (.consR (.consL .nil))
      (.expandDim ⟨1, by simp⟩ (.ref .nat [p.BT] "o_i"))
      (.expandDim ⟨0, by simp⟩ (.ref .nat [p.BT] "o_i")))
      (.broadcast (.const 1.0) [p.BT, p.BT]) (.broadcast (.const 0.0) [p.BT, p.BT])) u =
        some (lowerTriTile p.BT) := by
  simp only [evalOp, evalOp_ref, h, Option.bind_some, Option.bind_eq_bind]
  erw [evalOp_expandDim, evalOp_ref, h, evalOp_expandDim, evalOp_ref, h]
  simp only [Option.bind_some, Option.bind_eq_bind, Option.pure_def]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp [Tile.cop, Tile.expandDim, TileShape.dropInsertedIndex, Broadcast.leftIndex,
    Broadcast.rightIndex, ComparableDType.ge, lowerTriTile, Tile.select]

theorem invariant (p : Layout) (u : BlockState) (s o : RegionName) (a : Nat → ℝ)
    (hm : ∀ off, u.readMem s off = a off) :
    Holds u p (launch u) s o a 0 (initial p u) := by
  refine ⟨rfl, Nat.zero_le _, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [initial, launch]
  · simp [initial, launch]
  · simp [initial]
  · simp [initial, RealLoop.carry, RealMath.carry]
  · intro off; simpa [initial] using hm off
  · intro t j ha ht; simp at ht
  · intro r off hn; rfl

theorem chunks_cover (p : Layout) (hBT : 0 < p.BT) : p.T ≤ numChunks p * p.BT := by
  unfold numChunks
  have hd := Nat.div_add_mod (p.T + p.BT - 1) p.BT
  have hm := Nat.mod_lt (p.T + p.BT - 1) hBT
  rw [Nat.mul_comm]
  omega

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkCumsumVector.RealInit
