import bench.tritonbench_g.relu_strided_buffer.Proofs.Mathematical_SignedGridExecution

/-!
Actual source control-flow execution, parameterized only by a locally proved
per-tile statement implementation. The source keeps pid, num_tiles0, num_ctas,
the j loop and tile_id recomputation. Only memory and padding observations are
compared: source scratch registers need not equal the flattened tile executor.
The final source module discharges this local interface with signed perTile.
-/
namespace VeriTile.Bench.TritonBenchG.ReluStridedBuffer.SourceForRange
open VeriTile.Triton SignedView SignedTile SignedGrid

def ObsEq (s t : BlockState) : Prop := s.mem = t.mem ∧ s.undef = t.undef

theorem ObsEq.refl (s : BlockState) : ObsEq s s := ⟨rfl, rfl⟩
theorem ObsEq.trans {s t u : BlockState} (h : ObsEq s t) (h' : ObsEq t u) : ObsEq s u :=
  ⟨h.1.trans h'.1, h.2.trans h'.2⟩

theorem setReg_obs (s : BlockState) (name : RegName) (dtype : TileDType)
    (shape : TileShape) (value : Tile dtype shape) : ObsEq (s.setReg name dtype shape value) s :=
  ⟨rfl, rfl⟩

def TileImplementation (input output : View) (n B : Nat) (body : List Stmt) : Prop :=
  ∀ T s, s.regs .nat [] "tile_id" = some (Tile.scalar T) →
    ∃ final, stepStmts body s = some final ∧
      ObsEq final (finalState input output n (T * B) B s) ∧
      final.regs .nat [] "pid" = s.regs .nat [] "pid" ∧
      final.regs .nat [] "num_ctas" = s.regs .nat [] "num_ctas"

def tileIdStmt : Stmt :=
  .assign .nat [] "tile_id"
    (.add .nat .nil (.ref .nat [] "pid")
      (.mul .nat .nil (.ref .nat [] "j") (.ref .nat [] "num_ctas")))

def iterationStarts (P C B cur remaining : Nat) : List Nat :=
  (List.range' cur remaining).map (fun j => (P + j * C) * B)

theorem iterationStarts_zero (P C B cur : Nat) : iterationStarts P C B cur 0 = [] := rfl

theorem iterationStarts_succ (P C B cur remaining : Nat) :
    iterationStarts P C B cur (remaining + 1) =
      (P + cur * C) * B :: iterationStarts P C B (cur + 1) remaining := by
  simp only [iterationStarts, List.range'_succ, List.map_cons]

/-- The source's otherwise unused cdiv assignment is retained in the program. -/
def numTilesStmt (n B : Nat) : Stmt :=
  .assign .nat [] "num_tiles0"
    (.div .nat .nil
      (.sub .nat .nil (.add .nat .nil (.constNat n) (.constNat B)) (.constNat 1))
      (.constNat B))

theorem numTiles_step (n B : Nat) (s : BlockState) :
    stepStmt (numTilesStmt n B) s =
      some (s.setReg "num_tiles0" .nat [] (Tile.scalar ((n + B - 1) / B))) := by
  apply stepStmt_assign_eq_some
  simp only [evalOp_div, evalOp_sub, evalOp_add, evalOp_constNat]
  rfl

theorem singleton_step (stmt : Stmt) (s : BlockState) :
    stepStmts [stmt] s = stepStmt stmt s := by
  simp only [stepStmts]
  cases stepStmt stmt s <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ReluStridedBuffer.SourceForRange
