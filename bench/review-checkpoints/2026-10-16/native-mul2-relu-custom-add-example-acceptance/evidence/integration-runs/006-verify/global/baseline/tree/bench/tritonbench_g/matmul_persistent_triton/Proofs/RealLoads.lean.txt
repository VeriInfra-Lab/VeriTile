import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealLoadEvals
import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealInput

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealLoads
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

/-- Sequential composition of all five actual address/load statements, using
individual operational evaluations rather than an assumed whole-body step. -/
theorem executes (p : Parameters) (s : BlockState)
    (row : Fin p.BM → Nat) (column : Fin p.BN → Nat) (ki : Nat)
    (hk : s.regs .int [] "ki" = some (Tile.scalar (ki : Int)))
    (hm : s.regs .int [p.BM] "offs_am" =
      some (Tile.vec fun i => (RealTiles.clamp p.M (row i) : Int)))
    (hn : s.regs .int [p.BN] "offs_bn" =
      some (Tile.vec fun j => (RealTiles.clamp p.N (column j) : Int)))
    (hl : s.regs .nat [p.BK] "offs_k_for_mask" = some (Tile.vec Fin.val)) :
    stepStmts (RealProgram.loads p) s = some (result p s row column ki) := by
  let t1 := s.setReg "offs_k" .int [p.BK] (Tile.vec fun k => (ki*p.BK+k.val : Nat))
  let t2 := t1.setReg "a_ptrs" .ptr [p.BM,p.BK] (aAddresses p row ki)
  let t3 := t2.setReg "b_ptrs" .ptr [p.BK,p.BN] (bAddresses p column ki)
  let t4 := t3.setReg "a" .real [p.BM,p.BK] (RealTiles.aValues p (memoryInputs p s) row ki)
  have h1 := k_evaluates p s ki hk
  have h2 : evalOp (aPointers p) t1 = some (aAddresses p row ki) :=
    a_pointer_evaluates p t1 row ki (by simp [t1,hm]) (by simp [t1])
  have h3 : evalOp (bPointers p) t2 = some (bAddresses p column ki) :=
    b_pointer_evaluates p t2 column ki (by simp [t2,t1,hn]) (by simp [t2,t1])
  have h4 : evalOp (aLoad p) t3 = some (RealTiles.aValues p (memoryInputs p s) row ki) :=
    a_evaluates p t3 row ki (by simp [t3,t2]) (by simp [t3,t2,t1,hk]) (by simp [t3,t2,t1,hl])
  have h5 : evalOp (bLoad p) t4 = some (RealTiles.bValues p (memoryInputs p s) column ki) :=
    b_evaluates p t4 column ki (by simp [t4,t3])
      (by simp [t4,t3,t2,t1,hk]) (by simp [t4,t3,t2,t1,hl])
  dsimp only [t1] at h2
  dsimp only [t2,t1] at h3
  dsimp only [t3,t2,t1] at h4
  dsimp only [t4,t3,t2,t1] at h5
  rw [matches_source]
  simp only [code,stepStmts,stepStmt,h1,h2,h3,h4,h5,bind,Option.bind_some]
  rfl

/-- Independent original-array data, as opposed to prepared registers, supplies
the actual source A load values. -/
theorem a_readback (p : Parameters) (s : BlockState) (xs : RealMath.Inputs)
    (row : Fin p.BM → Nat) (column : Fin p.BN → Nat) (ki : Nat)
    (h : RealInput.Represents p xs s) (hM : 0 < p.M) :
    (result p s row column ki).regs .real [p.BM,p.BK] "a" =
      some (RealTiles.aValues p xs row ki) := by
  simp only [result,BlockState.setReg_ne_name,BlockState.setReg_same,ne_eq,String.reduceEq,not_false_eq_true]
  exact congrArg some (RealInput.a_values p xs s h hM row ki)

theorem b_readback (p : Parameters) (s : BlockState) (xs : RealMath.Inputs)
    (row : Fin p.BM → Nat) (column : Fin p.BN → Nat) (ki : Nat)
    (h : RealInput.Represents p xs s) (hN : 0 < p.N) :
    (result p s row column ki).regs .real [p.BK,p.BN] "b" =
      some (RealTiles.bValues p xs column ki) := by
  simp only [result,BlockState.setReg_ne_name,BlockState.setReg_same,ne_eq,String.reduceEq,not_false_eq_true]
  exact congrArg some (RealInput.b_values p xs s h hN column ki)

theorem register_frame (p : Parameters) (s : BlockState)
    (row : Fin p.BM → Nat) (column : Fin p.BN → Nat) (ki : Nat)
    (dtype : TileDType) (shape : TileShape) (name : RegName)
    (hn : name ∉ ["offs_k","a_ptrs","b_ptrs","a","b"]) :
    (result p s row column ki).regs dtype shape name = s.regs dtype shape name := by
  simp only [List.mem_cons, List.not_mem_nil, or_false, not_or] at hn
  simp [result,BlockState.setReg,hn.1,hn.2.1,hn.2.2.1,hn.2.2.2.1,hn.2.2.2.2]

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealLoads
