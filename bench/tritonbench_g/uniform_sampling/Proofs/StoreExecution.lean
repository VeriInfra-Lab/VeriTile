import bench.tritonbench_g.uniform_sampling.Proofs.Algorithm
import bench.tritonbench_g.uniform_sampling.Proofs.Memory
import VeriTile.Triton.Semantics.Eval

/-! Actual masked store semantics connected to the independent four-stream
memory formula. Each source store retains its complete B-lane iteration and
its own offset < N mask. Incoming conditions are register contents only.
-/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.StoreExecution
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000

theorem store_step {B : Nat} (y : RegionName) (N : Nat)
    (wordName offsetName : RegName) (s : BlockState)
    (values : TileIndex [B] → ℝ) (offsets : TileIndex [B] → Nat)
    (hv : s.regs .real [B] wordName = some ⟨fun i => some (values i)⟩)
    (ho : s.regs .nat [B] offsetName = some ⟨offsets⟩) :
    stepStmt (Algorithm.storeStmt y N B wordName offsetName) s =
      some ((TileShape.allIndices [B]).foldl (fun acc i =>
        if offsets i < N then acc.writeMem y (offsets i) (values i) else acc) s) := by
  simp [-Tile.scalar_eta, Algorithm.storeStmt, stepStmt, evalOp.eq_def, hv, ho,
    Tile.cop, Tile.scalar, ComparableDType.lt]

theorem store_field {B : Nat} (pid : Nat) (y : RegionName) (N : Nat)
    (seed offset : Int) (lower upper : ℝ) (f : Fin 4)
    (wordName offsetName : RegName) (s : BlockState)
    (hv : s.regs .real [B] wordName = some ⟨fun i =>
      some (ScalarSpec.value seed offset (SourceDomain.counterIndex pid B i.1.val)
        (Formula.field f.val) lower upper)⟩)
    (ho : s.regs .nat [B] offsetName =
      some ⟨fun i => SourceDomain.outputIndex pid B f i.1⟩) :
    stepStmt (Algorithm.storeStmt y N B wordName offsetName) s =
      some (Memory.storeField pid s y N B seed offset lower upper f) := by
  exact store_step y N wordName offsetName s _ _ hv ho

#axiomsClean store_step
#axiomsClean store_field

end
end VeriTile.Bench.TritonBenchG.UniformSampling.StoreExecution
