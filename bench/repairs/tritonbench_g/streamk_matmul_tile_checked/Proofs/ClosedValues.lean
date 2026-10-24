import bench.repairs.tritonbench_g.streamk_matmul_tile_checked.ClosedSource
import bench.repairs.tritonbench_g.streamk_matmul_tile_checked.Proofs.InstructionMap

namespace VeriTile.Bench.Repairs.StreamK.TileChecked.ClosedSource
open Whole001 Source VeriTile.Triton
noncomputable section
set_option maxHeartbeats 4000000

def viewA (p : Parameters) (s : Concurrent.State α) : A p → α := fun i=>s.input 0 i.val
def viewB (p : Parameters) (s : Concurrent.State α) : B p → α := fun i=>s.input 1 i.val

/-- Evaluating one closed symbolic primitive is definitionally the same
local instruction. This is an interpreter theorem, not a numerical premise. -/
theorem interpreter_map (ops : Instructions α) (s : Concurrent.State α) :
    Reader.InstructionMap symbolic ops (fun expression=>expression.eval (backend ops) s) :=
  ⟨rfl,fun _ _ _=>rfl,fun _=>rfl,fun _ _=>rfl⟩

theorem dotValue_eval (p : Parameters) (tile : Reader.Tile) (ops : Instructions α)
    (s : Concurrent.State α) (env : Reader.Env) (lane : Lane p) :
    (Reader.dotValue p tile symbolic (inputA p) (inputB p) env lane).eval (backend ops) s=
      Reader.dotValue p tile ops (viewA p s) (viewB p s) env lane :=
  Reader.dotValue_map p tile symbolic ops _ (interpreter_map ops s) _ _ env lane

theorem chunk_eval (p : Parameters) (ops : Instructions α) (s : Concurrent.State α)
    (tile lo hi : Nat) (lane : Lane p) :
    (chunkValue p symbolic (inputA p) (inputB p) tile lo hi lane).eval (backend ops) s=
      chunkValue p ops (viewA p s) (viewB p s) tile lo hi lane :=
  Reader.chunk_map p symbolic ops _ (interpreter_map ops s) _ _ tile lo hi lane

#axiomsClean interpreter_map
#axiomsClean dotValue_eval
#axiomsClean chunk_eval
end
end VeriTile.Bench.Repairs.StreamK.TileChecked.ClosedSource
