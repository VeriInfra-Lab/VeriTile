import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.RealStep
import VeriTile.Triton.Memory.FlattenR

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.RealAccess
open VeriTile.Triton IndexProgram RealProgram RealValues
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem assign (bounds : RegionBounds)
    (d : TileDType) (sh : TileShape) (name : RegName)
    (e : Op d sh) (v : Tile d sh) (rest : List Stmt) (u : BlockState)
    (safe : e.SafeAtR .triv bounds u) (evaluated : evalOp e u = some v)
    (tailSafe : Stmt.TraceSafeListR .triv bounds rest (u.setReg name d sh v)) :
    Stmt.TraceSafeListR .triv bounds (.assign d sh name e :: rest) u := by
  rw [Stmt.TraceSafeListR]
  refine ⟨by simpa only [Stmt.TraceSafeR] using safe, ?_⟩
  simpa only [stepStmtR_triv, stepStmt, evaluated, Option.bind_some] using tailSafe

theorem load_safe (bounds : RegionBounds) (h : Call) (c : Config)
    (feature row q : Nat) (u : BlockState)
    (hp : u.regs .ptr [c.BT,BS] "p_s" = some (pointer h c feature row q h.input h.inputStrides))
    (hm : u.regs .bool [c.BT,BS] "active" = some (mask h c feature q))
    (bounded : ∀ i : TileIndex [c.BT,BS], Geometry.active h c feature q i.1.val i.2.1.val →
      h.inputStrides.address h.H row (Geometry.time h c q i.1.val)
        (Geometry.column h feature i.2.1.val) < bounds h.input) :
    (RealProgram.loadOp c).SafeAtR .triv bounds u := by
  simpa [RealProgram.loadOp, Op.SafeAtR, Op.SafeAtR.eq_def, MemAccess.SafeAtR,
    MaskOpt.SafeAtR, MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
    MaskOpt.ActiveR, evalOpR_triv, evalOp.eq_def, hp, hm, pointer, mask] using bounded

theorem store_safe (bounds : RegionBounds) (h : Call) (c : Config)
    (feature row q : Nat) (u : BlockState)
    (hp : u.regs .ptr [c.BT,BS] "p_z" = some (pointer h c feature row q h.output h.outputStrides))
    (hm : u.regs .bool [c.BT,BS] "active" = some (mask h c feature q))
    (bounded : ∀ i : TileIndex [c.BT,BS], Geometry.active h c feature q i.1.val i.2.1.val →
      h.outputStrides.address h.H row (Geometry.time h c q i.1.val)
        (Geometry.column h feature i.2.1.val) < bounds h.output) :
    Stmt.TraceSafeR .triv bounds
      (.store .real [c.BT,BS] (.ptr (.ref .ptr [c.BT,BS] "p_z"))
        (.ref .real [c.BT,BS] "b_c") (.mask (.ref .bool [c.BT,BS] "active"))) u := by
  simpa [Stmt.TraceSafeR, Op.SafeAtR, Op.SafeAtR.eq_def, MemAccess.SafeAtR,
    MaskOpt.SafeAtR, MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
    MaskOpt.ActiveR, evalOpR_triv, evalOp.eq_def, hp, hm, pointer, mask] using bounded

#axiomsClean load_safe
#axiomsClean store_safe
end
end VeriTile.Bench.Repairs.ChunkCumsumVector.RealAccess
