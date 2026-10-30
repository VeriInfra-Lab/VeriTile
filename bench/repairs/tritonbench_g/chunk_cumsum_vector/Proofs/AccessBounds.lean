import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.Geometry

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.AccessBounds
open VeriTile.Triton

/-- Capacity of exactly the active logical input/output columns of one program. -/
def Bounds (h : Call) (p : Geometry.Program h) (bounds : RegionBounds) : Prop :=
  ∀ (time : Nat) (j : Fin BS), time < h.T → p.feature*BS+j.val < h.S →
    h.inputStrides.address h.H p.row time (p.feature*BS+j.val) < bounds h.input ∧
    h.outputStrides.address h.H p.row time (p.feature*BS+j.val) < bounds h.output

theorem chunk_bounds (h : Call) (c : Config) (positive : 0 < c.BT)
    (p : Geometry.Program h) (bounds : RegionBounds) (bounded : Bounds h p bounds)
    (q : Nat) (hq : q < Geometry.chunks h c) (i : TileIndex [c.BT,BS])
    (active : Geometry.active h c p.feature q i.1.val i.2.1.val) :
    h.inputStrides.address h.H p.row (Geometry.time h c q i.1.val)
      (Geometry.column h p.feature i.2.1.val) < bounds h.input ∧
    h.outputStrides.address h.H p.row (Geometry.time h c q i.1.val)
      (Geometry.column h p.feature i.2.1.val) < bounds h.output := by
  have coordinates := Geometry.active_coordinates h c p q i.1.val i.2.1.val active
  have within := (Geometry.active_iff h c p q i.1.val i.2.1.val
    (Geometry.chunk_start h c positive q hq)).mp active
  simpa only [coordinates.1, coordinates.2] using bounded _ i.2.1 within.1 within.2

end VeriTile.Bench.Repairs.ChunkCumsumVector.AccessBounds
