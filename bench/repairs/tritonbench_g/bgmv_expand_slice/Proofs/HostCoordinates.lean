import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.ViewGuard

namespace VeriTile.Bench.Repairs.BgmvExpandSlice.HostCoordinates
open ReturnedStorage

def program (h : Call) (view : View) (indices : Nat → Int) (i : Coord view)
    (active : Active h view indices i) : Geometry.Program h where
  batch := i.1.val
  row := i.2.val-h.slice
  batchInside := active.1
  rowInside := by have := active.2.1; have := active.2.2.1; omega

theorem original_column (h : Call) (view : View) (indices : Nat → Int) (i : Coord view)
    (active : Active h view indices i) : h.slice+(program h view indices i active).row=i.2.val := by
  have := active.2.1
  simp only [program]
  omega

end VeriTile.Bench.Repairs.BgmvExpandSlice.HostCoordinates
