import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Proofs.Geometry

namespace VeriTile.Bench.Repairs.ChunkCumsumKernel.Layout
open VeriTile.Triton
noncomputable section

structure Shape where
  B : Nat
  H : Nat
  T : Nat

abbrev Coord (c : Shape) := Fin c.B × Fin c.H × Fin c.T

def row (c : Shape) (i : Coord c) : Nat := i.1.val*c.H+i.2.1.val

def offset (c : Shape) (i : Coord c) : Nat := row c i*c.T+i.2.2.val

theorem row_bound (c : Shape) (i : Coord c) : row c i<c.B*c.H := by
  have hb := i.1.isLt
  have hh := i.2.1.isLt
  dsimp [row]
  exact lt_of_lt_of_le (Nat.add_lt_add_left hh _) (by
    simpa [Nat.succ_eq_add_one,Nat.add_mul] using
      Nat.mul_le_mul_right c.H (Nat.succ_le_of_lt hb))

#axiomsClean row_bound
end
end VeriTile.Bench.Repairs.ChunkCumsumKernel.Layout
