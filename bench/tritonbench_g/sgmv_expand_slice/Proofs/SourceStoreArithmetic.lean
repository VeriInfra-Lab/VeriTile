import bench.tritonbench_g.sgmv_expand_slice.Proofs.Geometry
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.SourceStoreArithmetic
set_option linter.unusedSimpArgs false

theorem row_lt (start lane tile length : Nat) :
    ((start : Int)+lane+(tile : Int)*32 < (start : Int)+length) ↔
      start+lane+tile*32 < start+length := by omega

theorem column_lt (lane tile slice columns : Nat) :
    ((lane : Int)+(tile : Int)*32+slice < (slice : Int)+columns) ↔
      lane+tile*32+slice < slice+columns := by omega

theorem output_offset (start rowLane rowTile length columnLane columnTile slice columns
    rowStride columnStride : Nat) :
    ((if (start : Int)+rowLane+(rowTile : Int)*32 < (start : Int)+length then
        ((start : Int)+rowLane+(rowTile : Int)*32)*rowStride else 0) +
      if (columnLane : Int)+(columnTile : Int)*32+slice < (slice : Int)+columns then
        ((columnLane : Int)+(columnTile : Int)*32+slice)*columnStride else (slice : Int)*columnStride).toNat =
    (if start+rowLane+rowTile*32 < start+length then start+rowLane+rowTile*32 else 0)*rowStride +
      (if columnLane+columnTile*32+slice < slice+columns then columnLane+columnTile*32+slice else slice)*columnStride := by
  simp only [row_lt,column_lt]
  split_ifs <;> norm_cast <;> simp only [Nat.zero_mul,Nat.zero_add,Int.toNat_natCast]

#axiomsClean row_lt
#axiomsClean column_lt
#axiomsClean output_offset
end VeriTile.Bench.Repairs.SgmvExpandSlice.SourceStoreArithmetic
