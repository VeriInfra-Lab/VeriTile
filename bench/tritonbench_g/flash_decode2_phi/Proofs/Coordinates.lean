import bench.tritonbench_g.flash_decode2_phi.Proofs.Host

namespace VeriTile.Bench.Repairs.FlashDecode2
open VeriTile.Triton FP.Structural
theorem row_base (H stride row : Nat) : row/H*(H*stride)+row%H*stride=row*stride := by
  have h := Nat.div_add_mod row H
  nlinarith

theorem row_log_offset {α : Type} (s : State α) (B H S row j : Nat) :
    eoff (Grid.setup B H row s) H S j=row*S+j := by
  dsimp [eoff,Grid.setup]
  rw [row_base]

theorem row_mid_offset {α : Type} (s : State α) (B H S D row j : Nat) (k : TileIndex [BD]) :
    voff (Grid.setup B H row s) H S D j k=row*S*D+j*D+lane k := by
  dsimp [voff,Grid.setup]
  rw [Nat.mul_assoc H S D,row_base]
  ring

end VeriTile.Bench.Repairs.FlashDecode2
