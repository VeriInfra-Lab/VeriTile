import bench.tritonbench_g.flash_decode2_phi.Sources.Kernels

namespace VeriTile.Bench.Repairs.FlashDecode2
open VeriTile.Triton FP.Structural
theorem decomposition (masked : Bool) (H S D Q BD : Nat) :
    (kernel masked H S D Q BD).surfaceBody = prologue H S D Q BD ++
      [.forRangeDyn "block_seq_n" (.constNat 0)
        (.ref .nat [] "block_n_size") (.constNat 1) (body masked H S D Q BD)] ++
      epilogue masked H S D Q BD := by
  cases masked <;> rfl

end VeriTile.Bench.Repairs.FlashDecode2
