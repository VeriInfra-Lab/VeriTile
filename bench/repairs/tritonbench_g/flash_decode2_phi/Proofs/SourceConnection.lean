import bench.repairs.tritonbench_g.flash_decode2_phi.Kernels

namespace VeriTile.Bench.Repairs.FlashDecode2
open VeriTile.Triton FP.Structural
/-- Proof-only decomposition of the complete canonical source. -/
def prologue (H S D Q BD : Nat) : List ComputeStmt :=
  (phi H S D Q BD).surfaceBody.take 10

def body (masked : Bool) (H S D Q BD : Nat) : List ComputeStmt :=
  match (kernel masked H S D Q BD).surfaceBody.getD 10 (.effectMarker "absent") with
  | ComputeStmt.forRangeInt _ _ _ _ b => b
  | _ => []

def epilogue (masked : Bool) (H S D Q BD : Nat) : List ComputeStmt :=
  (kernel masked H S D Q BD).surfaceBody.drop 11

theorem decomposition (masked : Bool) (H S D Q BD : Nat) :
    (kernel masked H S D Q BD).surfaceBody = prologue H S D Q BD ++
      [.forRangeInt "block_seq_n" (.constInt 0)
        (.ref .int [] "block_n_size") (.constNat 1) (body masked H S D Q BD)] ++
      epilogue masked H S D Q BD := by
  cases masked <;> rfl

end VeriTile.Bench.Repairs.FlashDecode2
