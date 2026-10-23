import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Contract

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast
noncomputable section

/-- The unchanged split-K zero hook establishes the typed prepared state.
This is a boundary lemma; the kernel-only entry does not execute the hook. -/
theorem zero_hook_prepares (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers)
    (m : Memory F H) : Zeroed a s c b (if c.S=1 then m else reset a s b m) := by
  intro split cell
  have bounded : cell.1.val*s.N+cell.2.val<s.M*s.N := by
    have hm:=Nat.mul_le_mul_right s.N cell.1.isLt
    simp only [Nat.succ_mul] at hm
    have hn:=cell.2.isLt
    omega
  simp [split,reset,outputAddress,
    show b.outOffset≤b.outOffset+cell.1.val*s.N+cell.2.val by omega,
    show b.outOffset+cell.1.val*s.N+cell.2.val<b.outOffset+s.M*s.N by omega]

#axiomsClean zero_hook_prepares
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast
