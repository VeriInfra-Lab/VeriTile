import Mathlib.Data.Nat.Basic
import VeriTile.Meta.StatementAudit

/-! Source index bounds, independent of Tensor wrappers and launch execution.
This reuses the bound formerly bundled with vector-addition host views. -/
namespace VeriTile.Triton.LaunchIndex

/-- Every launched lane of a 1024-element block, including the masked tail,
has a nonnegative signed-i32 index when the source extent is below 2^31. -/
theorem launched_index_i32 (n pid lane : Nat) (hn : n < 2^31)
    (hp : pid < (n+1023)/1024) (hl : lane < 1024) : pid*1024+lane < 2^31 := by
  have hcount : (n+1023)/1024 ≤ 2^21 := by omega
  have hpid : pid < 2^21 := by omega
  omega

#axiomsClean launched_index_i32
end VeriTile.Triton.LaunchIndex
