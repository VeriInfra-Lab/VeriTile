import VeriTile.Triton.Float.ScalarOps
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic
namespace VeriTile.Bench.Repairs.LightningAttention.NativeMask
open VeriTile Triton FP

/-- The source's intra-block mask converts only arange(0,64), not global
sequence indices. This finite software-profile fact covers every source pair. -/
def lane (i : Fin 64) : FP.Value .fp32 := FP.round .fp32 .ieeeValue (Rat.ofInt i.val)
def difference (i j : Fin 64) : FP.Value .fp32 := FP.sub .ieeeValue (lane i) (lane j)

set_option maxRecDepth 100000 in
set_option maxHeartbeats 10000000 in
theorem lane_exact : ∀ i : Fin 64, (lane i).toRat? = some (Rat.ofInt i.val) := by decide +kernel

set_option maxRecDepth 100000 in
set_option maxHeartbeats 20000000 in
theorem difference_exact : ∀ i j : Fin 64,
    (difference i j).toRat? = some (Rat.ofInt (i.val : Int) - Rat.ofInt (j.val : Int)) := by decide +kernel

set_option maxRecDepth 100000 in
set_option maxHeartbeats 20000000 in
theorem lower_mask : ∀ i j : Fin 64,
    (!FP.lt .ieeeValue (difference i j) (FP.Value.zero .fp32)) = decide (j ≤ i) := by decide +kernel

set_option maxRecDepth 100000 in
set_option maxHeartbeats 20000000 in
theorem upper_mask : ∀ i j : Fin 64,
    (!FP.lt .ieeeValue (FP.Value.zero .fp32) (difference i j)) = decide (i ≤ j) := by decide +kernel

/- Software exactness is a finite model theorem, not a hardware claim.
Actual compiler lowering of these fixed mask constants to the corresponding
integer comparison is part of the separately audited source binding. -/
#axiomsClean lane_exact
#axiomsClean difference_exact
#axiomsClean lower_mask
#axiomsClean upper_mask
end VeriTile.Bench.Repairs.LightningAttention.NativeMask
