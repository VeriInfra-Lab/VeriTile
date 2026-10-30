import bench.repairs.tritonbench_g.rotary_transform.Proofs.Guards

namespace VeriTile.Bench.Repairs.RotaryTransformTileChecked
open VeriTile.Bench.Repairs.RotaryTransform

/-- Exact largest live source matrix: split loads/stores have BK/2 columns;
interleaved pointers, masks, payloads and stores have BK columns. -/
def matrixElements (s : Shape) : Nat :=
  s.blockM * (if s.interleaved then s.blockK else s.blockK/2)

def StaticTiles (s : Shape) : Prop := matrixElements s ≤ 1048576

instance (s : Shape) : Decidable (StaticTiles s) := inferInstanceAs (Decidable (_ ≤ _))

def tileCheck (s : Shape) : Bool := Host.empty s || decide (StaticTiles s)

/-- Existing metadata/input guards plus one conditional source-formation check.
Prepared pointer checks remain the existing separate post-preparation checks. -/
def check (c : Checked.Call) (metadata : Physical.Bytes) : Bool :=
  Checked.check c metadata && tileCheck c.host.shape

theorem check_iff (c : Checked.Call) (metadata : Physical.Bytes) :
    check c metadata=Bool.true ↔ Checked.check c metadata=Bool.true ∧
      (Host.empty c.host.shape=Bool.false → StaticTiles c.host.shape) := by
  simp only [check,tileCheck,Bool.and_eq_true,Bool.or_eq_true,decide_eq_true_eq]
  cases h : Host.empty c.host.shape <;> simp [h]

theorem admitted (c : Checked.Call) (metadata : Physical.Bytes)
    (accepted : check c metadata=Bool.true) : Checked.check c metadata=Bool.true :=
  (check_iff c metadata).mp accepted |>.1

theorem tiles (c : Checked.Call) (metadata : Physical.Bytes)
    (accepted : check c metadata=Bool.true) (launch : Host.empty c.host.shape=Bool.false) :
    StaticTiles c.host.shape := (check_iff c metadata).mp accepted |>.2 launch

theorem empty_preserves (c : Checked.Call) (metadata : Physical.Bytes)
    (empty : Host.empty c.host.shape=Bool.true) : check c metadata=Checked.check c metadata := by
  simp [check,tileCheck,empty]

theorem blockK_min (s : Shape) : 32≤s.blockK :=
  le_trans (le_max_left _ _) (Nat.le_pow_clog (by decide : 1<2) _)

theorem vector_bounds (s : Shape) (tiles : StaticTiles s) :
    s.blockM≤1048576 ∧ s.blockK≤1048576 ∧ s.blockK/2≤1048576 := by
  have minimum := blockK_min s
  rcases blockM_cases s with bm|bm
  all_goals unfold StaticTiles matrixElements at tiles
  all_goals rw [bm] at tiles ⊢
  all_goals cases inter : s.interleaved <;> simp only [inter,Bool.false_eq_true,if_false,if_true] at tiles
  all_goals omega

#axiomsClean check_iff
#axiomsClean empty_preserves
#axiomsClean vector_bounds
#specNonCircular check avoiding [source,Native.selected]
end VeriTile.Bench.Repairs.RotaryTransformTileChecked
