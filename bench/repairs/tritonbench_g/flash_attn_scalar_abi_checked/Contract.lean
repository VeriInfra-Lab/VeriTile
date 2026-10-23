import bench.repairs.tritonbench_g.flash_attn.Contract

namespace VeriTile.Bench.Repairs.FlashAttnScalarABIChecked
open FlashAttn Checked

/-- Exactly the additional raw-integer check in the actual launch branch.
Every admitted shape is positive, so every successful call enters that branch.
Float payloads retain the original wrapper's finite/FP32-range checks. -/
def scalarBindChecks : RawScalar→Bool
  | .integer n=>decide (-(2^63 : Int)≤n ∧ n<2^64)
  | .float64 _=>Bool.true

def inputChecks (q k v : Tensor) (qo ko vo : SortedAxes) (scale : ScalarABI) : Bool :=
  Checked.inputChecks q k v qo ko vo scale && scalarBindChecks scale.raw

theorem check_iff (q k v : Tensor) (qo ko vo : SortedAxes) (scale : ScalarABI) :
    inputChecks q k v qo ko vo scale=Bool.true ↔
      Checked.inputChecks q k v qo ko vo scale=Bool.true ∧ scalarBindChecks scale.raw=Bool.true := by
  simp [inputChecks]

theorem original_checks (q k v : Tensor) (qo ko vo : SortedAxes) (scale : ScalarABI)
    (h : inputChecks q k v qo ko vo scale=Bool.true) :
    Checked.inputChecks q k v qo ko vo scale=Bool.true := (check_iff _ _ _ _ _ _ _).mp h |>.1

theorem integer_range (scale : ScalarABI) (n : Int) (raw : scale.raw=.integer n)
    (h : scalarBindChecks scale.raw=Bool.true) : -(2^63 : Int)≤n ∧ n<2^64 := by
  simpa [scalarBindChecks,raw] using h

theorem float_domain_unchanged (q k v : Tensor) (qo ko vo : SortedAxes) (scale : ScalarABI)
    (bits : BitVec 64) (raw : scale.raw=.float64 bits) :
    inputChecks q k v qo ko vo scale=Checked.inputChecks q k v qo ko vo scale := by
  simp [inputChecks,scalarBindChecks,raw]

#axiomsClean check_iff
#axiomsClean original_checks
#axiomsClean integer_range
#axiomsClean float_domain_unchanged
end VeriTile.Bench.Repairs.FlashAttnScalarABIChecked
