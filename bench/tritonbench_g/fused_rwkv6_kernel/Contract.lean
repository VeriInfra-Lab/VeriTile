import bench.tritonbench_g.fused_rwkv6_kernel.Proofs.Checked

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked
open FusedRwkv6Kernel
abbrev Shape := FusedRwkv6Kernel.Shape
abbrev Inputs := Checked.Inputs
abbrev PythonNumber := Checked.PythonNumber

/-- The original wrapper enters the JIT only on this branch. -/
def Launches (c : Shape) : Prop := c.T ≠ 0 ∨ c.final = true
instance (c : Shape) : Decidable (Launches c) := by unfold Launches; infer_instance

/-- Exact raw Python integer specialization interval. Float payloads, including
Inf and NaN, have no additional restriction. The input is never converted. -/
def ScalarABI : PythonNumber → Prop
  | .integer n => -(2^63 : Int) ≤ n ∧ n < 2^64
  | .float _ => True
instance (x : PythonNumber) : Decidable (ScalarABI x) := by
  cases x <;> unfold ScalarABI <;> infer_instance

def InputContract (c : Shape) (x : Inputs) (scale : PythonNumber) : Prop :=
  Checked.Guards c x ∧ (Launches c → ScalarABI scale)
instance (c : Shape) (x : Inputs) (scale : PythonNumber) :
    Decidable (InputContract c x scale) := by unfold InputContract; infer_instance

/-- The old checks plus the one branch-local candidate assertion. -/
def check (c : Shape) (x : Inputs) (scale : PythonNumber) : Bool :=
  Checked.admit c x && (if decide (Launches c) then decide (ScalarABI scale) else true)

theorem check_iff (c : Shape) (x : Inputs) (scale : PythonNumber) :
    check c x scale = true ↔ InputContract c x scale := by
  by_cases live : Launches c <;> simp [check,InputContract,live,Checked.admit_iff]

theorem base_checked (c : Shape) (x : Inputs) (scale : PythonNumber)
    (h : InputContract c x scale) : Checked.admit c x = true :=
  (Checked.admit_iff c x).mpr h.1

theorem no_launch_unchanged (c : Shape) (x : Inputs) (scale : PythonNumber)
    (empty : c.T=0) (disabled : c.final=false) :
    InputContract c x scale ↔ Checked.Guards c x := by
  simp [InputContract,Launches,empty,disabled]

theorem float_domain_unchanged (c : Shape) (x : Inputs) (bits : BitVec 64) :
    InputContract c x (.float bits) ↔ Checked.Guards c x := by
  simp [InputContract,ScalarABI]

theorem live_integer_range (c : Shape) (x : Inputs) (n : Int)
    (h : InputContract c x (.integer n)) (live : Launches c) :
    -(2^63 : Int) ≤ n ∧ n < 2^64 := h.2 live

#axiomsClean check_iff
#axiomsClean no_launch_unchanged
#axiomsClean float_domain_unchanged
#axiomsClean live_integer_range
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked
