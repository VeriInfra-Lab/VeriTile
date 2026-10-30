import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.SafetyLoop

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.SourceSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite Safety FusedRwkv6Kernel Source
noncomputable section
variable {α : Type} [Inhabited α]

/-- An absent Python H0/HT argument has no accessible backing cells. The
source flags, already supplied by the caller, determine which channels exist. -/
def optionalBounds (c : Shape) (bounds : RegionName → Nat) (r : RegionName) : Nat :=
  if (r="h0" ∧ c.initial=Bool.false) ∨ (r="ht" ∧ c.final=Bool.false) then 0 else bounds r

theorem optional_bounds (c : Shape) (s : State α) (bounds : RegionName → Nat)
    (hb : Bounds c s bounds) : Bounds c s (optionalBounds c bounds) := by
  constructor
  · intro j hj
    simpa [optionalBounds] using hb.bias j hj
  · intro hi l j hl hj
    simpa [optionalBounds,hi] using hb.seed hi l j hl hj
  · intro n hn j hj r hr
    simp only [List.mem_cons,List.not_mem_nil,or_false] at hr
    rcases hr with rfl|rfl|rfl <;>
      simpa [optionalBounds] using hb.key n hn j hj _ (by simp)
  · intro n hn l hl
    simpa [optionalBounds] using hb.value n hn l hl
  · intro n hn l hl
    simpa [optionalBounds] using hb.partialOutput n hn l hl
  · intro hf l j hl hj
    simpa [optionalBounds,hf] using hb.terminal hf l j hl hj

/-- Actual trace safety with zero capacity for each absent optional argument.
Thus neither H0 nor HT needs a fabricated accessible pointer when its Python
argument is None. Execution and output/frame proofs remain separate. -/
theorem absent_channels_safe (M : Algebra α) (c : Shape) (s : State α) (scale : α)
    (scalar : s.regs .real [] "scale"=some (fun _ => scale))
    (bounds : RegionName → Nat) (hb : Bounds c s bounds) :
    TraceSafe (engine M) (optionalBounds c bounds) (kernel c) s :=
  source_safe M c s scale scalar _ (optional_bounds c s bounds hb)

@[simp] theorem absent_initial_capacity (c : Shape) (bounds : RegionName → Nat)
    (h : c.initial=Bool.false) : optionalBounds c bounds "h0"=0 := by
  simp [optionalBounds,h]

@[simp] theorem absent_final_capacity (c : Shape) (bounds : RegionName → Nat)
    (h : c.final=Bool.false) : optionalBounds c bounds "ht"=0 := by
  simp [optionalBounds,h]

#axiomsClean optional_bounds
#axiomsClean absent_channels_safe
end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.SourceSafety
