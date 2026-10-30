import VeriTile.Triton.Float.Structural
namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked
open VeriTile.Triton FP.Structural
noncomputable section
/-- Bind the actual typed FP32 sum to the selected ordered FP32 reduction
primitive. Only the representation of the floating payload changes: the
axis, keep flag, complete input layout and order are unchanged. Other typed
reductions retain the backend's success/failure behavior. No reduction law,
cast identity or whole-program execution is assumed. -/
def sourceBackend {α : Type} (M : Algebra α) : Algebra α :=
  { M with reduceSumTyped := fun p dtype {shape} axis keep xs =>
      match dtype with
      | .fp32 => some (M.reduceSum (some .fp32) axis keep xs)
      | d => M.reduceSumTyped p d axis keep xs }
end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked
