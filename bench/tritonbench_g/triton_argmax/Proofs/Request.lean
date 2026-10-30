import bench.tritonbench_g.triton_argmax.Proofs.Tensor

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.Request
open VeriTile.Bench.Repairs.TritonArgmax Source Memory
noncomputable section

/-- Admitted scratch requests, excluding the rejected other-dtype branch. -/
inductive Kind where
  | floating : RealScratch.FloatingKind → Kind
  | integer : Formats.IntegerKind → Kind

def Kind.format : Kind → MidFormat
  | .floating kind => kind.format
  | .integer kind => Formats.format kind

def Kind.adapter : Kind → RealScratch.Adapter
  | .floating kind => .directCast kind.direct
  | .integer kind => .integerSafe kind

def Kind.isFloating : Kind → Bool
  | .floating _ => Bool.true
  | .integer _ => Bool.false

/-- Only native scalar conversions are parameters. Comparisons and NaN
classification are fixed by the requested stored format in Global.lean. -/
structure Casts where
  floating : (kind : RealScratch.FloatingKind) → Word → MidWord kind.format
  integer : (kind : Formats.IntegerKind) → ScalarBinding.IntegerPrimitives kind

def scratch (casts : Casts) : (kind : Kind) → Scratch kind.format
  | .floating kind => floatingScratch kind (casts.floating kind)
  | .integer kind => integerScratch kind (casts.integer kind)

/-- Literal omitted dtype is FP32. The supplied global dtype is preserved. -/
def select : Checked.DTypeArgument → Option Kind
  | .omitted => some (.floating .fp32)
  | .floating kind => some (.floating kind)
  | .integer kind => some (.integer kind)
  | .invalid _ => none

theorem selected_adapter (argument : Checked.DTypeArgument) (kind : Kind)
    (selected : select argument=some kind) : argument.adapter=some kind.adapter := by
  cases argument <;> simp only [select,Option.some.injEq] at selected
  all_goals try cases selected
  all_goals rfl

theorem format_adapter (kind : Kind) : kind.adapter.format=kind.format := by cases kind <;> rfl

theorem requested_bytes (casts : Casts) (kind : Kind) :
    (scratch casts kind).scalars.storeCast=(scratch casts kind).cast := rfl

#axiomsClean selected_adapter
#axiomsClean format_adapter
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.Request
