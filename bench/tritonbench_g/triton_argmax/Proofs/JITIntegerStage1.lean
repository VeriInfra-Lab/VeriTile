import bench.tritonbench_g.triton_argmax.Proofs.JITStage1
import bench.tritonbench_g.triton_argmax.Proofs.TypedIntegerMacro

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITInteger
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
open VeriTile.Bench.TritonBenchG.MatrixReduction
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 3000
set_option linter.unusedSimpArgs false

def sourceType : Formats.IntegerKind → SourceDType
  | .i8 => .i8 | .i16 => .i16 | .i32 => .i32 | .i64 => .i64
  | .u8 => .u8 | .u16 => .u16 | .u32 => .u32 | .u64 => .u64 | .boolean => .boolean

def channel : Formats.IntegerKind → TileDType
  | .i8 | .i16 | .i32 | .i64 => .int
  | .u8 | .u16 | .u32 | .u64 => .nat
  | .boolean => .bool

def cell (kind : Formats.IntegerKind) (word : BitVec kind.bits) : Cell Word :=
  match kind with
  | .i8 | .i16 | .i32 | .i64 => .mk .int word.toInt
  | .u8 | .u16 | .u32 | .u64 => .mk .nat word.toNat
  | .boolean => .mk .bool (decide (word.toInt ≠ 0))

/-- Only literal construction and individual source instructions are bound.
NaN selection and both clamps execute in the quotation. -/
def backend (kind : Formats.IntegerKind) (p : ScalarBinding.IntegerPrimitives kind)
    (rest : Algebra Word) : Algebra Word :=
  { Typed.backend rest with
    literal := fun precision dtype value =>
      if value = 0 then 0
      else if value = (kind.lower : ℝ) then ScalarBinding.lowerBits kind
      else if value = (kind.upper : ℝ) then ScalarBinding.upperBits kind
      else rest.literal precision dtype value
    binary := fun precision dtype op => match op with
      | .max => p.maximum32
      | _ => rest.binary precision dtype op
    minimum := fun _ _ => some p.minimum32
    floatToInt := fun _ _ _ => some (fun value => some (p.truncate value).toInt)
    floatToNat := fun _ _ _ => some (fun value => some (p.truncate value).toNat) }

def written (kind : Formats.IntegerKind) (p : ScalarBinding.IntegerPrimitives kind)
    (s : State Word) (X V : RegionName) (I : Region .int)
    (size block base : Nat) (positive : 0<block) : State Word :=
  let selected := Typed.firstIndex s X size block base positive
  (s.write V (s.pids 0+base) (cell kind
    (Formats.store kind (ScalarBinding.storePrimitives kind p) ((s.mem X selected).read .real)))).write
    (Region.cast I) (s.pids 0+base) (.mk .int (Int.ofNat selected))

theorem source_run (kind : Formats.IntegerKind) (p : ScalarBinding.IntegerPrimitives kind)
    (rest : Algebra Word) (s : State Word) (X V : RegionName) (I : Region .int)
    (size block base : Nat) (index64 : Bool) (positive : 0 < block) :
    ∃ t, exec (backend kind p rest)
      (JITSource.stage1 X V I (sourceType kind) block index64) (JITStage1.arguments s size base) = some t ∧
      t.mem = (written kind p s X V I size block base positive).mem := by
  cases kind <;> cases index64 <;>
    simp [sourceType, JITSource.stage1, JITStage1.arguments, ComputeKernel.surfaceBody,
      FP.Structural.exec, run, step, evalExpr, evalComputeOp, evalOp_unfold,
      ComputeDType.eraseDType, backend, Typed.backend, VeriTile.Bench.TritonBenchG.TritonArgmax.DimBackend.indexed,
      VeriTile.Bench.TritonBenchG.TritonArgmax.DimBackend.normalize,
      TileShape.axisDim, positive, Nat.ne_of_gt positive, TileShape.insertAxisIndex, TileShape.eraseAxis,
      ofFloat, toFloat, bop, numeric, numericLt, numericEq, natLt, TileShape.discreteMaximum,
      TileShape.discreteIntArgMax, TypedReduction.self_equal, TypedReduction.presence,
      TypedReduction.selected, flag, clean, Typed.globalLoaded, Typed.firstIndex,
      written, Option.bind, store, TileShape.allIndices, Region.cast,
      -Nat.cast_add, -Nat.cast_mul, ← Int.natCast_add, ← Int.natCast_mul, collectValues?, cell, Formats.store, ScalarBinding.storePrimitives, Formats.IntegerKind.lower, Formats.IntegerKind.upper, Formats.IntegerKind.exponent, Formats.IntegerKind.gap, Formats.IntegerKind.signed, Formats.IntegerKind.bits, ScalarBinding.lowerBits, ScalarBinding.upperBits]

  all_goals simp only [← apply_ite (fun n : Nat => (n : Int)), ← Int.natCast_add, Int.toNat_natCast, State.write, State.setReg_mem]
  all_goals norm_num
  all_goals rfl





#axiomsClean source_run
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITInteger
