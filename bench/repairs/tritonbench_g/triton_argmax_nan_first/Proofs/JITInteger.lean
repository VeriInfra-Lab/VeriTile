import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.JITIntegerStage1
namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITInteger
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
open VeriTile.Bench.TritonBenchG.MatrixReduction
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 3000
set_option linter.unusedSimpArgs false

def read (kind : Formats.IntegerKind) (value : Cell Word) : Int :=
  match kind with
  | .i8 | .i16 | .i32 | .i64 => value.read .int
  | .u8 | .u16 | .u32 | .u64 => (value.read .nat : Int)
  | .boolean => if value.read .bool then 1 else 0

def secondIndex (kind : Formats.IntegerKind) (s : State Word) (V : RegionName)
    (size block : Nat) : Nat :=
  TileShape.leftmostIntArgMax (fun lane : Fin block =>
    if lane.val<size then read kind (s.mem V lane.val) else kind.padding)

theorem second_run (kind : Formats.IntegerKind) (rest : Algebra Word) (s : State Word)
    (V : RegionName) (I O : Region .int) (size block : Nat) :
    ∃ t, FP.Structural.exec rest (JITSource.stage2 V I O (sourceType kind) block)
      (s.setReg "mid_size" .int [] (fun _ => (size : Int)))=some t ∧
      t.mem=(s.write (Region.cast O) 0 (.mk .int
        ((s.mem (Region.cast I) (secondIndex kind s V size block)).read .int))).mem := by
  cases kind <;>
    simp [sourceType, JITSource.stage2, ComputeKernel.surfaceBody, FP.Structural.exec,
      run, step, evalExpr, evalComputeOp, evalOp_unfold,
      TileShape.axisDim, TileShape.insertAxisIndex, TileShape.eraseAxis,
      ofFloat, toFloat, bop, numeric, numericLt, numericEq, natLt, TileShape.discreteIntArgMax,
      secondIndex, read, Option.bind, store, TileShape.allIndices, Region.cast,
      Formats.IntegerKind.padding, Formats.IntegerKind.lower, Formats.IntegerKind.exponent,
      Formats.IntegerKind.signed, Formats.IntegerKind.bits, Op.boolToInt,
      -Nat.cast_add, -Nat.cast_mul, ← Int.natCast_add, ← Int.natCast_mul]
  all_goals try rfl
  have lanes : (fun lane : Fin block =>
      if lane.val < size ∧ (s.mem V lane.val).read .bool = Bool.true then (1 : Int) else 0) =
      (fun lane : Fin block => if lane.val < size then
        (if (s.mem V lane.val).read .bool then (1 : Int) else 0) else 0) := by
    funext lane
    by_cases h : lane.val < size <;> simp [h]
  let finish (winner : Nat) :=
    (s.write (Region.cast O) 0 (.mk .int ((s.mem (Region.cast I) winner).read .int))).mem
  exact congrArg finish (congrArg TileShape.leftmostIntArgMax lanes)

theorem read_cell (kind : Formats.IntegerKind) (word : BitVec kind.bits) :
    read kind (cell kind word) = Formats.decodeStored kind (word.cast (Formats.format_width kind)) := by
  cases kind <;> simp [read, cell, Formats.decodeStored, Formats.IntegerKind.signed,
    Formats.format, Formats.IntegerKind.bits]
  change (if word.toInt = (0 : Int) then (0 : Int) else 1) =
    if word.toNat = (0 : Nat) then (0 : Int) else 1
  change BitVec 8 at word
  have zero : word.toInt = (0 : Int) ↔ word.toNat = (0 : Nat) := by
    constructor
    · intro h
      have eq : word = 0 := BitVec.eq_of_toInt_eq (by simpa using h)
      simp [eq]
    · intro h
      have eq : word = 0 := BitVec.eq_of_toNat_eq (by simpa using h)
      simp [eq]
  exact if_congr zero rfl rfl

#axiomsClean second_run
#axiomsClean read_cell

end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITInteger
