import bench.repairs.tritonbench_g.rbe_triton_transform.Kernels
import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy

/-! Task-local revision of the existing typed source and masked scatter proof.
FP32 storage, signed position and stride-correct imaginary neighbours. -/
namespace VeriTile.Bench.Repairs.RbeTritonTransform.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
def execution {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    {t : State α // FP.Structural.exec M (source p) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [source, typed, FP.Structural.exec, ComputeKernel.surfaceBody,
    FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr,
    FP.Structural.evalComputeOp, FP.Structural.evalOp_unfold,
    FP.Structural.evalOffsets, FP.Structural.store, FP.Structural.numeric,
    FP.Structural.bop, FP.Structural.natLt, FP.Structural.numericEq, ComputeDType.eraseDType,
    FP.Structural.ofFloat, FP.Structural.toFloat, FloatDType.numericDType, FloatDType.toTileDType,
    Region.cast, Option.bind, integralFloorDiv, integralMod]
  rfl

def parts (p : Parameters) : Nat := (p.columns + p.blockColumns - 1) / p.blockColumns
abbrev Index (p : Parameters) := TileIndex [p.blockRows, p.blockColumns / 2]

def row {α : Type} (p : Parameters) (s : State α) (i : Index p) : Nat :=
  s.pids 1 / parts p * p.blockRows + i.1.val

def column {α : Type} (p : Parameters) (s : State α) (i : Index p) : Nat :=
  s.pids 1 % parts p * p.blockColumns + i.2.1.val * 2

def inputOffset {α : Type} (p : Parameters) (s : State α) (imag : Bool) (i : Index p) : Nat :=
  s.pids 0 * p.inputBatchStride + p.inputRowStride * row p s i +
    p.inputColumnStride * column p s i + if imag then p.inputColumnStride else 0

def outputOffset {α : Type} (p : Parameters) (s : State α) (imag : Bool) (i : Index p) : Nat :=
  s.pids 0 * p.outputBatchStride + p.outputRowStride * row p s i +
    p.outputColumnStride * column p s i + if imag then p.outputColumnStride else 0

def active {α : Type} (p : Parameters) (s : State α) (imag : Bool) (i : Index p) : Prop :=
  row p s i < p.rows ∧ (if imag then 1 + column p s i else column p s i) < p.columns
instance {α : Type} (p : Parameters) (s : State α) (imag : Bool) :
    DecidablePred (active p s imag) := fun _ => inferInstanceAs (Decidable (_ ∧ _))

/-- Native pow/sin/cos and conversion callbacks remain explicit. This is a
backend-value reference, never Real.sin/Real.cos or an exact-rounding premise. -/
def phase {α : Type} (M : Algebra α) (p : Parameters) (s : State α) (i : Index p) : α :=
  let exponent := M.binary (some .fp32) .real .div
    (M.cast (some .fp32) .real .real (M.fromNat (some .fp32) (column p s i % p.frequencyDimension)))
    (M.literal (some .fp32) .real (p.frequencyDimension : ℝ))
  let power := M.binary (some .fp32) .real .pow (M.literal (some .fp32) .real p.theta) exponent
  M.binary (some .fp32) .real .div
    (M.cast (some .fp32) .real .real
      (M.fromInt (some .fp32) ((i.1.val : Int) + (p.startToken + (s.pids 1 / parts p * p.blockRows : Nat))))) power

def component {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (imag : Bool) (i : Index p) : α :=
  M.cast (some .fp32) .fp32 .real
    (if active p s imag i then (s.mem p.input (inputOffset p s imag i)).read .fp32
     else M.cast none .real .fp32 (M.literal none .real 0.0))

def value {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (imag : Bool) (i : Index p) : α :=
  let re := component M p s Bool.false i
  let im := component M p s Bool.true i
  let c := M.unary (some .fp32) .cos (phase M p s i)
  let sn := M.unary (some .fp32) .sin (phase M p s i)
  M.cast none .real .fp32
    (if imag then M.binary (some .fp32) .real .add
      (M.binary (some .fp32) .real .mul re sn) (M.binary (some .fp32) .real .mul im c)
    else M.binary (some .fp32) .real .sub
      (M.binary (some .fp32) .real .mul re c) (M.binary (some .fp32) .real .mul im sn))

def halfScatter {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (imag : Bool) (initial : State α) : State α :=
  (TileShape.allIndices [p.blockRows, p.blockColumns / 2]).foldl (fun t i =>
    if active p s imag i then t.write p.output (outputOffset p s imag i)
      (.mk .fp32 (value M p s imag i)) else t) initial

def written {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) : State α :=
  halfScatter M p s Bool.true (halfScatter M p s Bool.false s)

theorem execution_memory {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    (execution M p s).val.mem = (written M p s).mem := by
  simp +contextual [execution, written, halfScatter, masked_scatter_memory, value, component, phase,
    active, inputOffset, outputOffset, row, column, parts, FP.Structural.bop,
    Broadcast.leftIndex, Broadcast.rightIndex, Region.cast, integralFloorDiv, integralMod,
    and_assoc, and_left_comm, and_comm]

#axiomsClean execution
#axiomsClean execution_memory
end
end VeriTile.Bench.Repairs.RbeTritonTransform.Source
