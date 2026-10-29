import bench.tritonbench_g.relu_triton_kernel.Proofs.ActualIndices
import bench.tritonbench_g.relu_triton_kernel.Proofs.FP64Source

namespace VeriTile.Bench.TritonBenchG.ReluTritonKernel.Actual
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

inductive Storage where
  | integer (encoding : IntegerWords.Encoding)
  | fp16 | bf16 | fp32 | fp64
  deriving DecidableEq, Repr

def Storage.bytes : Storage → Nat
  | .integer e => FiniteAddress.elementBytes e
  | .fp16 | .bf16 => 2
  | .fp32 => 4
  | .fp64 => 8

abbrev Word (d : Storage) := BitVec (d.bytes*8)

def readWord (d : Storage) (memory : ByteMemory.Memory) (address : Nat) : Word d :=
  ByteMemory.loadLE d.bytes memory address

structure Numerics where
  shared : Algebra (BitVec 32)
  cmp16 : BitVec 32 → BitVec 32 → Bool
  cmp32 : BitVec 32 → BitVec 32 → Bool
  wide : FP64Source.Numerics

/-- Native +0 has the all-zero encoding in each admitted floating format.
The other inherited numerical operations retain their explicit ports. -/
def zeroBackend (b : Numerics) : Algebra (BitVec 32) :=
  {b.shared with literal := fun loc d q => if q=0 then 0 else b.shared.literal loc d q}

abbrev Stage : Storage → Type
  | .fp64 => BitVec 64
  | .integer _ => BitVec 32
  | .fp16 | .bf16 | .fp32 => BitVec 32

/-- The source comparison/select stage executes before the pid-zero gate. -/
def selectSource (b : Numerics) : (d : Storage) → Word d → Stage d
  | .integer (.signed w),v =>
      let word := v.cast (FiniteAddress.element_bytes_width (.signed w))
      if IntegerWords.signedComparison w word then b.shared.fromInt none word.toInt else 0
  | .integer (.unsigned w),v =>
      let word := v.cast (FiniteAddress.element_bytes_width (.unsigned w))
      if IntegerWords.unsignedComparison w word then b.shared.fromNat none word.toNat else 0
  | .integer .boolean,v =>
      let word := v.cast (FiniteAddress.element_bytes_width .boolean)
      if IntegerWords.booleanComparison word then b.shared.fromNat none (IntegerWords.boolBits word).toNat else 0
  | .fp16,v => if b.cmp16 0 (v.setWidth 32) then v.setWidth 32 else 0
  | .bf16,v => if b.cmp32 0 (b.shared.cast none .bf16 .fp32 (v.setWidth 32)) then v.setWidth 32 else 0
  | .fp32,v => if b.cmp32 0 v then v else 0
  | .fp64,v => FP64Source.select64 (FP64Source.orderedGeZero64 v) v 0

/-- Only the gated store executes the destination fp32 conversion. -/
def castStore (b : Numerics) : (d : Storage) → Stage d → BitVec 32
  | .fp16,v => b.shared.cast none .fp16 .fp32 v
  | .bf16,v => b.shared.cast none .bf16 .fp32 v
  | .fp64,v => b.wide.narrow64to32 v
  | .integer _,v => v
  | .fp32,v => v

def scalarSource (b : Numerics) (d : Storage) (v : Word d) : BitVec 32 :=
  castStore b d (selectSource b d v)

/-- Independent existing finite-word/reference value families. -/
def scalarFormula (b : Numerics) : (d : Storage) → Word d → BitVec 32
  | .integer e,v => IntegerWords.wordValue (zeroBackend b) e (v.cast (FiniteAddress.element_bytes_width e))
  | .fp16,v => TypedValue.resultValue (zeroBackend b) b.cmp16 b.cmp32 .fp16 (.mk .fp16 (v.setWidth 32))
  | .bf16,v => TypedValue.resultValue (zeroBackend b) b.cmp16 b.cmp32 .bf16 (.mk .bf16 (v.setWidth 32))
  | .fp32,v => TypedValue.resultValue (zeroBackend b) b.cmp16 b.cmp32 .fp32 (.mk .fp32 v)
  | .fp64,v => FP64Source.value b.wide v

theorem scalar_correspondence (b : Numerics) (d : Storage) (v : Word d) :
    scalarSource b d v=scalarFormula b d v := by
  cases d with
  | integer e => cases e <;> simp [scalarSource,selectSource,castStore,scalarFormula,IntegerWords.wordValue,zeroBackend]
  | fp16 | bf16 | fp32 => simp [scalarSource,selectSource,castStore,scalarFormula,TypedValue.resultValue,FP.Structural.Cell.read,zeroBackend]
  | fp64 => rfl

theorem storage_bytes_positive (d : Storage) : 0<d.bytes := by
  cases d with
  | integer e => exact FiniteAddress.element_bytes_positive e
  | fp16 | bf16 | fp32 | fp64 => decide

#specNonCircular scalarFormula avoiding [scalarSource,selectSource,castStore,TypedValue.source,FP.Structural.exec]
#axiomsClean scalar_correspondence
#axiomsClean storage_bytes_positive
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ReluTritonKernel.Actual
