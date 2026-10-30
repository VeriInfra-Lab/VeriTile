import VeriTile.Triton
import VeriTile.Triton.Float.ScalarBackend

/-! Additive execution of the existing llrint source node and integer-pointer
store conversions. The original Structural evaluator and all old theorems are
unchanged. The storage map is part of the source signature, since the current
DSL erases integer-pointer widths when constructing a store statement.

This executor is for straight-line kernels. Other statements retain the
existing interpreter behavior; it makes no claim to extend llrint inside
control flow. The two clients below have no control-flow statements.
-/
namespace QuantizeIntegerStage
open VeriTile.Triton FP.Structural
noncomputable section

abbrev IntegerStorage := RegionName → Option Nat
abbrev Llrint64 (α : Type) := α → BitVec 64
abbrev RuntimeRegisters := String → Option RegName

structure TypedKernel where
  program : ComputeKernel
  integerStorage : IntegerStorage
  runtimeRegisters : RuntimeRegisters

/-- Exact source-produced marker, bound to its integer register. Other opaque
operations remain unsupported. Each client proves its exact source AST. -/
def outputRegister : RuntimeRegisters := fun marker =>
  if marker = "runtime compute value `output`" then some "output" else none

def byteOutput (out : RegionName) : IntegerStorage :=
  fun r => if r = out then some 8 else none

/-- The callback supplies the native numerical llrint boundary, including its
chosen exceptional-value behavior. The result is a signed 64-bit word, never
an unconstrained mathematical integer or a toward-zero conversion. -/
noncomputable def evalExpr {α : Type} [Inhabited α] (M : Algebra α)
    (round : Option (Llrint64 α)) (bindings : RuntimeRegisters) :
    ComputeExpr dtype shape → State α → Option (Values α dtype shape)
  | .llrint arg, s => do
      let f ← round
      let values ← evalExpr M round bindings arg s
      return fun i => (f (values i)).toInt
  | .isFinite precision arg, s => do
      let values ← evalExpr M round bindings arg s
      let classify ← M.isFinite precision
      return fun i => classify (values i)
  | .fma precision bcAB bcC a b c, s => do
      let fused ← M.fma precision
      let av ← evalExpr M round bindings a s
      let bv ← evalExpr M round bindings b s
      let cv ← evalExpr M round bindings c s
      return fun i => fused
        (av (bcAB.leftIndex (bcC.leftIndex i)))
        (bv (bcAB.rightIndex (bcC.leftIndex i)))
        (cv (bcC.rightIndex i))
  | .fmaTyped dtype bcAB bcC a b c, s => do
      let fused ← M.fmaTyped dtype
      let av ← evalExpr M round bindings a s
      let bv ← evalExpr M round bindings b s
      let cv ← evalExpr M round bindings c s
      return fun i => ofFloat dtype (fused
        (toFloat dtype (av (bcAB.leftIndex (bcC.leftIndex i))))
        (toFloat dtype (bv (bcAB.rightIndex (bcC.leftIndex i))))
        (toFloat dtype (cv (bcC.rightIndex i))))
  | .alg e, s => FP.Structural.evalOp M none e s
  | .compute e, s => FP.Structural.evalComputeOp M e s
  | .opaque (dtype := .int) marker, s => do
      let name ← bindings marker
      s.regs .int _ name
  | .opaque _, _ => none
termination_by structural e _ => e

def storedInteger (storage : IntegerStorage) (region : RegionName) (value : Int) : Int :=
  match storage region with
  | some width => (BitVec.ofInt width value).toInt
  | none => value

/-- Narrowing happens at the executing store boundary. Address/mask evaluation,
lane order and writes are delegated unchanged to Structural.store. -/
noncomputable def store {α : Type} [Inhabited α] (M : Algebra α)
    (storage : IntegerStorage) :
    {dtype : TileDType} → {shape : TileShape} → MemAccess dtype shape →
    MaskOpt dtype shape → Values α dtype shape → State α → Option (State α)
  | .int, _, .region r off, mask, values, s =>
      FP.Structural.store M (.region r off) mask
        (fun i => storedInteger storage (Region.cast r) (values i)) s
  | .int, _, .ptr address, mask, values, s => do
      let addresses ← FP.Structural.evalOp M none address s
      FP.Structural.store M (.ptr address) mask
        (fun i => storedInteger storage (addresses i).1 (values i)) s
  | .int, _, .blockPtr address checkedAxes, mask, values, s => do
      let addresses ← FP.Structural.evalOp M none address s
      FP.Structural.store M (.blockPtr address checkedAxes) mask
        (fun i => storedInteger storage (addresses i).region (values i)) s
  | _, _, mem, mask, values, s => FP.Structural.store M mem mask values s

noncomputable def step {α : Type} [Inhabited α] (M : Algebra α)
    (round : Option (Llrint64 α)) (storage : IntegerStorage)
    (bindings : RuntimeRegisters) :
    ComputeStmt → State α → Option (State α)
  | .assign d sh name e, s => do
      let values ← evalExpr M round bindings e s
      return s.setReg name d sh values
  | .store _ _ mem e mask, s => do
      store M storage mem mask (← evalExpr M round bindings e s) s
  | .alg (.store _ _ mem e mask), s => do
      store M storage mem mask (← FP.Structural.evalOp M none e s) s
  | stmt, s => FP.Structural.step M stmt s

noncomputable def run {α : Type} [Inhabited α] (M : Algebra α)
    (round : Option (Llrint64 α)) (storage : IntegerStorage)
    (bindings : RuntimeRegisters) :
    List ComputeStmt → State α → Option (State α)
  | [], s => some s
  | stmt :: rest, s => do run M round storage bindings rest (← step M round storage bindings stmt s)

noncomputable def exec {α : Type} [Inhabited α] (M : Algebra α)
    (round : Option (Llrint64 α)) (kernel : TypedKernel) (s : State α) : Option (State α) :=
  run M round kernel.integerStorage kernel.runtimeRegisters kernel.program.surfaceBody s

theorem missing_llrint_fails {α : Type} [Inhabited α] (M : Algebra α)
    (arg : ComputeExpr .real shape) (s : State α) :
    evalExpr M none outputRegister (.llrint arg) s = none := rfl

theorem llrint_executes_word {α : Type} [Inhabited α] (M : Algebra α)
    (round : Llrint64 α) (arg : ComputeExpr .real shape) (s : State α)
    (values : Values α .real shape) (h : evalExpr M (some round) outputRegister arg s = some values) :
    evalExpr M (some round) outputRegister (.llrint arg) s = some (fun i => (round (values i)).toInt) := by
  simp [evalExpr, h]

theorem byte_word (out : RegionName) (value : Int) :
    BitVec.ofInt 8 (storedInteger (byteOutput out) out value) = BitVec.ofInt 8 value := by
  simp only [storedInteger, byteOutput, ite_true, BitVec.ofInt_toInt]

theorem fold_memory {α ι : Type} (l : List ι) (r : RegionName)
    (off : ι → Nat) (value : ι → Cell α) (active : ι → Prop) [DecidablePred active] (s : State α) :
    (l.foldl (fun t i => if active i then t.write r (off i) (value i) else t) s).mem =
      l.foldl (fun mem i => fun r' o =>
        if active i ∧ r' = r ∧ o = off i then value i else mem r' o) s.mem := by
  induction l generalizing s with
  | nil => rfl
  | cons i l ih =>
    by_cases hi : active i
    · simp only [List.foldl_cons, if_pos hi, ih]
      congr 1
      funext r' o
      simp [hi, State.write]
    · simp only [List.foldl_cons, if_neg hi, ih]
      congr 1
      funext r' o
      simp [hi]

namespace FP32
abbrev Word := FP.Value .fp32
instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
def scale127 : Word := FP.Value.ofBits .fp32 0x42fe0000
noncomputable def backend (rest : Algebra Word) (cfg : FP.Config) : Algebra Word :=
  let base := FP.Software.scalarFP32 rest cfg
  { base with literal := fun p d x =>
      if FP.Software.fp32Context p d && decide (x = 127) then scale127
      else base.literal p d x }
def scaled (cfg : FP.Config) (x scale : Word) : Word :=
  FP.mul cfg scale127 (FP.mul cfg x scale)
def outputWord (round : Llrint64 Word) (cfg : FP.Config) (x scale : Word) : BitVec 8 :=
  BitVec.ofInt 8 (round (scaled cfg x scale)).toInt
end FP32

#axiomsClean missing_llrint_fails
#axiomsClean llrint_executes_word
#axiomsClean byte_word
#axiomsClean fold_memory
end
end QuantizeIntegerStage
