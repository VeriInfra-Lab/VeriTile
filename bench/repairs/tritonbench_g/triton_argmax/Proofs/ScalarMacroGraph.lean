import bench.repairs.tritonbench_g.triton_argmax.Proofs.ConversionMacro

/-! Local source/emitted scalar node refinement for the named saturating
conversion. A path contains individual opcode/operand/constant equations; it
does not contain native conversion, output or memory-frame equalities. The
separate byte decoder still must attach these typed nodes and their input
reference to the exact selected reduction root and actual stores/loads. -/
namespace VeriTile.Bench.Repairs.TritonArgmax.ScalarMacroGraph
open Source Formats Memory ConversionMacro

inductive Operand where
  | register : Nat → Operand
  | literal : Word → Operand
  deriving DecidableEq,Repr

def Operand.eval (value : Nat → Word) : Operand → Word
  | .register r => value r
  | .literal w => w

inductive Opcode where
  | equal32 : Operand → Operand → Opcode
  | ordered32 : Operand → Operand → Opcode
  | maximum32 : Operand → Operand → Opcode
  | minimum32 : Operand → Operand → Opcode
  | select32 : Nat → Operand → Operand → Opcode
  | truncate : Bool → Nat → Operand → Opcode
  deriving DecidableEq,Repr

abbrev Graph := Nat → Option Opcode

/-- Ordinary single-instruction evaluation only. In particular the cast rule
requires the exact selected width and signedness in the instruction syntax. -/
structure Evaluation (graph : Graph) (kind : IntegerKind) (p : ScalarBinding.IntegerPrimitives kind) where
  word : Nat → Word
  predicate : Nat → Bool
  integer : Nat → BitVec kind.bits
  equal32 : ∀ r a b,graph r=some (.equal32 a b) →
    predicate r=ScalarBinding.eq32 (a.eval word) (b.eval word)
  ordered32 : ∀ r a b,graph r=some (.ordered32 a b) →
    predicate r=ConversionMacro.ordered32 (a.eval word) (b.eval word)
  maximum32 : ∀ r a b,graph r=some (.maximum32 a b) →
    word r=p.maximum32 (a.eval word) (b.eval word)
  minimum32 : ∀ r a b,graph r=some (.minimum32 a b) →
    word r=p.minimum32 (a.eval word) (b.eval word)
  select32 : ∀ r condition a b,graph r=some (.select32 condition a b) →
    word r=if predicate condition then a.eval word else b.eval word
  truncate : ∀ r a,graph r=some (.truncate kind.signed kind.bits a) →
    integer r=p.truncate (a.eval word)

structure SourcePath (graph : Graph) (kind : IntegerKind) (input cast : Nat) where
  condition : Nat
  clean : Nat
  lower : Nat
  upper : Nat
  comparison : graph condition=some (.equal32 (.register input) (.register input))
  selection : graph clean=some (.select32 condition (.register input) (.literal 0))
  lowerClamp : graph lower=some (.maximum32 (.register clean) (.literal (ScalarBinding.lowerBits kind)))
  upperClamp : graph upper=some (.minimum32 (.register lower) (.literal (ScalarBinding.upperBits kind)))
  conversion : graph cast=some (.truncate kind.signed kind.bits (.register upper))
  order : input < condition ∧ condition < clean ∧ clean < lower ∧ lower < upper ∧ upper < cast

structure EmittedPath (graph : Graph) (kind : IntegerKind) (input cast : Nat) where
  condition : Nat
  lower : Nat
  upper : Nat
  clean : Nat
  comparison : graph condition=some (.ordered32 (.register input) (.literal 0))
  lowerClamp : graph lower=some (.maximum32 (.register input) (.literal (ScalarBinding.lowerBits kind)))
  upperClamp : graph upper=some (.minimum32 (.register lower) (.literal (ScalarBinding.upperBits kind)))
  selection : graph clean=some (.select32 condition (.register upper) (.literal 0))
  conversion : graph cast=some (.truncate kind.signed kind.bits (.register clean))
  order : input < condition ∧ condition < lower ∧ lower < upper ∧ upper < clean ∧ clean < cast

theorem source_refines {input cast : Nat} (evaluation : Evaluation graph kind p) (path : SourcePath graph kind input cast) :
    evaluation.integer cast=(SaturatingConversion.mk kind).sourceStore p (evaluation.word input) := by
  rw [evaluation.truncate _ _ path.conversion]
  simp only [Operand.eval]
  rw [evaluation.minimum32 _ _ _ path.upperClamp]
  simp only [Operand.eval]
  rw [evaluation.maximum32 _ _ _ path.lowerClamp]
  simp only [Operand.eval]
  rw [evaluation.select32 _ _ _ _ path.selection,evaluation.equal32 _ _ _ path.comparison]
  rfl

theorem emitted_interpretation {input cast : Nat} (evaluation : Evaluation graph kind p) (path : EmittedPath graph kind input cast) :
    evaluation.integer cast=(SaturatingConversion.mk kind).emittedStore p (evaluation.word input) := by
  rw [evaluation.truncate _ _ path.conversion]
  simp only [Operand.eval]
  rw [evaluation.select32 _ _ _ _ path.selection,evaluation.ordered32 _ _ _ path.comparison]
  simp only [Operand.eval]
  rw [evaluation.minimum32 _ _ _ path.upperClamp]
  simp only [Operand.eval]
  rw [evaluation.maximum32 _ _ _ path.lowerClamp]
  rfl

theorem emitted_refines {input cast : Nat} (evaluation : Evaluation graph kind p) (path : EmittedPath graph kind input cast)
    (zero : ZeroClamp kind p) :
    evaluation.integer cast=(SaturatingConversion.mk kind).sourceStore p (evaluation.word input) := by
  rw [emitted_interpretation evaluation path,emitted_source _ p zero]

/-- Refinement extends through an actual native-width memory store/load and
the exact integer/bool decode, with no whole-program memory premise. -/
theorem emitted_store_load {input cast : Nat} (evaluation : Evaluation graph kind p) (path : EmittedPath graph kind input cast)
    (zero : ZeroClamp kind p) (mem : Primitives M) (state : M) (point : Address) :
    decodeStored kind (mem.read (Formats.format kind).bytes
      (mem.write (Formats.format kind).bytes state point
        ((evaluation.integer cast).cast (format_width kind))) point)=
      (SaturatingConversion.mk kind).native p (evaluation.word input) := by
  rw [mem.readSame,emitted_refines evaluation path zero]
  rfl

/-- Typed local source graph. These constructors are normalization helpers,
not a claim that an arbitrary selected compiler artifact has this graph. -/
def sourceGraph (kind : IntegerKind) : Graph
  | 1 => some (.equal32 (.register 0) (.register 0))
  | 2 => some (.select32 1 (.register 0) (.literal 0))
  | 3 => some (.maximum32 (.register 2) (.literal (ScalarBinding.lowerBits kind)))
  | 4 => some (.minimum32 (.register 3) (.literal (ScalarBinding.upperBits kind)))
  | 5 => some (.truncate kind.signed kind.bits (.register 4))
  | _ => none

def sourcePath (kind : IntegerKind) : SourcePath (sourceGraph kind) kind 0 5 where
  condition:=1
  clean:=2
  lower:=3
  upper:=4
  comparison:=rfl
  selection:=rfl
  lowerClamp:=rfl
  upperClamp:=rfl
  conversion:=rfl
  order:=by decide

def emittedGraph (kind : IntegerKind) : Graph
  | 1 => some (.ordered32 (.register 0) (.literal 0))
  | 2 => some (.maximum32 (.register 0) (.literal (ScalarBinding.lowerBits kind)))
  | 3 => some (.minimum32 (.register 2) (.literal (ScalarBinding.upperBits kind)))
  | 4 => some (.select32 1 (.register 3) (.literal 0))
  | 5 => some (.truncate kind.signed kind.bits (.register 4))
  | _ => none

def emittedPath (kind : IntegerKind) : EmittedPath (emittedGraph kind) kind 0 5 where
  condition:=1
  lower:=2
  upper:=3
  clean:=4
  comparison:=rfl
  lowerClamp:=rfl
  upperClamp:=rfl
  selection:=rfl
  conversion:=rfl
  order:=by decide

#axiomsClean source_refines
#axiomsClean emitted_interpretation
#axiomsClean emitted_refines
#axiomsClean emitted_store_load
#axiomsClean sourcePath
#axiomsClean emittedPath
end VeriTile.Bench.Repairs.TritonArgmax.ScalarMacroGraph
