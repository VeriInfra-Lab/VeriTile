import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8.Proofs.Execution

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Floating
noncomputable section
set_option maxHeartbeats 5000000
abbrev Word := BitVec 32

/-- Actual recorded SM90 instruction; K8 is smaller than source BK64.
Its input words are the actual FP32 storage bits interpreted by the TF32 ISA.
No unobserved scalar TF32 conversion is inserted into this graph. -/
inductive ISA where | wgmmaTF32
  deriving DecidableEq

def ISA.m (_ : ISA) : Nat := 64
def ISA.n (_ : ISA) : Nat := 256
def ISA.k (_ : ISA) : Nat := 8
def ISA.storage (_ : ISA) : Storage := .fp32
def ISA.name (_ : ISA) : String := "wgmma.mma_async.sync.aligned.m64n256k8.f32.tf32.tf32"

abbrev ISA.Left (op : ISA) (α : Type) := Fin op.m → Fin op.k → α
abbrev ISA.Right (op : ISA) (α : Type) := Fin op.k → Fin op.n → α
abbrev ISA.Acc (op : ISA) (α : Type) := Fin op.m → Fin op.n → α

/-- Scalar expression markers retain the complete supplied-accumulator dot.
There is deliberately no numerical evaluator for this whole-dot syntax. -/
inductive Expr where
  | raw (word : Word)
  | zero
  | cast (value : Expr)
  | leaky (value : Expr)
  | dot (width : Nat) (a : Fin 128 → Fin width → Expr)
      (b : Fin width → Fin 256 → Expr) (acc : Source.Acc Expr) (row : Fin 128) (column : Fin 256)

def arithmetic : Source.Arithmetic Expr Expr where
  inputZero := .zero
  accumulatorZero := .zero
  dot := Expr.dot
  activation enabled value := if enabled then .leaky value else value
  cast _ := Expr.cast

/-- Actual numerical graph contains the output conversion, scalar leaky-ReLU
comparison/multiplication, and fixed ISA MMA at ordered matrix/accumulator operands. -/
inductive Graph where
  | raw (word : Word)
  | cast (value : Graph)
  | leaky (swappedMultiply : Bool) (value : Graph)
  | instruction (op : ISA) (a : op.Left Graph) (b : op.Right Graph) (acc : op.Acc Graph)
      (row : Fin op.m) (column : Fin op.n)

structure Numerics where
  cast : Word → Word
  nonnegative : Word → Bool
  multiply : Word → Word → Word
  instruction : (op : ISA) → op.Left Word → op.Right Word → op.Acc Word → op.Acc Word

/-- The source literal is rounded to this exact FP32 word. A native
right-side placement is explicit and never assumed numerically harmless. -/
def slopeWord : Word := BitVec.ofNat 32 0x3c23d70a

def multiplySlope (numerics : Numerics) (swapped : Bool) (word : Word) : Word :=
  if swapped then numerics.multiply word slopeWord else numerics.multiply slopeWord word

def Graph.eval (numerics : Numerics) : Graph → Word
  | .raw word => word
  | .cast value => numerics.cast (value.eval numerics)
  | .leaky swapped value =>
      let word := value.eval numerics
      if numerics.nonnegative word then word else multiplySlope numerics swapped word
  | .instruction op a b acc row column => numerics.instruction op
      (fun i k => (a i k).eval numerics) (fun k j => (b k j).eval numerics)
      (fun i j => (acc i j).eval numerics) row column

/-- Source reference fixes the original left-side literal multiplication.
Only the explicit local multiplication port can justify native operand swaps. -/
def Graph.sourceEval (reference : Numerics) : Graph → Word
  | .raw word => word
  | .cast value => reference.cast (value.sourceEval reference)
  | .leaky _ value =>
      let word := value.sourceEval reference
      if reference.nonnegative word then word else reference.multiply slopeWord word
  | .instruction op a b acc row column => reference.instruction op
      (fun i k => (a i k).sourceEval reference) (fun k j => (b k j).sourceEval reference)
      (fun i j => (acc i j).sourceEval reference) row column

structure Domain where
  cast : Word → Prop
  nonnegative : Word → Prop
  mulSlope : Bool → Word → Prop
  instruction : (op : ISA) → op.Left Word → op.Right Word → op.Acc Word → Prop

/-- The four deferred numerical fields are one actual output conversion,
one comparison, one ordered slope multiplication, and one fixed-size ISA
instruction. No tl.dot, reduction, activation, or whole-output equality port. -/
structure LocalPorts (native reference : Numerics) (domain : Domain) : Prop where
  cast : ∀ word,domain.cast word → native.cast word=reference.cast word
  nonnegative : ∀ word,domain.nonnegative word → native.nonnegative word=reference.nonnegative word
  mulSlope : ∀ swapped word,domain.mulSlope swapped word →
    multiplySlope native swapped word=reference.multiply slopeWord word
  instruction : ∀ op a b acc,domain.instruction op a b acc →
    native.instruction op a b acc=reference.instruction op a b acc

def Graph.Admitted (native : Numerics) (domain : Domain) : Graph → Prop
  | .raw _ => True
  | .cast value => value.Admitted native domain ∧ domain.cast (value.eval native)
  | .leaky swapped value => value.Admitted native domain ∧ domain.nonnegative (value.eval native) ∧
      domain.mulSlope swapped (value.eval native)
  | .instruction op a b acc _ _ =>
      (∀ i k,(a i k).Admitted native domain) ∧ (∀ k j,(b k j).Admitted native domain) ∧
      (∀ i j,(acc i j).Admitted native domain) ∧
      domain.instruction op (fun i k => (a i k).eval native) (fun k j => (b k j).eval native)
        (fun i j => (acc i j).eval native)

theorem Graph.equivalent (native reference : Numerics) (domain : Domain)
    (ports : LocalPorts native reference domain) (graph : Graph) (admitted : graph.Admitted native domain) :
    graph.eval native=graph.sourceEval reference := by
  induction graph with
  | raw => rfl
  | cast value ih =>
    change native.cast (value.eval native)=reference.cast (value.sourceEval reference)
    rw [ports.cast _ admitted.2,ih admitted.1]
  | leaky swapped value ih =>
    change (if native.nonnegative (value.eval native) then value.eval native else multiplySlope native swapped (value.eval native))=
      (if reference.nonnegative (value.sourceEval reference) then value.sourceEval reference
       else reference.multiply slopeWord (value.sourceEval reference))
    rw [ports.nonnegative _ admitted.2.1,ports.mulSlope _ _ admitted.2.2,ih admitted.1]
  | instruction op a b acc row column ia ib ic =>
    change native.instruction op _ _ _ row column=reference.instruction op _ _ _ row column
    rw [ports.instruction op _ _ _ admitted.2.2.2]
    have aa : (fun i k => (a i k).eval native)=(fun i k => (a i k).sourceEval reference) := by
      funext i k; exact ia i k (admitted.1 i k)
    have bb : (fun k j => (b k j).eval native)=(fun k j => (b k j).sourceEval reference) := by
      funext k j; exact ib k j (admitted.2.1 k j)
    have cc : (fun i j => (acc i j).eval native)=(fun i j => (acc i j).sourceEval reference) := by
      funext i j; exact ic i j (admitted.2.2.1 i j)
    rw [aa,bb,cc]

theorem fixed_instruction_limits (op : ISA) : op.m<128 ∧ op.k=8 ∧ op.k<op.storage.blockK := by
  cases op <;> decide

#axiomsClean Graph.equivalent
#specNonCircular LocalPorts avoiding [Source.run,Source.accumulator,Expr.dot]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Floating
