import bench.tritonbench_g.triton_linear_activation.Proofs.KernelRealArithmetic
import bench.tritonbench_g.triton_linear_activation.Proofs.FPReference
import bench.tritonbench_g.triton_linear_activation.Proofs.Mathematical_FPLiterals

namespace VeriTile.Bench.Repairs.TritonLinearActivation.KernelNativeModel
open VeriTile.Triton FP.Structural KernelSource KernelValues
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

/-- Source expression graph, before compiler contraction. Even the bridge
from a typed dot result to the real channel remains an explicit cast. -/
inductive Term where
  | raw (word : Word)
  | literal (precision : Option ComputeDType) (dtype : FloatDType) (value : ℝ)
  | cast (precision : Option ComputeDType) (src dst : FloatDType) (value : Term)
  | binary (precision : Option ComputeDType) (dtype : FloatDType) (op : Binary) (left right : Term)
  | unary (precision : Option ComputeDType) (op : Unary) (value : Term)
  | dot (precision : Option ComputeDType) (left right out : FloatDType) (options : DotOptions)
      (kind : Instruction) (k : Nat) (xs ys : Fin k → Term)

instance : Inhabited Term := ⟨.raw (FP.Value.zero .fp32)⟩
instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def algebra : Algebra Term where
  literal := Term.literal
  negInf := .raw (FP.Value.infinity .fp32 Bool.true)
  binary := Term.binary
  unary := Term.unary
  cast := Term.cast
  fromNat := fun p n => .literal p .real n
  fromInt := fun p n => .literal p .real n
  fp32Bits := fun bits => .raw (FP.Value.ofBits .fp32 bits.bits.toNat)
  fp32Load := id
  reduceMax := fun _ _ _ _ _ => default
  reduceSum := fun _ _ _ _ _ => default

def dot (c : Fin 98) (mode : Mode) : Dot Term where
  apply := fun precision left right out options =>
    KernelRealArithmetic.laneDot (fun {k} xs ys =>
      .dot precision left right out options (selectedInstruction c mode) k xs ys)

/-- An explicit compiler policy; contracting dot/add and scalar multiply/add
changes evaluation, not an asserted floating identity. These are compiler-model
boundaries, not claims that an uninspected GPU binary follows this policy. -/
structure CompilerPolicy where
  contractDotAdd : Bool
  contractMultiplyAdd : Bool
  deriving DecidableEq

def sourceOrder : CompilerPolicy := ⟨Bool.false,Bool.false⟩
def contracted : CompilerPolicy := ⟨Bool.true,Bool.true⟩

/-- The only matrix arithmetic port is one named fixed K8/K16 instruction
lane. Arbitrary-K dots are constructed as a finite chronological instruction
sequence. Input FP16 words remain raw half containers; TF32 receives raw FP32. -/
def instructionDot (profile : LocalProfile) (kind : Instruction) (k : Nat)
    (xs ys : Fin k → Word) (seed : Word) : Word :=
  foldN (fun group acc => profile.instructionLane kind
    (fun lane => if h : group*kind.width+lane.val<k then xs ⟨_,h⟩ else literalWord .zero)
    (fun lane => if h : group*kind.width+lane.val<k then ys ⟨_,h⟩ else literalWord .zero) acc)
    (k/kind.width) seed

def castWord (src dst : FloatDType) (x : Word) : Word :=
  match src,dst with
  | .fp16,.real | .fp16,.fp32 => unpackHalf x
  | .real,.fp16 | .fp32,.fp16 => packHalf x
  | _,_ => x

def binaryWord (profile : LocalProfile) : Binary → Word → Word → Word
  | .add => FP.add FP.Config.ieeeValue
  | .sub => FP.sub FP.Config.ieeeValue
  | .mul => FP.mul FP.Config.ieeeValue
  | .div => profile.divFullF32
  | .max => maxnum
  | .pow => fun _ _ => FP.Value.qNaN .fp32

def unaryWord (profile : LocalProfile) : Unary → Word → Word
  | .tanh => profile.libdeviceTanhF32
  | .erf => profile.libdeviceErfF32
  | _ => id

/-- Source literal tags bind the actual fixed binary32 constants. The two
sqrt expressions are source tags for host constants, not exact host Real sqrt. -/
def constantWord (x : ℝ) : Word :=
  VeriTile.Bench.TritonBenchG.TritonLinearActivation.FPValue.literalWord
    (fun _ => FP.Value.qNaN .fp32) x

def evaluate (profile : LocalProfile) (policy : CompilerPolicy) : Term → Word
  | .raw word => word
  | .literal _ d x => castWord .real d (constantWord x)
  | .cast _ src dst value => castWord src dst (evaluate profile policy value)
  | .binary _ _ .add a (.cast _ .fp32 .real (.dot _ _ _ _ _ kind k xs ys)) =>
      if policy.contractDotAdd then
        instructionDot profile kind k (fun j => evaluate profile policy (xs j))
          (fun j => evaluate profile policy (ys j)) (evaluate profile policy a)
      else FP.add FP.Config.ieeeValue (evaluate profile policy a)
        (instructionDot profile kind k (fun j => evaluate profile policy (xs j))
          (fun j => evaluate profile policy (ys j)) (literalWord .zero))
  | .binary _ _ .add a (.binary _ _ .mul b c) =>
      if policy.contractMultiplyAdd then
        FP.fma FP.Config.ieeeValue (evaluate profile policy b) (evaluate profile policy c) (evaluate profile policy a)
      else FP.add FP.Config.ieeeValue (evaluate profile policy a)
        (FP.mul FP.Config.ieeeValue (evaluate profile policy b) (evaluate profile policy c))
  | .binary _ _ op a b => binaryWord profile op (evaluate profile policy a) (evaluate profile policy b)
  | .unary _ op value => unaryWord profile op (evaluate profile policy value)
  | .dot _ _ _ _ _ kind k xs ys =>
      instructionDot profile kind k (fun j => evaluate profile policy (xs j))
        (fun j => evaluate profile policy (ys j)) (literalWord .zero)

def mapValue {α β : Type} (f : α → β) : (d : TileDType) → Value α d → Value β d
  | .real | .fp32 | .fp16 | .bf16 | .f8e4 | .f8e5 => f
  | .nat | .int | .bool | .ptr | .blockPtr => id

def mapCell {α β : Type} (f : α → β) : Cell α → Cell β
  | .mk d value => .mk d (mapValue f d value)

def mapState {α β : Type} (f : α → β) (s : State α) : State β where
  mem := fun r o => mapCell f (s.mem r o)
  regs := fun d sh n => (s.regs d sh n).map (fun values i => mapValue f d (values i))
  pids := s.pids
  numPids := s.numPids
  undef := fun d r o => mapValue f d (s.undef d r o)

/-- Actual native entry of this explicit compiler model: source syntax builds
its expression graph, then the selected compiler policy evaluates that graph.
No graph rewrite is passed off as a floating-point equality. -/
def execute (profile : LocalProfile) (policy : CompilerPolicy) (c : Fin 98)
    (mode : Mode) (p : Parameters) (initial : State Word) : Option (State Word) :=
  (exec (engine algebra (dot c mode)) (source mode p) (mapState Term.raw initial)).map
    (mapState (evaluate profile policy))

/-- A scalar recurrence independent of interpreter control and output stores. -/
def preExpression (c : Fin 98) (mode : Mode) (p : Parameters) (s : State Term)
    (i : TileIndex [p.blockRows,p.blockColumns]) : Term :=
  foldN (fun cur acc => .binary (some .fp32) .real .add acc
    (.cast (some .fp32) .fp32 .real
      (.dot (some .fp32) (dtype mode) (dtype mode) .fp32 {} (selectedInstruction c mode) p.blockInner
        (fun k => aValues algebra mode p s cur (i.1,k,PUnit.unit))
        (fun k => bValues algebra mode p s cur (k,i.2.1,PUnit.unit)))))
    (ceilDiv p.inner p.blockInner) (initial algebra mode p s i)

theorem accumulated_expression (c : Fin 98) (mode : Mode) (p : Parameters)
    (s : State Term) (i : TileIndex [p.blockRows,p.blockColumns]) :
    accumulated algebra (dot c mode) mode p s i=preExpression c mode p s i := by
  unfold accumulated preExpression
  generalize ceilDiv p.inner p.blockInner=count
  induction count with
  | zero => rfl
  | succ count ih =>
    simpa only [foldN,KernelValues.update,algebra,dot,KernelRealArithmetic.laneDot] using
      (congrArg (fun x => Term.binary (some .fp32) .real .add x
        (.cast (some .fp32) .fp32 .real
          (.dot (some .fp32) (dtype mode) (dtype mode) .fp32 {} (selectedInstruction c mode) p.blockInner
            (fun k => aValues algebra mode p s count (i.1,k,PUnit.unit))
            (fun k => bValues algebra mode p s count (k,i.2.1,PUnit.unit))))) ih)

def preWord (profile : LocalProfile) (policy : CompilerPolicy) (c : Fin 98)
    (mode : Mode) (p : Parameters) (s : State Word) (i : TileIndex [p.blockRows,p.blockColumns]) : Word :=
  evaluate profile policy (preExpression c mode p (mapState Term.raw s) i)

def savedWord (profile : LocalProfile) (policy : CompilerPolicy) (c : Fin 98)
    (mode : Mode) (p : Parameters) (s : State Word) (i : TileIndex [p.blockRows,p.blockColumns]) : Word :=
  castWord .real (dtype mode) (preWord profile policy c mode p s i)

def resultWord (profile : LocalProfile) (policy : CompilerPolicy) (c : Fin 98)
    (mode : Mode) (p : Parameters) (s : State Word) (i : TileIndex [p.blockRows,p.blockColumns]) : Word :=
  castWord .real (dtype mode) (evaluate profile policy
    (KernelValues.activation algebra p.activation (preExpression c mode p (mapState Term.raw s) i)))

theorem instruction_coverage (c : Fin 98) (mode : Mode) :
    (configs c).bk/(selectedInstruction c mode).width*(selectedInstruction c mode).width=(configs c).bk :=
  instruction_divides c mode

#axiomsClean accumulated_expression
#axiomsClean instruction_coverage
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonLinearActivation.KernelNativeModel
