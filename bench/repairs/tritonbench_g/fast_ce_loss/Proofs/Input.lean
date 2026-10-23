import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.HostReal

namespace VeriTile.Bench.Repairs.FastCeLoss.Host
open Source
noncomputable section

/-- Fixed finite IEEE binary64 decoding for Python floats. Sign is retained in
ABI bits; both signed zeros have mathematical value zero and disable a branch. -/
def float64Value (bits : BitVec 64) : Option ℝ :=
  let exponent : Nat := (bits.toNat/2^52)%2048
  let fraction : Nat := bits.toNat%2^52
  if exponent=2047 then none else
    let significand : Nat := if exponent=0 then fraction else 2^52+fraction
    let power : Int := (if exponent=0 then -1022 else (exponent : Int)-1023)-52
    some ((if bits.toNat/2^63=0 then 1 else -1)*(significand : ℝ)*(2:ℝ)^power)

def pythonValue : PythonScalar → Option ℝ
  | .integer n => some (n : ℝ)
  | .floating bits => float64Value bits

def realScalars : ScalarInterpreter (WithBot ℝ) :=
  ⟨fun n => some (n : ℝ),float64Value⟩

def Descriptor.resolved (d : Descriptor) : Prop := d.isNeg=Bool.false ∧ d.isConj=Bool.false

/-- Normal public physical ABI, including the external scalar upstream. Raw or
resolved views may have nonzero offsets and negative stored values. No lazy
negative/conjugate external views, tensor denominators or denominator gradients
are claimed. Internal expanded autograd strides are derived, not constrained. -/
structure InputContract {σ : Type} (live : σ → Nat → Prop) (args : Arguments)
    (upstream : Descriptor) (native : σ) (integers : Address → Int) : Prop where
  positive : 0<args.vocab
  logitsShape : args.logits.tensor.shape=[args.batch,args.sequence,args.vocab]
  labelsShape : args.labels.tensor.shape=[args.batch,args.sequence]
  upstreamShape : upstream.tensor.shape=[]
  logitsDType : args.logits.dtype=.fp32
  labelsDType : args.labels.dtype=.int64
  upstreamDType : upstream.dtype=.fp32
  labelsDevice : args.labels.device=args.logits.device
  upstreamDevice : upstream.device=args.logits.device
  logitsContiguous : args.logits.contiguous=Bool.true
  labelsContiguous : args.labels.contiguous=Bool.true
  logitsResolved : args.logits.resolved
  labelsResolved : args.labels.resolved
  upstreamResolved : upstream.resolved
  logitsBounds : args.logits.bounded
  labelsBounds : args.labels.bounded
  upstreamBounds : upstream.bounded
  logitsLive : live native args.logits.tensor.storage
  labelsLive : live native args.labels.tensor.storage
  upstreamLive : live native upstream.tensor.storage
  typedSeparation : args.logits.tensor.storage≠args.labels.tensor.storage
  upstreamTypedSeparation : upstream.tensor.storage≠args.labels.tensor.storage
  labelsValid : ∀ row<args.shape.rows,
    integers (addr args.labels.tensor row)=-100 ∨
    (0 ≤ integers (addr args.labels.tensor row) ∧ integers (addr args.labels.tensor row) < args.vocab)
  capBinding : pythonValue args.capArgument=some args.options.softcap
  scaleBinding : pythonValue args.scaleArgument=some args.options.scaling
  denominator : match args.items with
    | none => 0<Source.count args.shape (fun row => integers (addr args.labels.tensor row))
    | some value => ∃ real,pythonValue value=some real ∧ 0<real

#specNonCircular InputContract avoiding [Math.loss,Math.gradient,Complete.lseGraph,Complete.scalarGraph,API.gradientGraph]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.Host
