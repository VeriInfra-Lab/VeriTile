import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.KernelModel
import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.RealLocal

namespace VeriTile.Bench.Repairs.FastCeLoss.KernelReal
open VeriTile.Triton FP.Structural KernelSource KernelModel
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

def lanes {α : Type} {shape : TileShape} (ax : Fin shape.length) (keep : Bool)
    (xs : Values α .real shape) :
    TileIndex (TileShape.reduceShape shape ax keep) → Fin (TileShape.axisDim shape ax) → α :=
  match keep with
  | Bool.false => fun i k => xs (TileShape.insertAxisIndex shape ax i k)
  | Bool.true => fun i k => xs (TileShape.replaceAxisIndex shape ax i k)

/-- Ideal-real interpretation. This is separate from native FP32 instructions;
masked negative infinity stays bottom and contributes zero exponential mass. -/
def algebra : Algebra (WithBot ℝ) where
  literal := fun _ _ x => some x
  negInf := none
  binary := fun _ _ op => match op with
    | .add => RealProof.arithmetic.add
    | .sub => RealProof.arithmetic.sub
    | .mul => RealProof.arithmetic.mul
    | .div => RealProof.arithmetic.div
    | .max => max
    | .pow => fun _ _ => none
  unary := fun _ op => match op with
    | .exp => RealProof.arithmetic.exp
    | .log => RealProof.arithmetic.log
    | .libdeviceTanh => RealProof.arithmetic.tanh
    | .tanh => RealProof.arithmetic.tanh
    | _ => fun _ => none
  cast := fun _ _ _ => id
  fromNat := fun _ n => some (n : ℝ)
  fromInt := fun _ n => some (n : ℝ)
  fp32Bits := fun _ => none
  fp32Load := id
  reduceSum := fun _ {shape} ax keep xs out => RealProof.arithmetic.sum (lanes ax keep xs out)
  reduceMax := fun _ {shape} ax keep xs out => RealProof.arithmetic.maximum (lanes ax keep xs out)

structure Flags (p : Parameters) : Prop where
  cap : p.hasSoftcap=decide (p.softcap≠0)
  scale : p.hasScaling=decide (p.scaling≠0)

def options (p : Parameters) (flags : Flags p) : Source.Options :=
  ⟨p.softcap,p.scaling,p.hasSoftcap,p.hasScaling,flags.cap,flags.scale⟩

theorem transform_finite (p : Parameters) (flags : Flags p) (x : ℝ) :
    cap algebra p (forwardScale algebra p (some x))=
      some (Math.transformed p.softcap p.scaling x) := by
  have h := RealProof.transform_finite (options p flags) x
  simpa only [Source.transform,options,cap,forwardScale,capPartial,
    literal,binary,unary,algebra,RealProof.arithmetic] using h

theorem logits_finite (p : Parameters) (initial : State (WithBot ℝ))
    (flags : Flags p) (x : Nat → ℝ) (chunk : Nat)
    (inputs : ∀ col<p.vocab,
      (initial.mem p.logits (initial.pids 0*p.logitsStride+col)).read .real=some (x col)) :
    forwardLogits algebra p initial chunk = fun (lane,_) =>
      if chunk*p.block+lane.val<p.vocab then
        some (Math.transformed p.softcap p.scaling (x (chunk*p.block+lane.val))) else none := by
  funext i
  rcases i with ⟨lane,u⟩
  by_cases active : chunk*p.block+lane.val<p.vocab
  · simp only [forwardLogits,active,if_true,loaded,inputs _ active,algebra,KernelModel.cast,id_eq]
    exact transform_finite p flags _
  · simp only [forwardLogits,active,if_false,algebra]

theorem lse_stable (p : Parameters) (initial : State (WithBot ℝ)) (chunk : Nat) :
    lseValue algebra p initial chunk=Source.stableLSE RealProof.arithmetic
      (fun lane => forwardLogits algebra p initial chunk (lane,PUnit.unit)) := by
  rfl

/-- The chunk formula is the logarithm of the exponential mass of its active
logical columns. No padded column survives the repaired source mask. -/
theorem lse_finite (p : Parameters) (initial : State (WithBot ℝ))
    (flags : Flags p) (x : Nat → ℝ) (chunk : Nat)
    (positive : 0<p.block) (nonempty : chunk*p.block<p.vocab)
    (inputs : ∀ col<p.vocab,
      (initial.mem p.logits (initial.pids 0*p.logitsStride+col)).read .real=some (x col)) :
    lseValue algebra p initial chunk=some (Real.log
      (∑ lane ∈ (Finset.univ : Finset (Fin p.block)).filter (fun j => chunk*p.block+j.val<p.vocab),
        Real.exp (Math.transformed p.softcap p.scaling (x (chunk*p.block+lane.val))))) := by
  rw [lse_stable,logits_finite p initial flags x chunk inputs]
  exact RealProof.stable_active _ _ positive ⟨⟨0,positive⟩,by simp [nonempty]⟩

/-- The backward formula uses the supplied saved LSE and row upstream value.
It does not assume any host normalization or saved-LSE recomputation. -/
def gradient (p : Parameters) (label : Int) (col : Nat) (x lse dloss : ℝ) : ℝ :=
  if label = -100 then 0 else dloss *
    (Real.exp (Math.transformed p.softcap p.scaling x-lse) - if (col : Int)=label then 1 else 0) *
    Math.derivative p.softcap p.scaling x

theorem gradient_finite (p : Parameters) (initial : State (WithBot ℝ))
    (flags : Flags p) (lane : Fin p.block) (x lse dloss : ℝ)
    (active : initial.pids 1*p.block+lane.val<p.vocab)
    (input : (initial.mem p.logits
      (initial.pids 0*p.logitsStride+(initial.pids 1*p.block+lane.val))).read .real=some x)
    (saved : (initial.mem p.lse (initial.pids 0)).read .real=some lse)
    (upstream : (initial.mem p.upstream (initial.pids 0*p.upstreamStride)).read .real=some dloss) :
    gradientValue algebra p initial lane=
      some (gradient p (label p initial) (initial.pids 1*p.block+lane.val) x lse dloss) := by
  unfold gradientValue gradient backwardDerivative backwardCorrected backwardProbability
    upstreamValue savedLSE loaded backwardScale cap capPartial KernelModel.cast literal binary unary
  rw [flags.cap,flags.scale]
  simp only [active,input,saved,upstream,if_true]
  by_cases hc : p.softcap=0 <;> by_cases hs : p.scaling=0
  all_goals by_cases ignored : label p initial = -100
  all_goals by_cases target : ((initial.pids 1*p.block+lane.val : Nat) : Int)=label p initial
  all_goals simp only [target,if_true,if_false]
  all_goals simp [algebra,RealProof.arithmetic,Math.transformed,Math.derivative,Math.scale,
    hc,hs,ignored,target,active,input,saved,upstream,WithBot.realTanh,WithBot.realExp,mul_comm]
    <;> congr 1 <;> ring

#axiomsClean transform_finite
#axiomsClean logits_finite
#axiomsClean lse_finite
#axiomsClean gradient_finite
#specNonCircular gradient avoiding [forward,chunked,backward,FP.Structural.exec,FP.Structural.run]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.KernelReal
