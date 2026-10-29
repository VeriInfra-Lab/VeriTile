import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.KernelModel

namespace VeriTile.Bench.Repairs.FastCeLoss.LogicalModel
open VeriTile.Triton FP.Structural KernelSource KernelModel
noncomputable section
variable {α : Type}

/-- Ordered FP32 tile expression, over logical values rather than memory. -/
def logits (M : Algebra α) (p : Parameters) (chunk : Nat) (x : Fin p.block → α) :
    Values α .real [p.block] := fun j =>
  if chunk*p.block+j.1.val<p.vocab then
    cast M (cap M p (forwardScale M p (M.fp32Load (x j.1)))) else M.negInf

def maximum (M : Algebra α) (p : Parameters) (chunk : Nat) (x : Fin p.block → α) : α :=
  M.reduceMax (some .fp32) ⟨0,by simp⟩ Bool.false (logits M p chunk x) PUnit.unit

def lse (M : Algebra α) (p : Parameters) (chunk : Nat) (x : Fin p.block → α) : α :=
  let m := maximum M p chunk x
  binary M .add m (unary M .log
    (M.reduceSum (some .fp32) ⟨0,by simp⟩ Bool.false
      (fun j => unary M .exp (binary M .sub (logits M p chunk x j) m)) PUnit.unit))

def forwardLoss (M : Algebra α) (p : Parameters) (x : Fin p.block → α)
    (label : Int) (target : α) : α :=
  if label = -100 then literal M 0 else
    binary M .sub (lse M p 0 x) (cast M (cap M p (forwardScale M p (M.fp32Load target))))

def chunkLoss (M : Algebra α) (p : Parameters) (label : Int) (target : α) : α :=
  if label = -100 then literal M 0 else
    binary M .mul (literal M (-1)) (cast M (cap M p (forwardScale M p (cast M target))))

/-- The existing source-ordered derivative, including the literal ignored-row
zero, actual FP32 loads and stores, and backward multiplication order. -/
def gradient (M : Algebra α) (p : Parameters) (col : Nat) (label : Int)
    (x saved upstream : α) : α :=
  let scaled := backwardScale M p (M.fp32Load x)
  let probability := unary M .exp (binary M .sub (cast M (cap M p scaled)) (M.fp32Load saved))
  let corrected := if (col : Int)=label then binary M .sub probability (literal M 1) else probability
  let derivative := if p.hasSoftcap then
    let t := capPartial M p scaled
    binary M .mul (backwardScale M p corrected)
      (binary M .sub (literal M 1) (binary M .mul t t))
    else backwardScale M p corrected
  cast M (if label = -100 then literal M 0 else
    binary M .mul (M.fp32Load upstream) derivative)

variable [Inhabited α]

theorem logits_binding (M : Algebra α) (p : Parameters) (s : State α) (chunk : Nat)
    (x : Fin p.block → α)
    (binding : ∀ lane : Fin p.block, chunk*p.block+lane.val<p.vocab →
      (s.mem p.logits (s.pids 0*p.logitsStride+(chunk*p.block+lane.val))).read .real=x lane) :
    KernelModel.forwardLogits M p s chunk=logits M p chunk x := by
  funext j
  rcases j with ⟨lane,unit⟩
  by_cases active : chunk*p.block+lane.val<p.vocab
  · simp [KernelModel.forwardLogits,loaded,logits,active,binding lane active]
  · simp [KernelModel.forwardLogits,loaded,logits,active]

theorem lse_binding (M : Algebra α) (p : Parameters) (s : State α) (chunk : Nat)
    (x : Fin p.block → α)
    (binding : ∀ lane : Fin p.block, chunk*p.block+lane.val<p.vocab →
      (s.mem p.logits (s.pids 0*p.logitsStride+(chunk*p.block+lane.val))).read .real=x lane) :
    lseValue M p s chunk=lse M p chunk x := by
  simp only [lseValue,lseTile,maxValue,maxTile,logits_binding M p s chunk x binding,lse,maximum]

theorem forward_loss_binding (M : Algebra α) (p : Parameters) (s : State α)
    (x : Fin p.block → α) (target : α)
    (binding : ∀ lane : Fin p.block, lane.val<p.vocab →
      (s.mem p.logits (s.pids 0*p.logitsStride+lane.val)).read .real=x lane)
    (gatherBinding : label p s ≠ -100 →
      (s.mem p.logits (s.pids 0*p.logitsStride+(label p s).toNat-(-(label p s)).toNat)).read .real=target) :
    KernelModel.forwardLoss M p s=forwardLoss M p x (label p s) target := by
  have values := lse_binding M p s 0 x (by simpa using binding)
  by_cases ignored : label p s = -100
  · simp [KernelModel.forwardLoss,forwardLoss,ignored]
  · simp [KernelModel.forwardLoss,forwardLoss,ignored,values,gathered,gatherBinding ignored]

theorem chunk_loss_binding (M : Algebra α) (p : Parameters) (s : State α) (target : α)
    (gatherBinding : label p s ≠ -100 →
      (s.mem p.logits (s.pids 0*p.logitsStride+(label p s).toNat-(-(label p s)).toNat)).read .real=target) :
    KernelModel.chunkLoss M p s=chunkLoss M p (label p s) target := by
  by_cases ignored : label p s = -100
  · simp [KernelModel.chunkLoss,chunkLoss,ignored]
  · simp [KernelModel.chunkLoss,chunkLoss,chunkTarget,ignored,gatherBinding ignored]

theorem gradient_binding (M : Algebra α) (p : Parameters) (s : State α)
    (lane : Fin p.block) (x saved upstream : α)
    (active : s.pids 1*p.block+lane.val<p.vocab)
    (input : (s.mem p.logits (s.pids 0*p.logitsStride+(s.pids 1*p.block+lane.val))).read .real=x)
    (savedBinding : (s.mem p.lse (s.pids 0)).read .real=saved)
    (upstreamBinding : label p s ≠ -100 →
      (s.mem p.upstream (s.pids 0*p.upstreamStride)).read .real=upstream) :
    gradientValue M p s lane=gradient M p (s.pids 1*p.block+lane.val) (label p s) x saved upstream := by
  by_cases ignored : label p s = -100
  · simp [gradientValue,gradient,ignored]
  · simp only [gradientValue,gradient,ignored,if_false,backwardDerivative,backwardCorrected,
      backwardProbability,upstreamValue,savedLSE,loaded,active,if_true,
      input,savedBinding,upstreamBinding ignored]

#specNonCircular logits avoiding [KernelSource.forward,KernelSource.chunked,KernelSource.backward,exec,run]
#specNonCircular gradient avoiding [KernelSource.forward,KernelSource.chunked,KernelSource.backward,exec,run]
#axiomsClean logits_binding
#axiomsClean lse_binding
#axiomsClean forward_loss_binding
#axiomsClean chunk_loss_binding
#axiomsClean gradient_binding
end
end VeriTile.Bench.Repairs.FastCeLoss.LogicalModel
