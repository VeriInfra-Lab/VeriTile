import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.KernelCompiler
import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.KernelModel

namespace VeriTile.Bench.Repairs.FastCeLoss.KernelFMA
open VeriTile.Triton FP.Structural KernelSource KernelModel
open VeriTile.Bench.Repairs.SoftmaxFlaggems.Native
noncomputable section
set_option maxHeartbeats 4000000

def capLiteral (p : Parameters) : Term := .literal (some .fp32) .real p.softcap

def forwardPartial (R : Reductions) (p : Parameters) (initial : State Term) : Term :=
  KernelCompiler.lower (capPartial (algebra R) p (forwardScale (algebra R) p (gathered (algebra R) p initial)))

def backwardPartial (R : Reductions) (p : Parameters) (initial : State Term) (lane : Fin p.block) : Term :=
  KernelCompiler.lower (capPartial (algebra R) p (backwardScale (algebra R) p
    (loaded (algebra R) p initial (initial.pids 1*p.block+lane.val))))

/-- Actual forward loss site: the source cast disappears before contraction
of `lse-cap*tanh(...)`. Its product operand carries the selected sign flip. -/
 theorem forward_softcap_fused (R : Reductions) (I : Instructions) (S : Schedule)
    (p : Parameters) (initial : State Term) (enabled : p.hasSoftcap=Bool.true)
    (active : label p initial≠ -100)
    (selected : S.fusion (.binary (some .fp32) .real .sub
      (shape (KernelCompiler.lower (lseValue (algebra R) p initial 0)))
      (.binary (some .fp32) .real .mul (shape (capLiteral p))
        (shape (forwardPartial R p initial))))=.negativeLeftFirst) :
    KernelCompiler.observe I S (KernelModel.cast (algebra R) (forwardLoss (algebra R) p initial))=
      I.fmaRN (KernelCompiler.eval I S (capLiteral p) ^^^ 0x80000000)
        (KernelCompiler.eval I S (forwardPartial R p initial))
        (KernelCompiler.eval I S (KernelCompiler.lower (lseValue (algebra R) p initial 0))) := by
  have graph : KernelCompiler.lower (KernelModel.cast (algebra R) (forwardLoss (algebra R) p initial))=
      .binary (some .fp32) .real .sub
        (KernelCompiler.lower (lseValue (algebra R) p initial 0))
        (.binary (some .fp32) .real .mul (capLiteral p) (forwardPartial R p initial)) := by
    simp only [forwardLoss,active,if_false,cap,enabled,if_true,KernelModel.cast,binary,literal,
      algebra,KernelCompiler.lower,capLiteral,forwardPartial]
  rw [KernelCompiler.observe,graph]
  exact KernelCompiler.subtract_product_fused I S _ _ _ selected

/-- Actual backward exponential argument: `cap*tanh(...)-savedLSE` contracts
with a negated addend. The exponential remains its own primitive boundary. -/
 theorem backward_probability_fused (R : Reductions) (I : Instructions) (S : Schedule)
    (p : Parameters) (initial : State Term) (lane : Fin p.block) (enabled : p.hasSoftcap=Bool.true)
    (selected : S.fusion (.binary (some .fp32) .real .sub
      (.binary (some .fp32) .real .mul (shape (capLiteral p))
        (shape (backwardPartial R p initial lane)))
      (shape (KernelCompiler.lower (savedLSE (algebra R) p initial))))=.left) :
    KernelCompiler.observe I S (backwardProbability (algebra R) p initial lane)=
      I.unary (some .fp32) .exp
        (I.fmaRN (KernelCompiler.eval I S (capLiteral p))
          (KernelCompiler.eval I S (backwardPartial R p initial lane))
          (KernelCompiler.eval I S (KernelCompiler.lower (savedLSE (algebra R) p initial)) ^^^ 0x80000000)) := by
  have graph : KernelCompiler.lower (backwardProbability (algebra R) p initial lane)=
      .unary (some .fp32) .exp (.binary (some .fp32) .real .sub
        (.binary (some .fp32) .real .mul (capLiteral p) (backwardPartial R p initial lane))
        (KernelCompiler.lower (savedLSE (algebra R) p initial))) := by
    simp only [backwardProbability,cap,enabled,if_true,KernelModel.cast,binary,unary,literal,
      algebra,KernelCompiler.lower,capLiteral,backwardPartial]
  rw [KernelCompiler.observe,graph]
  change I.unary _ _ (KernelCompiler.eval I S _)=_
  rw [KernelCompiler.product_subtract_fused I S _ _ _ selected]

#axiomsClean forward_softcap_fused
#axiomsClean backward_probability_fused
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.KernelFMA
