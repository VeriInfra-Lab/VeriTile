import bench.repairs.tritonbench_g.cross_entropy2.Proofs.ScalarPrograms

namespace VeriTile.Bench.Repairs.CrossEntropy2.Floating
open Source
noncomputable section
set_option maxHeartbeats 10000000

/-- Native backward scalar graph. The two flags record local register
operand choices; both source order and the commuted PTX variants are covered.
An ignored result selects the literal zero after the arithmetic graph. -/
def nativeGradient (n : Ports) (o : Host.Options) (target : Int) (logit L upstream : Word)
    (col : Nat) (commutedFMA commutedOutput : Bool) : Word :=
  let probability := n.exp (n.sub (n.mul (n.literal o.logitScale) logit) L)
  let correction := n.mul (n.add (n.literal o.squareScale) (n.literal o.squareScale)) L
  let corrected := if commutedFMA then n.fma probability correction probability else n.fma correction probability probability
  let hit := (col : Int)=target-o.start
  let adjusted := if o.hasSmoothing then
    n.sub (if hit then n.sub corrected (n.sub (n.literal 1) (n.literal o.smoothing)) else corrected)
      (n.div (n.literal o.smoothing) (n.natural o.total))
    else if hit then n.add corrected (n.literal (-1)) else corrected
  let scaledUpstream := n.mul (n.literal o.logitScale) upstream
  if target=o.ignored then n.literal 0 else
    if commutedOutput then n.mul adjusted scaledUpstream else n.mul scaledUpstream adjusted

theorem gradient_native (n : Ports) (profile : Profile) (laws : Peepholes n) (one : ConstantOne n)
    (o : Host.Options) (target : Int) (logit L upstream : Expr) (col : Nat)
    (commutedFMA commutedOutput : Bool) :
    word n profile (Host.gradientScalar algebra o target logit L upstream col)=
      nativeGradient n o target (word n profile logit) (word n profile L) (word n profile upstream)
        col commutedFMA commutedOutput := by
  change n.natural 1=n.literal 1 at one
  have logitComm := laws.mulComm (evaluate n profile logit).value (n.literal o.logitScale)
  have upstreamComm := laws.mulComm (evaluate n profile upstream).value (n.literal o.logitScale)
  by_cases ignored : target=o.ignored
  · simp only [Host.gradientScalar,nativeGradient,ignored,↓reduceIte,word,algebra,evaluate]
  · cases smoothing : o.hasSmoothing <;> by_cases hit : (col : Int)=target-o.start <;>
      cases commutedFMA <;> cases commutedOutput
    all_goals simp only [Host.gradientScalar,nativeGradient,ignored,smoothing,hit,↓reduceIte,
      Bool.false_eq_true,word,algebra,evaluate,add,sub]
    all_goals simp only [one,laws.twice,laws.minusOne,logitComm,upstreamComm]
    all_goals try rw [laws.fmaComm (n.mul (n.add _ _) _) _]
    all_goals first | rfl | exact laws.mulComm _ _

#axiomsClean gradient_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Floating
