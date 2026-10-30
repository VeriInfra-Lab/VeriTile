import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.IO
import bench.repairs.tritonbench_g.lightning_attention.Proofs.TargetLoss

namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly
open VeriTile Triton FP.Structural
noncomputable section

/-- O[t,c] = Σ(u≤t) (Σa Q[t,a] K[u,a]) V[u,c]. -/
def forwardFormula (d : Host.Dimensions) (x : (forwardIO d).Inputs) :
    (forwardIO d).Outputs := fun ⟨0,_⟩ (h,t,c,()) =>
  Target.output (queryValues d x h) (keyValues d x h) (valueValues d x h) t c

/-- Independent coordinate VJP of L = Σ(t,c) G[t,c] O[t,c]. -/
def backwardFormula (d : Host.Dimensions) (x : (backwardIO d).Inputs) :
    (backwardIO d).Outputs := fun port => match port with
  | ⟨0,_⟩ => fun (h,t,a,()) =>
      Target.dq (keyValues d x h) (valueValues d x h) (incomingValues d x h) t a
  | ⟨1,_⟩ => fun (h,t,a,()) =>
      Target.dk (queryValues d x h) (valueValues d x h) (incomingValues d x h) t a
  | ⟨2,_⟩ => fun (h,t,c,()) =>
      Target.dv (queryValues d x h) (keyValues d x h) (incomingValues d x h) t c

/-- The returned logical gradients are actual coordinate derivatives of the
independent loss, not merely a separately supplied gradient formula. -/
def LogicalDerivatives (d : Host.Dimensions) : Prop :=
  ∀ x : (backwardIO d).Inputs, ∀ h : Fin (d.B*d.H),
    (∀ t a, HasDerivAt (fun ε => Target.loss
      (Target.bump (queryValues d x h) t a ε) (keyValues d x h)
      (valueValues d x h) (incomingValues d x h)) (backwardFormula d x (0 : Fin 3) (h,t,a,())) 0) ∧
    (∀ t a, HasDerivAt (fun ε => Target.loss
      (queryValues d x h) (Target.bump (keyValues d x h) t a ε)
      (valueValues d x h) (incomingValues d x h)) (backwardFormula d x (1 : Fin 3) (h,t,a,())) 0) ∧
    (∀ t c, HasDerivAt (fun ε => Target.loss
      (queryValues d x h) (keyValues d x h)
      (Target.bump (valueValues d x h) t c ε) (incomingValues d x h))
      (backwardFormula d x (2 : Fin 3) (h,t,c,())) 0)

#specNonCircular forwardFormula avoiding [FP.Structural.exec,FP.Structural.run,FP.Structural.runSchedule]
#specNonCircular backwardFormula avoiding [FP.Structural.exec,FP.Structural.run,FP.Structural.runSchedule]
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly
