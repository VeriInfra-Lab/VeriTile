import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelBackwardReference
import bench.tritonbench_g.fused_recurrent_delta.Proofs.Mathematics
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelReal
open VeriTile.Triton FP.Structural KernelRegisters KernelBackwardRegisters
open scoped BigOperators
noncomputable section

def sum {shape : TileShape} (ax : Fin shape.length) (keep : Bool) (x : Values ℝ .real shape) :
    Values ℝ .real (TileShape.reduceShape shape ax keep) := by
  cases keep
  · exact fun out => ∑ k : Fin (TileShape.axisDim shape ax),x (TileShape.insertAxisIndex shape ax out k)
  · exact fun out => ∑ k : Fin (TileShape.axisDim shape ax),x (TileShape.replaceAxisIndex shape ax out k)
def algebra : Algebra ℝ where
  literal := fun _ _ x => x
  negInf := 0
  binary := fun _ _ op a b => match op with | .add => a+b | .sub => a-b | .mul => a*b | _ => 0
  unary := fun _ _ _ => 0
  cast := fun _ _ _ x => x
  fromNat := fun _ x => x
  fromInt := fun _ x => x
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => 0
  reduceSum := fun _ {_} => sum

theorem forward_sum (s : Shape) (x : ForwardTile s ℝ) (j : Fin s.FBV) :
    KernelReference.keySum algebra s x j=∑ i : Fin s.BK,x (j,i,PUnit.unit) := rfl
theorem key_sum (s : Shape) (x : Tile s ℝ) (j : Fin s.BBV) :
    KernelBackwardReference.keySum algebra s x j=∑ i : Fin s.BK,x (i,j,PUnit.unit) := rfl
theorem value_sum (s : Shape) (x : Tile s ℝ) (i : Fin s.BK) :
    KernelBackwardReference.valueSum algebra s x i=∑ j : Fin s.BBV,x (i,j,PUnit.unit) := rfl
theorem beta_sum (s : Shape) (x : Fin s.BBV → ℝ) :
    KernelBackwardReference.betaSum algebra s x=∑ j : Fin s.BBV,x j := rfl

/-- The real specification keeps a full zero-padded mathematical column.
The source execution is not used to define this input or recurrence. -/
def column (s : Shape) (f : Flags) (bh iv : Nat) (u : State ℝ) (j : Fin s.FBV) : Math.Input s.BK where
  q := fun t i => if i.val<s.K then (u.mem "q" (keyAddress s bh t i.val)).read .real else 0
  k := fun t i => if i.val<s.K then (u.mem "k" (keyAddress s bh t i.val)).read .real else 0
  x := fun t => if iv*s.FBV+j.val<s.V then (u.mem "v" (valAddress s bh iv s.FBV t j.val)).read .real else 0
  beta := fun t => if f.vectorBeta then
    if iv*s.FBV+j.val<s.V then (u.mem "beta" (valAddress s bh iv s.FBV t j.val)).read .real else 0
    else (u.mem "beta" (betaAddress s bh t)).read .real
  h0 := fun i => if f.initial && decide (i.val<s.K) && decide (iv*s.FBV+j.val<s.V) then
    (u.mem "h0" (stateAddress s bh iv s.FBV i.val j.val)).read .real else 0

theorem forward_state (s : Shape) (f : Flags) (bh iv : Nat) (u : State ℝ)
    (n : Nat) (j : Fin s.FBV) (i : Fin s.BK) :
    KernelReference.forward algebra s (KernelReference.forwardInput algebra s f bh iv u) n (j,i,PUnit.unit)=
      Math.state (column s f bh iv u j) n i := by
  induction n generalizing i with
  | zero =>
    simp [KernelReference.forward,KernelReference.forwardInput,KernelReference.add,KernelReference.zero,
      KernelReference.read,KernelReference.cast,algebra,Math.state,column]
    all_goals split <;> simp_all
  | succ n ih =>
    simp only [KernelReference.forward,KernelReference.keySum,algebra,sum]
    change KernelReference.forward algebra s (KernelReference.forwardInput algebra s f bh iv u) n (j,i,PUnit.unit)+
      (KernelReference.forwardInput algebra s f bh iv u).key n i *
        (((KernelReference.forwardInput algebra s f bh iv u).value n j -
          ∑ k : Fin s.BK,KernelReference.forward algebra s (KernelReference.forwardInput algebra s f bh iv u) n (j,k,PUnit.unit)*
            (KernelReference.forwardInput algebra s f bh iv u).key n k)*
          (KernelReference.forwardInput algebra s f bh iv u).beta n j)=_
    simp only [ih,Math.state,Math.next,Math.residual,Math.dot]
    simp [KernelReference.forwardInput,KernelReference.read,KernelReference.cast,KernelReference.zero,algebra,column]
    ring

theorem forward_residual (s : Shape) (f : Flags) (bh iv : Nat) (u : State ℝ) (t : Nat) (j : Fin s.FBV) :
    KernelReference.residual algebra s (KernelReference.forwardInput algebra s f bh iv u) t j=
      Math.residual (Math.state (column s f bh iv u j) t) ((column s f bh iv u j).k t)
        ((column s f bh iv u j).x t) := by
  unfold KernelReference.residual
  rw [forward_sum]
  simp_rw [KernelReal.forward_state s f bh iv u]
  simp [KernelReference.sub,KernelReference.mul,KernelReference.forwardInput,
    KernelReference.read,KernelReference.cast,KernelReference.zero,algebra,Math.residual,Math.dot,column]

theorem forward_output (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) (u : State ℝ) (t : Nat) (j : Fin s.FBV) :
    KernelReference.output algebra s (KernelReference.forwardInput algebra s f bh iv u) scale t j=
      Math.output scale (column s f bh iv u j) t := by
  unfold KernelReference.output
  rw [forward_sum]
  simp_rw [KernelReal.forward_state s f bh iv u]
  simp only [KernelReference.mul,KernelReference.scalar]
  change (∑ i : Fin s.BK, Math.state (column s f bh iv u j) (t+1) i *
    ((KernelReference.forwardInput algebra s f bh iv u).query t i * scale))=_
  simp only [Math.output,Math.dot,Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  simp only [KernelReference.forwardInput,KernelReference.read,KernelReference.cast,KernelReference.zero,
    algebra,column,decide_eq_true_eq]
  ring

#axiomsClean forward_state
#axiomsClean forward_residual
#axiomsClean forward_output
#specNonCircular column avoiding [KernelProgram.source,FP.Structural.exec,FP.Structural.run,KernelReference.forwardInput]
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelReal
