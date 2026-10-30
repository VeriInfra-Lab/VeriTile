import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelRealAlgebra
import bench.tritonbench_g.fused_recurrent_delta.Proofs.TensorVJP
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelReal
open VeriTile.Triton FP.Structural KernelRegisters KernelBackwardRegisters
open scoped BigOperators
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 14000000

/-- Entry data for first-order backward: v contains the independently defined
primal residual, while q, k, beta and H0 retain their primal values. This is an
input relationship; the actual forward store theorem supplies the residual. -/
structure Saved (s : Shape) (b : KernelBackwardReference.Input s ℝ)
    (a : Fin s.BBV → Math.Input s.BK) (g : Nat → Fin s.BBV → ℝ) (terminal : Fin s.BBV → Math.Vec s.BK) : Prop where
  query : ∀ t i j,b.query t i=(a j).q t i
  key : ∀ t i j,b.key t i=(a j).k t i
  beta : ∀ t j,b.beta t j=(a j).beta t
  residual : ∀ t,t<s.T → ∀ j,b.value t j=Math.residual (Math.state (a j) t) ((a j).k t) ((a j).x t)
  incoming : ∀ t j,b.incoming t j=g t j
  initial : ∀ i j,b.initial (i,j,PUnit.unit)=(a j).h0 i
  terminal : ∀ i j,b.terminal (i,j,PUnit.unit)=terminal j i

theorem reverse_state (s : Shape) (scale : ℝ) (b : KernelBackwardReference.Input s ℝ)
    (a : Fin s.BBV → Math.Input s.BK) (g : Nat → Fin s.BBV → ℝ) (terminal : Fin s.BBV → Math.Vec s.BK)
    (saved : Saved s b a g terminal) (n : Nat) (j : Fin s.BBV) (i : Fin s.BK) :
    KernelBackwardReference.reverse algebra s scale b n (i,j,PUnit.unit)=
      Math.reverse scale (a j) (fun t => g t j) (terminal j) s.T n i := by
  induction n generalizing i with
  | zero => exact saved.terminal i j
  | succ n ih =>
    rw [KernelBackwardReference.reverse]
    unfold KernelBackwardReference.retreat KernelBackwardReference.dv KernelBackwardReference.rawDv
    rw [key_sum]
    change (KernelBackwardReference.enhance algebra s scale b (s.T-1-n)
      (KernelBackwardReference.reverse algebra s scale b n) (i,j,PUnit.unit))-
      b.key (s.T-1-n) i *
        ((∑ k : Fin s.BK,KernelBackwardReference.enhance algebra s scale b (s.T-1-n)
          (KernelBackwardReference.reverse algebra s scale b n) (k,j,PUnit.unit)*b.key (s.T-1-n) k)*b.beta (s.T-1-n) j)=_
    simp only [KernelBackwardReference.enhance,KernelReference.add,KernelReference.mul,KernelReference.scalar]
    change (KernelBackwardReference.reverse algebra s scale b n (i,j,PUnit.unit)+
      (b.query (s.T-1-n) i*scale)*b.incoming (s.T-1-n) j)-
      b.key (s.T-1-n) i *((∑ k : Fin s.BK,
        (KernelBackwardReference.reverse algebra s scale b n (k,j,PUnit.unit)+
          (b.query (s.T-1-n) k*scale)*b.incoming (s.T-1-n) j)*b.key (s.T-1-n) k)*b.beta (s.T-1-n) j)=_
    simp_rw [ih,saved.query _ _ j,saved.key _ _ j,saved.incoming,saved.beta]
    simp only [Math.reverse,Math.pullback,Math.dot]
    have terms : (fun k : Fin s.BK =>
        (Math.reverse scale (a j) (fun t => g t j) (terminal j) s.T n k+
          ((a j).q (s.T-1-n) k*scale)*g (s.T-1-n) j)*(a j).k (s.T-1-n) k)=
      (fun k : Fin s.BK =>
        (Math.reverse scale (a j) (fun t => g t j) (terminal j) s.T n k+
          scale*g (s.T-1-n) j*(a j).q (s.T-1-n) k)*(a j).k (s.T-1-n) k) := by funext k; ring
    rw [terms]
    ring

theorem recompute_state (s : Shape) (b : KernelBackwardReference.Input s ℝ)
    (a : Fin s.BBV → Math.Input s.BK) (g : Nat → Fin s.BBV → ℝ) (terminal : Fin s.BBV → Math.Vec s.BK)
    (saved : Saved s b a g terminal) (n : Nat) (bound : n≤s.T) (j : Fin s.BBV) (i : Fin s.BK) :
    KernelBackwardReference.forward algebra s b n (i,j,PUnit.unit)=Math.state (a j) n i := by
  induction n with
  | zero => exact saved.initial i j
  | succ n ih =>
    rw [KernelBackwardReference.forward]
    change KernelBackwardReference.forward algebra s b n (i,j,PUnit.unit)+
      b.key n i*(b.value n j*b.beta n j)=_
    rw [ih (by omega),saved.key _ _ j,saved.residual n (by omega),saved.beta]
    simp only [Math.state,Math.next]
    ring

def enhancedColumn (s : Shape) (scale : ℝ) (a : Math.Input s.BK) (g : Nat → ℝ)
    (terminal : Math.Vec s.BK) (t : Nat) : Math.Vec s.BK :=
  fun i => Math.reverse scale a g terminal s.T (s.T-1-t) i+scale*g t*a.q t i
theorem enhanced_column (s : Shape) (scale : ℝ) (b : KernelBackwardReference.Input s ℝ)
    (a : Fin s.BBV → Math.Input s.BK) (g : Nat → Fin s.BBV → ℝ) (terminal : Fin s.BBV → Math.Vec s.BK)
    (saved : Saved s b a g terminal) (t : Nat) (j : Fin s.BBV) (i : Fin s.BK) :
    KernelBackwardReference.enhance algebra s scale b t
      (KernelBackwardReference.reverse algebra s scale b (s.T-1-t)) (i,j,PUnit.unit)=
      enhancedColumn s scale (a j) (fun n => g n j) (terminal j) t i := by
  change KernelBackwardReference.reverse algebra s scale b (s.T-1-t) (i,j,PUnit.unit)+
    (b.query t i*scale)*b.incoming t j=_
  rw [reverse_state s scale b a g terminal saved,saved.query _ _ j,saved.incoming]
  unfold enhancedColumn
  ring

#axiomsClean reverse_state
#axiomsClean recompute_state
#axiomsClean enhanced_column
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelReal
