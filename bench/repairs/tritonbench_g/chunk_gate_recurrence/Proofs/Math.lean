import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.Base
namespace VeriTile.Bench.Repairs.ChunkGateRecurrence
open scoped BigOperators
noncomputable section
namespace MathTarget
namespace Old
export VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealMath
  (Inputs initial forward forward_zero forward_step reverse reverse_end reverse_step)
end Old

def scalar (c : Shape) (a : Inputs ℝ) (p j : Nat) : Old.Inputs 1 1 :=
  ⟨fun t _ _ => a.x p t j,fun t => a.gate (p/c.tiles) t,
    fun _ _ => a.seed p j,fun t _ _ => a.incoming p t j⟩

/-- Independent exclusive prefix formula, using only caller inputs. -/
def forward (c : Shape) (a : Inputs ℝ) (seed : Bool) (p t j : Nat) : ℝ :=
  (if seed then a.seed p j else 0)*(∏ u∈Finset.range t, a.gate (p/c.tiles) u)+
    ∑ u∈Finset.range t, a.x p u j * ∏ v∈Finset.Ico (u+1) t, a.gate (p/c.tiles) v

/-- Independent adjoint suffix. It does not mention any source execution or
stored intermediate. The terminal E_N is zero, including N=0. -/
def reverse (c : Shape) (a : Inputs ℝ) (p t j : Nat) : ℝ :=
  ∑ u∈Finset.Ico t c.N, a.incoming p u j * ∏ v∈Finset.Ico t u, a.gate (p/c.tiles) v

def dgate (c : Shape) (a : Inputs ℝ) (seed : Bool) (h t : Nat) : ℝ :=
  ∑ q : Fin c.tiles, ∑ j : Fin 1024,
    reverse c a (h*c.tiles+q) (t+1) j * forward c a seed (h*c.tiles+q) t j

@[simp] theorem forward_zero (c : Shape) (a : Inputs ℝ) (seed : Bool) (p j : Nat) :
    forward c a seed p 0 j=(if seed then a.seed p j else 0) := by simp [forward]

theorem forward_step (c : Shape) (a : Inputs ℝ) (seed : Bool) (p t j : Nat) :
    forward c a seed p (t+1) j =
      forward c a seed p t j*a.gate (p/c.tiles) t+a.x p t j := by
  simpa [Old.forward,Old.initial,scalar,forward] using Old.forward_step (scalar c a p j) seed t 0 0

@[simp] theorem reverse_end (c : Shape) (a : Inputs ℝ) (p j : Nat) : reverse c a p c.N j=0 := by
  simp [reverse]

theorem reverse_step (c : Shape) (a : Inputs ℝ) (p t j : Nat) (ht : t<c.N) :
    reverse c a p t j=reverse c a p (t+1) j*a.gate (p/c.tiles) t+a.incoming p t j := by
  simpa [Old.reverse,scalar,reverse] using Old.reverse_step (scalar c a p j) c.N t ht 0 0

/-- Exact first-order variation of the polynomial recurrence. -/
def tangent (c : Shape) (a : Inputs ℝ) (seed : Bool) (p j : Nat)
    (ds : ℝ) (dx dg : Nat → ℝ) : Nat → ℝ
  | 0 => if seed then ds else 0
  | t+1 => tangent c a seed p j ds dx dg t*a.gate (p/c.tiles) t+
      forward c a seed p t j*dg t+dx t

theorem telescope (f : Nat → ℝ) (n : Nat) :
    (∑ t∈Finset.range n, (f t-f (t+1)))=f 0-f n := by
  induction n with
  | zero => simp
  | succ n ih => rw [Finset.sum_range_succ,ih]; ring

/-- Symbolic adjoint identity for arbitrary perturbations, not a bounded test.
Together with coordinate_coverage, summing this identity covers every K/V
coordinate and gives the gate contraction, input gradient and seed gradient. -/
theorem adjoint_pairing (c : Shape) (a : Inputs ℝ) (seed : Bool) (p j : Nat)
    (ds : ℝ) (dx dg : Nat → ℝ) :
    (∑ t∈Finset.range c.N, a.incoming p t j*tangent c a seed p j ds dx dg t) =
      reverse c a p 0 j*(if seed then ds else 0)+
      ∑ t∈Finset.range c.N, (reverse c a p (t+1) j*dx t+
        reverse c a p (t+1) j*forward c a seed p t j*dg t) := by
  let z := tangent c a seed p j ds dx dg
  have localId (t : Nat) (ht : t<c.N) :
      a.incoming p t j*z t =
      (reverse c a p t j*z t-reverse c a p (t+1) j*z (t+1))+
      (reverse c a p (t+1) j*dx t+reverse c a p (t+1) j*forward c a seed p t j*dg t) := by
    rw [reverse_step c a p t j ht]
    dsimp only [z,tangent]
    ring
  calc
    _ = ∑ t∈Finset.range c.N, ((reverse c a p t j*z t-reverse c a p (t+1) j*z (t+1))+
      (reverse c a p (t+1) j*dx t+reverse c a p (t+1) j*forward c a seed p t j*dg t)) := by
        apply Finset.sum_congr rfl
        intro t ht
        exact localId t (Finset.mem_range.mp ht)
    _ = _ := by rw [Finset.sum_add_distrib,telescope]; simp [z,tangent]
end MathTarget

namespace RealProof
theorem forward (c : Shape) (a : Inputs ℝ) (seed : Bool) (p t j : Nat) :
    Reference.forward realOps c a seed p t j=MathTarget.forward c a seed p t j := by
  induction t with
  | zero => simp [Reference.forward,realOps]
  | succ t ih => rw [Reference.forward,MathTarget.forward_step]; change _* _+_= _; rw [ih]

theorem reverse (c : Shape) (a : Inputs ℝ) (p n j : Nat) (hn : n≤c.N) :
    Reference.reverse realOps c a p n j=MathTarget.reverse c a p (c.N-n) j := by
  induction n with
  | zero => simp [Reference.reverse,realOps]
  | succ n ih =>
    have hindex : c.N-(n+1)=c.N-1-n := by omega
    have hnext : c.N-n=(c.N-1-n)+1 := by omega
    rw [Reference.reverse,MathTarget.reverse_step c a p (c.N-(n+1)) j (by omega)]
    change _* _+_= _
    rw [ih (by omega),hindex,hnext]

theorem dx (c : Shape) (a : Inputs ℝ) (p t j : Nat) (ht : t<c.N) :
    Reference.dx realOps c a p t j=MathTarget.reverse c a p (t+1) j := by
  unfold Reference.dx
  split_ifs with h
  · rw [reverse c a p _ j (by omega),show c.N-(c.N-1-t)=t+1 by omega]
  · rw [show t+1=c.N by omega,MathTarget.reverse_end]
    rfl

theorem tileGate (c : Shape) (trees : Trees c) (a : Inputs ℝ) (seed : Bool) (p t : Nat) (ht : t<c.N) :
    Reference.tileGate realOps c trees a seed p t=
      ∑ j : Fin 1024, MathTarget.reverse c a p (t+1) j*MathTarget.forward c a seed p t j := by
  unfold Reference.tileGate
  split_ifs with h
  · rw [Reduction.product_real]
    apply Finset.sum_congr rfl
    intro j _
    rw [reverse c a p _ j (by omega),show c.N-(c.N-1-t)=t+1 by omega,forward]
  · simp [show t+1=c.N by omega,realOps]

theorem dgate (c : Shape) (trees : Trees c) (a : Inputs ℝ) (seed : Bool) (h t : Nat) (ht : t<c.N) :
    Reference.dgate realOps c trees a seed h t=MathTarget.dgate c a seed h t := by
  unfold Reference.dgate MathTarget.dgate
  rw [Reduction.real]
  apply Finset.sum_congr rfl
  intro q _
  exact tileGate c trees a seed _ t ht
end RealProof
#specNonCircular MathTarget.forward avoiding [Reference.forward,Memory.tile]
#specNonCircular MathTarget.reverse avoiding [Reference.reverse]
#axiomsClean MathTarget.adjoint_pairing
#axiomsClean RealProof.forward
#axiomsClean RealProof.reverse
#axiomsClean RealProof.dgate
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence
