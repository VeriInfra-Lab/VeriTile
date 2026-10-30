import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.KernelMath
import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Spec

/-! The mathematical objective is the local VJP of the gated chunk primal.
H and dH are actual entry-memory inputs. Rows and keys outside the logical
chunk are zero-padded; the gate derivative is retained separately per key tile.
Summing key tiles and converting cumulative-gate derivatives by reverse cumsum
are separate kernels. -/
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.KernelObjective
open Whole KernelMath
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

def lastRow (c : Config) (cta : CTA c) : Fin 64 :=
  ⟨min 64 (c.T-cta.time.val*64)-1,by have := Nat.min_le_left 64 (c.T-cta.time.val*64); omega⟩
def input (c : Config) (x : Spec.Tensors) (cta : CTA c) : Input 64 c.BK where
  query r e := Spec.qk c x.query cta.head.val cta.time.val r.val (cta.key.val*c.BK+e.val)
  key r e := Spec.qk c x.key cta.head.val cta.time.val r.val (cta.key.val*c.BK+e.val)
  gate r := Spec.gate c x cta.head.val cta.time.val r.val
def coefficients (c : Config) (x : Spec.Tensors) (cta : CTA c) : Coefficients 64 c.BK where
  incoming r e := (∑ j ∈ Finset.range c.V,
    Spec.vd c x.outputGrad cta.head.val cta.time.val r.val j *
      Spec.hist c x.history cta.head.val cta.time.val (cta.key.val*c.BK+e.val) j)*c.scale
  outgoing r e := ∑ j ∈ Finset.range c.V,
    Spec.vd c x.value cta.head.val cta.time.val r.val j *
      Spec.hist c x.historyGrad cta.head.val cta.time.val (cta.key.val*c.BK+e.val) j
  score r u := if u.val≤r.val then (∑ j ∈ Finset.range c.V,
    Spec.vd c x.outputGrad cta.head.val cta.time.val r.val j *
      Spec.vd c x.value cta.head.val cta.time.val u.val j)*c.scale else 0
  history e := ∑ j ∈ Finset.range c.V,
    Spec.hist c x.history cta.head.val cta.time.val (cta.key.val*c.BK+e.val) j *
      Spec.hist c x.historyGrad cta.head.val cta.time.val (cta.key.val*c.BK+e.val) j

def objective (c : Config) (x : Spec.Tensors) (cta : CTA c) (a : Input 64 c.BK) : ℝ :=
  loss (coefficients c x cta) a (lastRow c cta)

theorem gate_last (c : Config) (x : Spec.Tensors) (cta : CTA c) :
    (input c x cta).gate (lastRow c cta)=Spec.lastGate c x cta.head.val cta.time.val := by
  have start := time_start_lt c cta
  have bound : cta.time.val*64+(lastRow c cta).val<c.T := by simp only [lastRow]; omega
  have eq : cta.time.val*64+(lastRow c cta).val=min (cta.time.val*64+64) c.T-1 := by
    simp only [lastRow]; omega
  change (if cta.time.val*64+(lastRow c cta).val<c.T then _ else _)=_
  rw [if_pos bound]
  exact congrArg (x.gate cta.head.val) eq

theorem bases (c : Config) (x : Spec.Tensors) (cta : CTA c) (r : Fin 64) (e : Fin c.BK) :
    baseQ (coefficients c x cta) (input c x cta) r e=
      Spec.baseQ c x cta.head.val cta.time.val r.val (cta.key.val*c.BK+e.val) ∧
    baseK (coefficients c x cta) (input c x cta) (lastRow c cta) r e=
      Spec.baseK c x cta.head.val cta.time.val r.val (cta.key.val*c.BK+e.val) := by
  constructor
  · simp only [baseQ,coefficients,input,Spec.baseQ]
    ring
  · unfold baseK
    rw [gate_last]
    simp only [coefficients,input,Spec.baseK,zero_sub,sub_eq_add_neg,zero_add]
    rw [add_comm (-(Spec.gate c x cta.head.val cta.time.val r.val))]
    ring
theorem scores (c : Config) (x : Spec.Tensors) (cta : CTA c) (r u : Fin 64) :
    score (coefficients c x cta) (input c x cta) r u=
      Spec.score c x cta.head.val cta.time.val r.val u.val := by
  simp only [score,coefficients,input,Spec.score]
  split_ifs <;> ring
theorem gradients_qk (c : Config) (x : Spec.Tensors) (cta : CTA c) (r : Fin 64) (e : Fin c.BK) :
    dq (coefficients c x cta) (input c x cta) r e=
      Spec.dq c x cta.head.val cta.time.val r.val (cta.key.val*c.BK+e.val) ∧
    dk (coefficients c x cta) (input c x cta) (lastRow c cta) r e=
      Spec.dk c x cta.head.val cta.time.val r.val (cta.key.val*c.BK+e.val) := by
  constructor
  · rw [dq,Spec.dq,(bases c x cta r e).1]
    congr 1
    apply Finset.sum_congr rfl
    intro u _
    rw [scores]
    rfl
  · rw [dk,Spec.dk,(bases c x cta r e).2]
    congr 1
    apply Finset.sum_congr rfl
    intro u _
    rw [scores]
    rfl
theorem correction_eq (c : Config) (x : Spec.Tensors) (cta : CTA c) :
    correction (coefficients c x cta) (input c x cta) (lastRow c cta)=
      Spec.correction c x cta.key.val cta.head.val cta.time.val := by
  rw [correction,gate_last,Spec.correction]
  apply congrArg₂ (fun a b : ℝ=>a+b)
  · apply congrArg (fun a : ℝ=>a*Real.exp (Spec.lastGate c x cta.head.val cta.time.val))
    dsimp only [coefficients]
    exact Finset.sum_comm
  · apply Finset.sum_congr rfl
    intro r _
    apply Finset.sum_congr rfl
    intro e _
    rw [(bases c x cta r e).2]
    rfl
theorem gradient_g (c : Config) (x : Spec.Tensors) (cta : CTA c) (r : Fin 64) (hr : activeG c cta r) :
    dg (coefficients c x cta) (input c x cta) (lastRow c cta) r=
      Spec.dg c x cta.key.val cta.head.val cta.time.val r.val := by
  have bound : r.val≤(lastRow c cta).val := by
    have rr := r.isLt
    simp only [lastRow]
    unfold activeG at hr
    omega
  have eq : (r=lastRow c cta) ↔ ¬ r.val<min 64 (c.T-cta.time.val*64)-1 := by
    simp only [Fin.ext_iff,lastRow] at bound ⊢
    omega
  rw [dg,Spec.dg]
  apply congrArg₂ (fun a b : ℝ=>a+b)
  · apply Finset.sum_congr rfl
    intro e _
    rw [(gradients_qk c x cta r e).1,(gradients_qk c x cta r e).2]
    rfl
  · rw [correction_eq]
    split_ifs <;> simp_all [lastRow]

/-- No source execution or computed gradient is a premise of this theorem. -/
theorem objective_gradient (c : Config) (x : Spec.Tensors) (cta : CTA c) (direction : Input 64 c.BK) :
    HasDerivAt (fun z=>objective c x cta (perturb (input c x cta) direction z))
      (pairing (coefficients c x cta) (input c x cta) direction (lastRow c cta)) 0 :=
  gradients _ _ _ _

#axiomsClean gradients_qk
#axiomsClean gradient_g
#axiomsClean objective_gradient
#specNonCircular objective avoiding [KernelMath.dq,KernelMath.dk,KernelMath.dg,Whole.runGrid]
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.KernelObjective
