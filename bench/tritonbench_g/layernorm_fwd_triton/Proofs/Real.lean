import bench.tritonbench_g.layernorm_fwd_triton.Proofs.Model
import bench.tritonbench_g.layernorm_fwd_triton.Proofs.Mathematical_LayernormFwdTriton

/-! Independent mathematical interpretation. The physical Real inputs and
epsilon are arbitrary; neither is obtained by decoding native FP words.
Only the existing full-N lane-partition algebra is reused from the legacy
monolith. No legacy host-trusted execution headline is used. -/
namespace VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
open VeriTile.Bench.TritonBenchG.LayernormFwdTriton
namespace RealProof
noncomputable section
open scoped Classical
set_option maxHeartbeats 2000000

/-- The logical specification is the usual full feature-axis mean/variance. -/
def meanSpec (N : Nat) (x : Nat → ℝ) : ℝ := (∑ j : Fin N, x j.val)/(N:ℝ)
def varianceSpec (N : Nat) (x : Nat → ℝ) : ℝ :=
  (∑ j : Fin N, (x j.val-meanSpec N x)^2)/(N:ℝ)
def outputSpec (N : Nat) (x w : Nat → ℝ) (eps : ℝ) (col : Nat) : ℝ :=
  ((x col-meanSpec N x)*(Real.sqrt (varianceSpec N x+eps))⁻¹)*w col

def ops : Native.Ops ℝ where
  zero := 0
  one := 1
  count := fun n => n
  add := (·+·)
  sub := (·-·)
  mul := (·*·)
  div := (·/·)
  fma := fun a b c => a*b+c
  sqrt := Real.sqrt

def virtual (x : Nat → ℝ) : BlockState :=
  { (default : BlockState) with mem := fun _ j => MemCell.real (x j) }

def lanePrefix (x : Nat → ℝ) (N off : Nat) (j : Fin 128) : ℝ :=
  layernormMeanLanePrefix (virtual x) "X" 0 0 N 128 off j

@[simp] theorem virtual_read (x : Nat → ℝ) (j : Nat) :
    (virtual x).readMem "X" (xColOffset (virtual x) 0 0 j)=x j := by
  simp [virtual,xColOffset,BlockState.readMem]

@[simp] theorem prefix_zero (x : Nat → ℝ) (N : Nat) : lanePrefix x N 0=(fun _ => 0) := by
  funext j
  exact layernormMeanLanePrefix_zero (virtual x) "X" 0 0 N 128 j

theorem prefix_step (x : Nat → ℝ) (N off : Nat) (h : off%128=0) (j : Fin 128) :
    lanePrefix x N (off+128) j=lanePrefix x N off j+(if off+j.val<N then x (off+j.val) else 0) := by
  simpa only [lanePrefix,virtual_read] using
    layernormMeanLanePrefix_step (virtual x) "X" 0 0 N 128 off j h

def addStep (x : Nat → ℝ) (N off : Nat) (a : Fin 128 → ℝ) : Fin 128 → ℝ :=
  fun j => a j+(if off+j.val<N then x (off+j.val) else 0)

theorem fold_prefix (x : Nat → ℝ) (N off : Nat) (h : off%128=0) :
    ∃ final, N≤final ∧
      foldRange (addStep x N) off N 128 (lanePrefix x N off)=lanePrefix x N final := by
  by_cases hc : off<N
  · rw [foldRange]
    simp only [show (128:Nat)≠0 by decide,hc,if_false,if_true]
    have he : addStep x N off (lanePrefix x N off)=lanePrefix x N (off+128) := by
      funext j
      exact (prefix_step x N off h j).symm
    rw [he]
    exact fold_prefix x N (off+128) (layernormLoopOffset_mod_step off 128 h)
  · exact ⟨off,by omega,by simp [foldRange,hc]⟩
termination_by N-off
decreasing_by omega

theorem prefix_total (x : Nat → ℝ) (N off : Nat) (h : N≤off) :
    (∑ j : Fin 128, lanePrefix x N off j)=∑ j : Fin N, x j.val := by
  have ht := layernormMeanAccumulatorSpec_final_sum_eq_fullN
    (virtual x) "X" 0 0 N 128 off (by decide) h
  simpa only [layernormMeanAccumulatorSpec,lanePrefix,WithBot.unbotD_coe,virtual_read] using ht

theorem accumulation_total (x : Nat → ℝ) (N : Nat) :
    (∑ j : Fin 128, foldRange (addStep x N) 0 N 128 (fun _ => 0) j)=∑ j : Fin N, x j.val := by
  obtain ⟨final,hfinal,he⟩ := fold_prefix x N 0 (by decide)
  rw [prefix_zero] at he
  rw [he]
  exact prefix_total x N final hfinal

theorem ordered_add (swap : Bool) (a b : ℝ) : Native.ordered swap (·+·) a b=a+b := by
  cases swap <;> simp [Native.ordered,add_comm]
theorem ordered_mul (swap : Bool) (a b : ℝ) : Native.ordered swap (·*·) a b=a*b := by
  cases swap <;> simp [Native.ordered,mul_comm]

theorem divideCount_real (p : Native.Plan N) (a : ℝ) : Native.divideCount ops p a=a/(N:ℝ) := by
  unfold Native.divideCount
  split
  · rename_i h
    rw [p.eraseUnitValid h]
    simp
  · rfl

theorem tree_total (tree : Native.SumTree)
    (h : tree.leaves.Perm (List.finRange 128)) (x : Fin 128 → ℝ) :
    tree.eval (·+·) x=∑ j, x j := by
  rw [VeriTile.Bench.TritonBenchG.LayerNormFwd.Whole.sumTree_real,(h.map x).sum_eq]
  rw [← List.ofFn_eq_map,List.sum_ofFn]

theorem mean_real (p : Native.Plan N) (x : Nat → ℝ) : Native.mean ops p x=meanSpec N x := by
  have step : Native.meanStep ops p x=addStep x N := by
    funext off a j
    simp [Native.meanStep,Native.maskedX,Native.cast32,ops,ordered_add,addStep]
  rw [Native.mean,divideCount_real]
  change p.meanTree.eval (·+·) (Native.meanAcc ops p x)/(N:ℝ)=_
  rw [tree_total _ p.meanComplete]
  rw [Native.meanAcc,step]
  change (∑ j : Fin 128, foldRange (addStep x N) 0 N 128 (fun _ => 0) j)/(N:ℝ)=_
  rw [accumulation_total]
  rfl

theorem carry_real (recipe : Native.Carry) (z acc : ℝ) :
    Native.carry ops recipe z acc=acc+z^2 := by
  cases recipe with
  | fused => simp [Native.carry,ops,pow_two,add_comm]
  | separate swap => simp [Native.carry,ops,ordered_add,pow_two]

theorem variance_real (p : Native.Plan N) (x : Nat → ℝ) (mu : ℝ) :
    Native.variance ops p x mu=(∑ j : Fin N, (x j.val-mu)^2)/(N:ℝ) := by
  have step : Native.varianceStep ops p x mu=addStep (fun j => (x j-mu)^2) N := by
    funext off a j
    simp only [Native.varianceStep,carry_real]
    by_cases h : off+j.val<N
    · simp [Native.centered,Native.maskedX,Native.cast32,ops,h,addStep]
    · simp [Native.centered,ops,h,addStep]
  rw [Native.variance,divideCount_real]
  change p.varianceTree.eval (·+·) (Native.varianceAcc ops p x mu)/(N:ℝ)=_
  rw [tree_total _ p.varianceComplete]
  rw [Native.varianceAcc,step]
  change (∑ j : Fin 128, foldRange (addStep (fun j => (x j-mu)^2) N) 0 N 128 (fun _ => 0) j)/(N:ℝ)=_
  rw [accumulation_total]

/-- Real correctness of the complete three-pass numerical interpretation,
including N=0 and every scalar eps. No FP-to-Real equation is a premise. -/
theorem reference_real (p : Native.Plan N) (x w : Nat → ℝ) (eps : ℝ) (col : Nat) :
    Native.reference ops p x w eps col=outputSpec N x w eps col := by
  simp only [Native.reference,mean_real,variance_real]
  simp [Native.output,Native.inverseStd,Native.cast32,ops,ordered_add,ordered_mul,
    outputSpec,varianceSpec,one_div]

#axiomsClean accumulation_total
#axiomsClean mean_real
#axiomsClean variance_real
#axiomsClean reference_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end RealProof
end VeriTile.Bench.Repairs.LayernormFwdTriton
