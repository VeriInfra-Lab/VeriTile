import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelScalar

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel.RealTarget
open VeriTile.Triton
open Kernel Scalar
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 4000000

def realModel (revBias : Bool) : NativeModel ℝ where
  biasReverse := revBias
  literal := id
  cast := fun _ _ => id
  add := (· + ·)
  mul := (· * ·)
  maximum := max
  exp := Real.exp
  log := Real.log
  minimum := min
  lessEqual := fun x y => decide (x ≤ y)
  fma := fun x y z => x*y+z
  unused := 0

theorem list_range_sum (f : Nat → ℝ) (n : Nat) :
    ((List.range n).map f).sum = ∑ j ∈ Finset.range n, f j := by
  induction n with
  | zero => simp
  | succ n ih => simp [List.range_succ,Finset.sum_range_succ,ih]

/-- The independent logical-coordinate formula. Padded times store zero dt;
the cumulative output continues the inclusive prefix through those lanes. -/
def prepared (c : Geometry) (a : LogicalInputs ℝ) (hasBias softplus : Bool)
    (lower : ℝ) (upper : UpperBound) (b h k j : Nat) : ℝ :=
  if h < c.heads ∧ j < c.chunk ∧ k*c.chunk+j < c.sequence then
    let x := a.dt b (k*c.chunk+j) h + if hasBias then a.bias h else 0
    upper.cap (max (if softplus then Math.softplus x else x) lower)
  else 0

def cumulative (c : Geometry) (a : LogicalInputs ℝ) (hasBias softplus : Bool)
    (lower : ℝ) (upper : UpperBound) (b h k j : Nat) : ℝ :=
  ∑ t ∈ Finset.range (j+1), prepared c a hasBias softplus lower upper b h k t * a.rate h

def upperWord : UpperBound → ℝ
  | .finite hi => hi
  | .posInf => 0

/-- Exact-Real precision erasure, including the default positive-infinite upper
clamp. The zero placeholder for +∞ never denotes a finite clamp operand. -/
def model (rev : Bool) (upper : UpperBound) : NativeModel ℝ :=
  match upper with
  | .finite _ => realModel rev
  | .posInf => { realModel rev with minimum := fun x _ => x }

theorem preparation (rev : Bool) (c : Geometry) (a : LogicalInputs ℝ)
    (hasBias softplus : Bool) (lower : ℝ) (upper : UpperBound) (b h k j : Nat) :
    preparedValue (model rev upper) c a hasBias softplus lower (upperWord upper) b h k j =
      prepared c a hasBias softplus lower upper b h k j := by
  by_cases active : h < c.heads ∧ j < c.chunk ∧ k*c.chunk+j < c.sequence
  · simp only [preparedValue,prepared,if_pos active]
    cases upper <;> cases rev <;> cases hasBias <;> cases softplus <;>
      norm_num [referencePrepared,model,realModel,upperWord,UpperBound.cap,Math.softplus,add_comm]
  · simp only [preparedValue,prepared,if_neg active]
    cases upper <;> norm_num [model,realModel]

theorem tree (rev : Bool) (upper : UpperBound) (f : Nat → ℝ) (rate : ℝ) (t : SumTree n) :
    referenceTree (model rev upper) (fun i => f i.val) rate t =
      (t.leaves.map (fun j => f j*rate)).sum := by
  induction t with
  | leaf i => cases upper <;> simp [referenceTree,SumTree.leaves,model,realModel]
  | add l r il ir =>
    cases upper <;> change referenceTree _ _ _ l + referenceTree _ _ _ r = _
    all_goals rw [il,ir]; simp [SumTree.leaves]
  | fused i r ih =>
    cases upper <;> change f i.val*rate + referenceTree _ _ _ r = _
    all_goals rw [ih]; simp [SumTree.leaves]

theorem cumulative_formula (rev : Bool) (c : Geometry) (plan : PrefixPlan c.BH c.BC)
    (a : LogicalInputs ℝ) (hasBias softplus : Bool) (lower : ℝ) (upper : UpperBound)
    (b h k : Nat) (j : Fin c.BC) :
    cumulativeValue (model rev upper) c plan a hasBias softplus lower (upperWord upper) b h k j =
      cumulative c a hasBias softplus lower upper b h k j.val := by
  unfold cumulativeValue
  simp only [preparation]
  have hc : (model rev upper).cast .fp32 .real (a.rate h) = a.rate h := by
    cases upper <;> rfl
  rw [hc,tree]
  have hp := (plan.leaves ⟨h%c.BH,Nat.mod_lt h c.headPositive⟩ j).map
    (fun t => prepared c a hasBias softplus lower upper b h k t * a.rate h)
  rw [hp.sum_eq,list_range_sum]
  rfl


#guard_msgs (drop info) in
#specNonCircular prepared avoiding [Kernel.launch,Scalar.interpret]
#guard_msgs (drop info) in
#specNonCircular cumulative avoiding [Kernel.launch,Scalar.interpret]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel.RealTarget
