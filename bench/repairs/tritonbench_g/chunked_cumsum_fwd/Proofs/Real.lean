import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.Host
import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.RawObservedReal

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.RealProof
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd
open RawBoundary RawObserved Backend HostModel
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 4000000

/-- The independent logical-coordinate formula. Padded times store zero dt;
the cumulative output continues the inclusive prefix through those lanes. -/
def prepared (c : Whole.Host) (a : Logical ℝ) (hasBias softplus : Bool)
    (lower : ℝ) (upper : UpperBound) (b h k j : Nat) : ℝ :=
  if h < c.heads ∧ j < c.chunk ∧ k*c.chunk+j < c.sequence then
    let x := a.dt b (k*c.chunk+j) h + if hasBias then a.bias h else 0
    upper.cap (max (if softplus then Math.softplus x else x) lower)
  else 0

def cumulative (c : Whole.Host) (a : Logical ℝ) (hasBias softplus : Bool)
    (lower : ℝ) (upper : UpperBound) (b h k j : Nat) : ℝ :=
  ∑ t ∈ Finset.range (j+1), prepared c a hasBias softplus lower upper b h k t * a.rate h

def upperWord : UpperBound → ℝ
  | .finite hi => hi
  | .posInf => 0

/-- Exact-Real precision erasure, including the default positive-infinite upper
clamp. The zero placeholder for +∞ never denotes a finite clamp operand. -/
def model (rev : Bool) (upper : UpperBound) : NativeModel ℝ :=
  match upper with
  | .finite _ => RawObserved.realModel rev
  | .posInf => { RawObserved.realModel rev with minimum := fun x _ => x }

theorem preparation (rev : Bool) (c : Whole.Host) (a : Logical ℝ)
    (hasBias softplus : Bool) (lower : ℝ) (upper : UpperBound) (b h k j : Nat) :
    preparedValue (model rev upper) c a hasBias softplus lower (upperWord upper) b h k j =
      prepared c a hasBias softplus lower upper b h k j := by
  by_cases active : h < c.heads ∧ j < c.chunk ∧ k*c.chunk+j < c.sequence
  · simp only [preparedValue,prepared,if_pos active]
    cases upper <;> cases rev <;> cases hasBias <;> cases softplus <;>
      norm_num [referencePrepared,model,RawObserved.realModel,upperWord,UpperBound.cap,Math.softplus,add_comm]
  · simp only [preparedValue,prepared,if_neg active]
    cases upper <;> norm_num [model,RawObserved.realModel]

theorem tree (rev : Bool) (upper : UpperBound) (f : Nat → ℝ) (rate : ℝ) (t : SumTree n) :
    referenceTree (model rev upper) (fun i => f i.val) rate t =
      (t.leaves.map (fun j => f j*rate)).sum := by
  induction t with
  | leaf i => cases upper <;> simp [referenceTree,SumTree.leaves,model,RawObserved.realModel]
  | add l r il ir =>
    cases upper <;> change referenceTree _ _ _ l + referenceTree _ _ _ r = _
    all_goals rw [il,ir]; simp [SumTree.leaves]
  | fused i r ih =>
    cases upper <;> change f i.val*rate + referenceTree _ _ _ r = _
    all_goals rw [ih]; simp [SumTree.leaves]

theorem cumulative_formula (rev : Bool) (c : Whole.Host) (plan : PrefixPlan c.BH c.BC)
    (a : Logical ℝ) (hasBias softplus : Bool) (lower : ℝ) (upper : UpperBound)
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
  rw [hp.sum_eq,RawObserved.list_range_sum]
  rfl

variable {call : Call ℝ} {entry : PhysicalMemory ℝ} {old : Nat → Prop}
  {logical : Logical ℝ} {g : HostModel.Geometry call}
  {a : AllocationResult ℝ call g.host entry old}

def Values (call : Call ℝ) (upper : UpperBound) (logical : Logical ℝ)
    {g : HostModel.Geometry call} {a : AllocationResult ℝ call g.host entry old}
    (p : Prepared call g logical a) (final : PhysicalMemory ℝ) : Prop :=
  ∀ b h k j, b < g.host.batches → h < g.host.heads → k < g.host.chunks → j < g.host.chunk →
    p.views.out.read final (Whole.packed g.host b h k j) =
      prepared g.host logical call.hasBias call.softplus call.lower upper b h k j ∧
    p.views.sum.read final (Whole.packed g.host b h k j) =
      cumulative g.host logical call.hasBias call.softplus call.lower upper b h k j

theorem values (upper : UpperBound) (binding : call.upper = upperWord upper)
    (p : Prepared call g logical a) (plans : Plans ℝ p.host) (rev : Bool)
    (hm : plans.selected.model = model rev upper) (final : PhysicalMemory ℝ)
    (val : HostModel.Values call logical p plans final) : Values call upper logical p final := by
  intro b h k j hb hh hk hj
  have v := val b h k j hb hh hk hj
  rw [hm,binding,preparation,cumulative_formula] at v
  exact v

def RealRuntime (upper : UpperBound) (rt : HostModel.Runtime call logical entry old) : Prop :=
  ∀ g a p plans, rt.setup g a p = .ok plans → ∃ rev, plans.selected.model = model rev upper

/-- Every actual returned pair has the independent formulas and whole caller
frame. Failures remain exactly the source-ordered host outcomes. -/
def HostSpec (call : Call ℝ) (upper : UpperBound) (logical : Logical ℝ)
    (rt : HostModel.Runtime call logical entry old) (outcome : HostModel.Outcome ℝ) : Prop :=
  HostModel.HostSpec call logical rt outcome ∧
  (HostModel.Success call logical rt outcome → ∃ g a p plans final,
    HostModel.geometry call = .ok g ∧ rt.allocate g = .ok a ∧ rt.inputs g a = p ∧
    rt.setup g a p = .ok plans ∧ outcome = .returned a.sumObject a.outObject a.sum a.out
      [g.host.batches,g.host.heads,g.host.chunks,g.host.chunk] call.dt.device final ∧
    Values call upper logical p final ∧
    (∀ allocation off, ¬ p.views.written p.host allocation off → final allocation off = p.memory allocation off) ∧
    (∀ allocation off, old allocation → final allocation off = entry allocation off))

theorem outcomes (call : Call ℝ) (upper : UpperBound) (binding : call.upper = upperWord upper)
    (logical : Logical ℝ) (rt : HostModel.Runtime call logical entry old) (real : RealRuntime upper rt) :
    HostSpec call upper logical rt (HostModel.execute call logical rt) := by
  refine ⟨HostModel.outcomes call logical rt,?_⟩
  rintro ⟨g,a,p,plans,final,hg,ha,hp,ht,returned,val,frame,caller⟩
  obtain ⟨rev,hm⟩ := real g a p plans ht
  exact ⟨g,a,p,plans,final,hg,ha,hp,ht,returned,values upper binding p plans rev hm final val,frame,caller⟩

#guard_msgs (drop info) in
#specNonCircular prepared avoiding [HostModel.execute,RawSource.exec,RawObserved.interpret]
#guard_msgs (drop info) in
#specNonCircular cumulative avoiding [HostModel.execute,RawSource.exec,RawObserved.interpret]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.RealProof
