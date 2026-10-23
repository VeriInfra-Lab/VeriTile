import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.RawObserved
import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.WholePublic

namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.RawObserved
open VeriTile.Triton FP.Structural Math Program Backend RawBackend
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000

structure InputWords (α : Type) where
  dt : Nat → α
  rate : Nat → α
  bias : Nat → α

/-- Only initial input-word bindings, not outputs or execution results. Source
port identities are typed views; an allocation adapter can map distinct input
views to the same physical input allocation. Outputs are fresh. -/
def InputGraph {α : Type} (r : Regions) (s : State (Term α)) (a : InputWords α) : Prop :=
  (∀ off, (s.mem r.dt off).read .fp32 = .input (a.dt off)) ∧
  (∀ off, (s.mem r.bias off).read .real = .input (a.bias off)) ∧
  (∀ off, (s.mem r.rate off).read .fp32 = .input (a.rate off))

def referenceLogicalPrepared (M : NativeModel α) (c : Whole.Host) (a : InputWords α)
    (hasBias useSoftplus : Bool) (lower upper : α) (b h k j : Nat) : α :=
  if h < c.heads ∧ j < c.chunk ∧ k*c.chunk+j < c.sequence then
    referencePrepared M hasBias useSoftplus lower upper Bool.true
      (a.dt (b*c.dtBatch+(k*c.chunk+j)*c.dtTime+h*c.dtHead)) (a.bias (h*c.biasHead))
  else M.literal 0.0

def referenceLogicalCumulative (M : NativeModel α) (c : Whole.Host) (plan : PrefixPlan c.BH c.BC)
    (a : InputWords α) (hasBias useSoftplus : Bool) (lower upper : α)
    (b h k : Nat) (j : Fin c.BC) : α :=
  referenceTree M (fun t => referenceLogicalPrepared M c a hasBias useSoftplus lower upper b h k t.val)
    (M.cast .fp32 .real (a.rate (h*c.rateHead)))
    (plan.tree ⟨h%c.BH,Nat.mod_lt h c.headPositive⟩ j)

theorem logical_preparation (M : NativeModel α) (c : Whole.Host) (plan : PrefixPlan c.BH c.BC)
    (r : Regions) (s : State (Term α)) (a : InputWords α) (read : InputGraph r s a)
    (hasBias useSoftplus : Bool) (lower upper : α) (b h k j : Nat) :
    interpret M (RawBackend.logicalPrepared c (primitives M c plan) r hasBias useSoftplus (.input lower) (.input upper) s b h k j) =
      referenceLogicalPrepared M c a hasBias useSoftplus lower upper b h k j := by
  unfold RawBackend.logicalPrepared referenceLogicalPrepared
  split_ifs
  · rw [read.1,read.2.1,source_preparation]
  · rfl

theorem logical_cumulative (M : NativeModel α) (c : Whole.Host) (plan : PrefixPlan c.BH c.BC)
    (r : Regions) (s : State (Term α)) (a : InputWords α) (read : InputGraph r s a)
    (hasBias useSoftplus : Bool) (lower upper : α) (b h k : Nat) (j : Fin c.BC) :
    interpret M (RawBackend.logicalCumulative c (primitives M c plan) r hasBias useSoftplus (.input lower) (.input upper) s b h k j) =
      referenceLogicalCumulative M c plan a hasBias useSoftplus lower upper b h k j := by
  simp only [RawBackend.logicalCumulative,primitives,graphAlgebra,source_scan,
    RawBackend.logicalRate,read.2.2,interpret,referenceLogicalCumulative]
  congr 1
  funext t
  exact logical_preparation M c plan r s a read hasBias useSoftplus lower upper b h k t.val

def LogicalOutputs (M : NativeModel α) (c : Whole.Host) (plan : PrefixPlan c.BH c.BC)
    (r : Regions) (a : InputWords α) (hasBias useSoftplus : Bool) (lower upper : α)
    (v : State (Term α)) : Prop :=
  ∀ b h k j, b < c.batches → h < c.heads → k < c.chunks → ∀ hj : j < c.chunk,
    interpret M ((v.mem r.out (Whole.packed c b h k j)).read .real) =
      referenceLogicalPrepared M c a hasBias useSoftplus lower upper b h k j ∧
    interpret M ((v.mem r.sum (Whole.packed c b h k j)).read .real) =
      referenceLogicalCumulative M c plan a hasBias useSoftplus lower upper b h k
        ⟨j,lt_of_lt_of_le hj c.chunk_le_BC⟩

theorem logical_outputs (M : NativeModel α) (c : Whole.Host) (plan : PrefixPlan c.BH c.BC)
    (r : Regions) (s : State (Term α)) (a : InputWords α) (read : InputGraph r s a)
    (hasBias useSoftplus : Bool) (lower upper : α) (v : State (Term α))
    (values : RawBackend.Values c (primitives M c plan) r hasBias useSoftplus (.input lower) (.input upper) s c.grid v) :
    LogicalOutputs M c plan r a hasBias useSoftplus lower upper v := by
  intro b h k j hb hh hk hj
  obtain ⟨l,hl,i,t,active,hb',hh',hk',hj'⟩ := Whole.cover c b h k j hb hh hk hj
  have val := values l hl (i,t,PUnit.unit) active
  have hout := congrArg (fun cell : Cell (Term α) => interpret M (cell.read .real)) val.1
  have hsum := congrArg (fun cell : Cell (Term α) => interpret M (cell.read .real)) val.2
  simp only [Cell.read_same,logical_preparation M c plan r s a read,logical_cumulative M c plan r s a read,
    Whole.out_packed,hb',hh',hk',hj'] at hout hsum
  have ht : t = (⟨j,lt_of_lt_of_le hj c.chunk_le_BC⟩ : Fin c.BC) := Fin.ext hj'
  rw [ht] at hsum
  exact ⟨hout,hsum⟩

structure HostResult (α : Type) where
  graph : State (Term α)
  first : RegionName
  second : RegionName
  shape : List Nat

def runHost (M : NativeModel α) (c : Whole.Host) (plan : PrefixPlan c.BH c.BC)
    (r : Regions) (hasBias useSoftplus : Bool) (lower upper : α)
    (ls : List Launch) (s : State (Term α)) : Option (HostResult α) :=
  (RawSource.runTiles (graphAlgebra M c plan) Term.minimum c (RawSource.source c r hasBias useSoftplus) (.input lower) (.input upper) ls s).map
    (fun v => ⟨v,r.sum,r.out,[c.batches,c.heads,c.chunks,c.chunk]⟩)

def HostCorrect (M : NativeModel α) (c : Whole.Host) (plan : PrefixPlan c.BH c.BC)
    (r : Regions) (a : InputWords α) (hasBias useSoftplus : Bool) (lower upper : α)
    (ls : List Launch) (s : State (Term α)) : Prop :=
  ∃ result, runHost M c plan r hasBias useSoftplus lower upper ls s = some result ∧
    result.first = r.sum ∧ result.second = r.out ∧ result.shape = [c.batches,c.heads,c.chunks,c.chunk] ∧
    LogicalOutputs M c plan r a hasBias useSoftplus lower upper result.graph ∧
    RawBackend.Frame c r c.grid s result.graph

/-- Original source-derived backend values for every host output, including
partial chunks and heads, arbitrary input strides, all scheduling orders,
and all untouched typed cells. No source/candidate equivalence is used. -/
theorem host_correct (M : NativeModel α) (c : Whole.Host) (plan : PrefixPlan c.BH c.BC)
    (r : Regions) (fresh : Whole.Fresh r) (s : State (Term α)) (a : InputWords α) (read : InputGraph r s a)
    (hasBias useSoftplus : Bool) (lower upper : α)
    (ls : List Launch) (schedule : ls.Perm c.grid) :
    HostCorrect M c plan r a hasBias useSoftplus lower upper ls s := by
  obtain ⟨v,ran,values,frame⟩ := RawBackend.grid_correct c (primitives M c plan) r fresh
    hasBias useSoftplus (.input lower) (.input upper) ls schedule s
  refine ⟨⟨v,r.sum,r.out,[c.batches,c.heads,c.chunks,c.chunk]⟩,?_,rfl,rfl,rfl,?_,frame⟩
  · simpa only [runHost,primitives,ran,Option.map_some] using congrArg
      (Option.map (fun v => HostResult.mk v r.sum r.out [c.batches,c.heads,c.chunks,c.chunk])) ran
  · exact logical_outputs M c plan r s a read hasBias useSoftplus lower upper v values

#guard_msgs (drop info) in
#specNonCircular referenceLogicalCumulative avoiding [interpret,RawBackend.sourceRun,runHost]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.RawObserved
