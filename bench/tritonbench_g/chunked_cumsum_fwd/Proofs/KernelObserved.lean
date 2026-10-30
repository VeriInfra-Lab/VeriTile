import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelGrid
import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelScalar

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel.Scalar
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

structure InputWords (α : Type) where
  dt : Nat → α
  rate : Nat → α
  bias : Nat → α

/-- Only initial input-word bindings, not outputs or execution results. Source
port identities are typed views; an allocation adapter can map distinct input
views to the same physical input allocation. Outputs are fresh. -/
def InputGraph {α : Type} (s : State (Term α)) (a : InputWords α) : Prop :=
  (∀ off, (s.mem "dt" off).read .fp32 = .input (a.dt off)) ∧
  (∀ off, (s.mem "bias" off).read .real = .input (a.bias off)) ∧
  (∀ off, (s.mem "rate" off).read .fp32 = .input (a.rate off))

def referenceLogicalPrepared (M : NativeModel α) (c : Geometry) (a : InputWords α)
    (hasBias useSoftplus : Bool) (lower upper : α) (b h k j : Nat) : α :=
  if h < c.heads ∧ j < c.chunk ∧ k*c.chunk+j < c.sequence then
    referencePrepared M hasBias useSoftplus lower upper Bool.true
      (a.dt (b*c.dtBatch+(k*c.chunk+j)*c.dtTime+h*c.dtHead)) (a.bias (h*c.biasHead))
  else M.literal 0.0

def referenceLogicalCumulative (M : NativeModel α) (c : Geometry) (plan : PrefixPlan c.BH c.BC)
    (a : InputWords α) (hasBias useSoftplus : Bool) (lower upper : α)
    (b h k : Nat) (j : Fin c.BC) : α :=
  referenceTree M (fun t => referenceLogicalPrepared M c a hasBias useSoftplus lower upper b h k t.val)
    (M.cast .fp32 .real (a.rate (h*c.rateHead)))
    (plan.tree ⟨h%c.BH,Nat.mod_lt h c.headPositive⟩ j)

theorem logical_preparation (M : NativeModel α) (c : Geometry) (plan : PrefixPlan c.BH c.BC)
    (s : State (Term α)) (a : InputWords α) (read : InputGraph s a)
    (hasBias useSoftplus : Bool) (lower upper : α) (b h k j : Nat) :
    interpret M (Kernel.logicalPrepared c (primitives M c plan) hasBias useSoftplus (.input lower) (.input upper) s b h k j) =
      referenceLogicalPrepared M c a hasBias useSoftplus lower upper b h k j := by
  unfold Kernel.logicalPrepared referenceLogicalPrepared
  split_ifs
  · rw [read.1,read.2.1,source_preparation]
  · rfl

theorem logical_cumulative (M : NativeModel α) (c : Geometry) (plan : PrefixPlan c.BH c.BC)
    (s : State (Term α)) (a : InputWords α) (read : InputGraph s a)
    (hasBias useSoftplus : Bool) (lower upper : α) (b h k : Nat) (j : Fin c.BC) :
    interpret M (Kernel.logicalCumulative c (primitives M c plan) hasBias useSoftplus (.input lower) (.input upper) s b h k j) =
      referenceLogicalCumulative M c plan a hasBias useSoftplus lower upper b h k j := by
  simp only [Kernel.logicalCumulative,primitives,graphAlgebra,source_scan,
    Kernel.logicalRate,read.2.2,interpret,referenceLogicalCumulative]
  congr 1
  funext t
  exact logical_preparation M c plan s a read hasBias useSoftplus lower upper b h k t.val

def LogicalOutputs (M : NativeModel α) (c : Geometry) (plan : PrefixPlan c.BH c.BC)
    (a : InputWords α) (hasBias useSoftplus : Bool) (lower upper : α)
    (v : State (Term α)) : Prop :=
  ∀ b h k j, b < c.batches → h < c.heads → k < c.chunks → ∀ hj : j < c.chunk,
    interpret M ((v.mem "out" (Kernel.packed c b h k j)).read .real) =
      referenceLogicalPrepared M c a hasBias useSoftplus lower upper b h k j ∧
    interpret M ((v.mem "sum" (Kernel.packed c b h k j)).read .real) =
      referenceLogicalCumulative M c plan a hasBias useSoftplus lower upper b h k
        ⟨j,lt_of_lt_of_le hj c.chunk_le_BC⟩

theorem logical_outputs (M : NativeModel α) (c : Geometry) (plan : PrefixPlan c.BH c.BC)
    (s : State (Term α)) (a : InputWords α) (read : InputGraph s a)
    (hasBias useSoftplus : Bool) (lower upper : α) (v : State (Term α))
    (values : Kernel.Values c (primitives M c plan) hasBias useSoftplus (.input lower) (.input upper) s c.grid v) :
    LogicalOutputs M c plan a hasBias useSoftplus lower upper v := by
  intro b h k j hb hh hk hj
  obtain ⟨l,hl,i,t,active,hb',hh',hk',hj'⟩ := Kernel.cover c b h k j hb hh hk hj
  have val := values l hl (i,t,PUnit.unit) active
  have hout := congrArg (fun cell : Cell (Term α) => interpret M (cell.read .real)) val.1
  have hsum := congrArg (fun cell : Cell (Term α) => interpret M (cell.read .real)) val.2
  simp only [Cell.read_same,logical_preparation M c plan s a read,logical_cumulative M c plan s a read,
    Kernel.out_packed,hb',hh',hk',hj'] at hout hsum
  have ht : t = (⟨j,lt_of_lt_of_le hj c.chunk_le_BC⟩ : Fin c.BC) := Fin.ext hj'
  rw [ht] at hsum
  exact ⟨hout,hsum⟩


/-- Materialize input graph leaves directly from the supplied word reader. -/
def initial (a : InputWords α) : State (Term α) where
  mem := fun r off => if r="dt" then .mk .fp32 (.input (a.dt off))
    else if r="rate" then .mk .fp32 (.input (a.rate off))
    else if r="bias" then .mk .real (.input (a.bias off)) else .mk .real .unused
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

theorem initial_graph (a : InputWords α) : InputGraph (initial a) a := by
  refine ⟨?_,?_,?_⟩ <;> intro off <;> simp [initial,Cell.read]

def logicalInputs (c : Geometry) (a : InputWords α) : LogicalInputs α where
  dt := fun b t h => a.dt (b*c.dtBatch+t*c.dtTime+h*c.dtHead)
  rate := fun h => a.rate (h*c.rateHead)
  bias := fun h => a.bias (h*c.biasHead)

theorem reference_prepared (M : NativeModel α) (c : Geometry) (a : InputWords α)
    (hasBias useSoftplus : Bool) (lower upper : α) (b h k j : Nat) :
    referenceLogicalPrepared M c a hasBias useSoftplus lower upper b h k j =
      preparedValue M c (logicalInputs c a) hasBias useSoftplus lower upper b h k j := by
  have hzero : (0.0 : ℝ) = 0 := by norm_num
  simp only [referenceLogicalPrepared,preparedValue,logicalInputs,hzero]

theorem reference_cumulative (M : NativeModel α) (c : Geometry) (plan : PrefixPlan c.BH c.BC)
    (a : InputWords α) (hasBias useSoftplus : Bool) (lower upper : α)
    (b h k : Nat) (j : Fin c.BC) :
    referenceLogicalCumulative M c plan a hasBias useSoftplus lower upper b h k j =
      cumulativeValue M c plan (logicalInputs c a) hasBias useSoftplus lower upper b h k j := by
  simp only [referenceLogicalCumulative,cumulativeValue,reference_prepared,logicalInputs]

/-- The full source launch produces both independent word recurrences. -/
theorem complete_from (M : NativeModel α) (c : Geometry) (sizes : Sizes c)
    (plan : PrefixPlan c.BH c.BC) (a : InputWords α) (s : State (Term α))
    (read : InputGraph s a) (hasBias useSoftplus : Bool) (lower upper : α) :
    ∃ v, Kernel.launch (graphAlgebra M c plan) c hasBias useSoftplus (.input lower) (.input upper)
        s = some v ∧ LogicalOutputs M c plan a hasBias useSoftplus lower upper v ∧
      Kernel.Frame c c.grid s v := by
  obtain ⟨v,ran,values,frame⟩ := Kernel.grid_correct c (primitives M c plan) sizes
    hasBias useSoftplus (.input lower) (.input upper) c.grid (List.Perm.refl _) s
  exact ⟨v,ran,logical_outputs M c plan s a read
    hasBias useSoftplus lower upper v values,frame⟩

theorem complete (M : NativeModel α) (c : Geometry) (sizes : Sizes c)
    (plan : PrefixPlan c.BH c.BC) (a : InputWords α)
    (hasBias useSoftplus : Bool) (lower upper : α) :
    ∃ v, Kernel.launch (graphAlgebra M c plan) c hasBias useSoftplus (.input lower) (.input upper)
        (initial a) = some v ∧ LogicalOutputs M c plan a hasBias useSoftplus lower upper v ∧
      Kernel.Frame c c.grid (initial a) v := by
  obtain ⟨v,ran,values,frame⟩ := Kernel.grid_correct c (primitives M c plan) sizes
    hasBias useSoftplus (.input lower) (.input upper) c.grid (List.Perm.refl _) (initial a)
  exact ⟨v,ran,logical_outputs M c plan (initial a) a (initial_graph a)
    hasBias useSoftplus lower upper v values,frame⟩

#guard_msgs (drop info) in
#specNonCircular referenceLogicalPrepared avoiding [interpret,Kernel.sourceRun,Kernel.launch]
#guard_msgs (drop info) in
#specNonCircular referenceLogicalCumulative avoiding [interpret,Kernel.sourceRun,Kernel.launch]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel.Scalar
