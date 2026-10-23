import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.Native
import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.RawHostFacade

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.HostModel
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd
open RawBoundary RawObserved Backend
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000

structure Logical (α : Type) where
  dt : Nat → Nat → Nat → α
  rate : Nat → α
  bias : Nat → α

structure NegativeFlags where
  dt : Bool
  rate : Bool
  bias : Bool

/-- Caller logical coordinates, including lazy negative flags. Real conjugation
is the identity. These definitions never inspect a candidate result. -/
def logicalInputs (call : Call α) (flags : NegativeFlags) (neg : α → α)
    (entry : PhysicalMemory α) : Logical α where
  dt := fun b t h =>
    let x := call.dt.view.read entry
      (b*call.dt.stride 0+t*call.dt.stride 1+h*call.dt.stride 2)
    if flags.dt then neg x else x
  rate := fun h =>
    let x := call.rate.view.read entry (h*call.rate.stride 0)
    if flags.rate then neg x else x
  bias := fun h =>
    let x := call.biasView.read entry (h*call.biasStride)
    if flags.bias then neg x else x

theorem integer_ceil (n d : Nat) (hd : 0 < d) :
    (- (-(n : Int) / (d : Int))).toNat = Whole.ceilDiv n d := by
  rw [Int.neg_ediv,HostBoundary.ceilDiv_remainder n d hd]
  have hp : (0 : Int) < d := by omega
  have hs : (d : Int).sign = 1 := Int.sign_eq_one_of_pos hp
  have hq : 0 ≤ (n : Int) / (d : Int) := Int.ediv_nonneg (by omega) (by omega)
  by_cases hz : n % d = 0
  · simp [Int.natCast_dvd_natCast,Nat.dvd_iff_mod_eq_zero,← Int.natCast_ediv,hz,hs] <;> omega
  · simp [Int.natCast_dvd_natCast,Nat.dvd_iff_mod_eq_zero,← Int.natCast_ediv,hz,hs] <;> omega

/-- Positive integer chunk geometry, computed with exact Python integer ceiling. -/
structure Geometry (call : Call α) where
  host : Whole.Host
  shape : call.dt.shape = [host.batches,host.sequence,host.heads]
  chunk : host.chunk = call.chunk.toNat
  biasShape : ∀ bias, call.bias = some bias → bias.shape = [host.heads]
  rateShape : call.rate.shape = [host.heads]

def geometry (call : Call α) : Except BoundaryFailure (Geometry call) :=
  match shape : call.dt.shape with
  | [b,s,h] =>
    if rate : call.rate.shape ≠ [h] then .error .rateAssertion
    else if bias : ∃ t, call.bias = some t ∧ t.shape ≠ [h] then .error .biasAssertion
    else if zero : call.chunk = 0 then .error .zeroDivision
    else if neg : call.chunk < 0 then .error .negativeAllocation
    else .ok ⟨makeHost call b s h (by omega),shape,rfl,
      by simpa using bias,by simpa using rate⟩
  | _ => .error .rank

theorem geometry_failure (call : Call α) (reason : BoundaryFailure)
    (failed : geometry call = .error reason) : BoundarySpec call reason := by
  unfold geometry at failed
  split at failed
  · rename_i b s h shape
    split_ifs at failed with rate bias zero neg
    · cases failed; exact ⟨b,s,h,shape,rate⟩
    · cases failed; exact ⟨b,s,h,shape,by simpa using rate,bias⟩
    · cases failed
      exact ⟨b,s,h,shape,by simpa using rate,by simpa using bias,zero⟩
    · cases failed
      exact ⟨b,s,h,shape,by simpa using rate,by simpa using bias,neg⟩
  · cases failed
    simp_all [BoundarySpec]

/-- Actual two fresh torch.empty objects, before source launch. -/
structure AllocationResult (α : Type) (call : Call α) (c : Whole.Host)
    (entry : PhysicalMemory α) (old : Nat → Prop) extends RawBoundary.Allocated α call c entry old where
  outObject : Nat
  sumObject : Nat
  objectsDistinct : outObject ≠ sumObject

/-- Direct input observation after the two output allocations. The original
Tensor objects, strides and heap are retained. Only input readback and ordinary
allocation frames occur here; there is no copy, resolution, or output law. -/
structure Prepared (call : Call α) (g : Geometry call) (logical : Logical α)
    (a : AllocationResult α call g.host entry old) where
  dt : RawBoundary.Tensor
  rate : RawBoundary.Tensor
  bias : Option RawBoundary.Tensor
  memory : PhysicalMemory α
  dtShape : dt.shape = call.dt.shape
  rateShape : rate.shape = call.rate.shape
  biasPresent : bias.isSome = call.bias.isSome
  biasShape : ∀ t, bias = some t → t.shape = [g.host.heads]
  dtDevice : dt.device = call.dt.device
  rateDevice : rate.device = call.rate.device
  biasDevice : ∀ t, bias = some t → ∃ original, call.bias = some original ∧ t.device = original.device
  kept : Nat → Prop
  callerProtected : ∀ allocation, old allocation → kept allocation
  dtProtected : kept dt.view.allocation
  rateProtected : kept rate.view.allocation
  biasProtected : kept ((bias.map RawBoundary.Tensor.view).getD dt.view).allocation
  outFresh : ¬ kept a.out.allocation
  sumFresh : ¬ kept a.sum.allocation
  frame : ∀ allocation off, old allocation → memory allocation off = a.memory allocation off
  dtValues : ∀ b t h, b < g.host.batches → t < g.host.sequence → h < g.host.heads →
    dt.view.read memory (b*dt.stride 0+t*dt.stride 1+h*dt.stride 2) = logical.dt b t h
  rateValues : ∀ h, h < g.host.heads → rate.view.read memory (h*rate.stride 0) = logical.rate h
  biasValues : ∀ t, bias = some t → ∀ h, h < g.host.heads →
    t.view.read memory (h*t.stride 0) = logical.bias h

  dtOriginal : dt=call.dt
  rateOriginal : rate=call.rate
  biasOriginal : bias=call.bias
  memoryOriginal : memory=a.memory

variable {α : Type} {call : Call α} {entry : PhysicalMemory α} {old : Nat → Prop}
  {g : Geometry call} {logical : Logical α} {a : AllocationResult α call g.host entry old}

def Prepared.host (p : Prepared call g logical a) : Whole.Host :=
  { g.host with
    dtBatch := p.dt.stride 0
    dtTime := p.dt.stride 1
    dtHead := p.dt.stride 2
    rateHead := p.rate.stride 0
    biasHead := (p.bias.map (fun t => t.stride 0)).getD 0 }

def Prepared.views (p : Prepared call g logical a) : Views :=
  ⟨p.dt.view,p.rate.view,(p.bias.map RawBoundary.Tensor.view).getD p.dt.view,a.out,a.sum,a.distinct⟩

theorem Prepared.fresh (p : Prepared call g logical a) : FreshPhysical p.views p.kept :=
  ⟨p.dtProtected,p.rateProtected,p.biasProtected,p.outFresh,p.sumFresh⟩

/-- Independent coordinate-level FP expression, with the selected actual
ordered native prefix tree. All ten scalar ports remain explicit. -/
def preparedValue (M : NativeModel α) (c : Whole.Host) (logical : Logical α)
    (hasBias softplus : Bool) (lower upper : α) (b h k j : Nat) : α :=
  if h < c.heads ∧ j < c.chunk ∧ k*c.chunk+j < c.sequence then
    referencePrepared M hasBias softplus lower upper Bool.true
      (logical.dt b (k*c.chunk+j) h) (logical.bias h)
  else M.literal 0.0

def cumulativeValue (M : NativeModel α) (c : Whole.Host) (plan : PrefixPlan c.BH c.BC)
    (logical : Logical α) (hasBias softplus : Bool) (lower upper : α)
    (b h k : Nat) (j : Fin c.BC) : α :=
  referenceTree M (fun t => preparedValue M c logical hasBias softplus lower upper b h k t.val)
    (M.cast .fp32 .real (logical.rate h))
    (plan.tree ⟨h%c.BH,Nat.mod_lt h c.headPositive⟩ j)

theorem Prepared.preparation (p : Prepared call g logical a) (block : HeadBlock)
    (M : NativeModel α) (b h k j : Nat) (hb : b < g.host.batches) :
    referenceLogicalPrepared M (reblock p.host block) (p.views.words p.memory)
      call.hasBias call.softplus call.lower call.upper b h k j =
    preparedValue M (reblock p.host block) logical call.hasBias call.softplus call.lower call.upper b h k j := by
  unfold referenceLogicalPrepared preparedValue
  split_ifs with active
  · have dt := p.dtValues b (k*g.host.chunk+j) h hb active.2.2 active.1
    simp only [Views.words,Prepared.views,Prepared.host,reblock] at *
    rw [dt]
    cases hbias : p.bias with
    | none =>
      have no : call.hasBias = Bool.false := by simpa [Call.hasBias,hbias] using p.biasPresent.symm
      simp [no,referencePrepared]
    | some bias =>
      have bi := p.biasValues bias hbias h active.1
      simp only [hbias,Option.map_some,Option.getD_some] at *
      rw [bi]
  · rfl

theorem Prepared.cumulative (p : Prepared call g logical a) (block : HeadBlock)
    (M : NativeModel α) (plan : PrefixPlan block.size p.host.BC)
    (b h k : Nat) (j : Fin p.host.BC) (hb : b < g.host.batches) (hh : h < g.host.heads) :
    referenceLogicalCumulative M (reblock p.host block) plan (p.views.words p.memory)
      call.hasBias call.softplus call.lower call.upper b h k j =
    cumulativeValue M (reblock p.host block) plan logical call.hasBias call.softplus call.lower call.upper b h k j := by
  unfold referenceLogicalCumulative cumulativeValue
  have rate := p.rateValues h hh
  simp only [Views.words,Prepared.views,Prepared.host,reblock] at rate ⊢
  rw [rate]
  congr 1
  funext t
  exact p.preparation block M b h k t.val hb

structure Plans (α : Type) (c : Whole.Host) where
  trials : List (LaunchPlan α c)
  selected : LaunchPlan α c

structure Runtime (call : Call α) (logical : Logical α) (entry : PhysicalMemory α) (old : Nat → Prop) where
  allocate : (g : Geometry call) → Except AllocationFailure (AllocationResult α call g.host entry old)
  inputs : (g : Geometry call) → (a : AllocationResult α call g.host entry old) →
    Prepared call g logical a
  setup : (g : Geometry call) → (a : AllocationResult α call g.host entry old) →
    (p : Prepared call g logical a) → Except String (Plans α p.host)

inductive Outcome (α : Type) where
  | boundaryFailure (reason : BoundaryFailure)
  | allocationFailure (reason : AllocationFailure)
  | setupFailure (reason : String)
  | kernelFailure
  | returned (firstObject secondObject : Nat) (first second : View) (shape : List Nat) (device : Nat) (memory : PhysicalMemory α)

def execute (call : Call α) (logical : Logical α) (rt : Runtime call logical entry old) : Outcome α :=
  match geometry call with
  | .error reason => .boundaryFailure reason
  | .ok g => match rt.allocate g with
    | .error reason => .allocationFailure reason
    | .ok a =>
      let p := rt.inputs g a
      match rt.setup g a p with
      | .error reason => .setupFailure reason
      | .ok plans => match tunedRun p.views p.host call.hasBias call.softplus call.lower call.upper
          plans.trials plans.selected p.memory with
        | none => .kernelFailure
        | some final => .returned a.sumObject a.outObject a.sum a.out
            [g.host.batches,g.host.heads,g.host.chunks,g.host.chunk] call.dt.device final

def Values (call : Call α) (logical : Logical α) {g : Geometry call}
    {a : AllocationResult α call g.host entry old} (p : Prepared call g logical a)
    (plans : Plans α p.host) (final : PhysicalMemory α) : Prop :=
  ∀ b h k j, b < g.host.batches → h < g.host.heads → k < g.host.chunks → ∀ hj : j < g.host.chunk,
    p.views.out.read final (Whole.packed g.host b h k j) =
      preparedValue plans.selected.model (reblock p.host plans.selected.block) logical
        call.hasBias call.softplus call.lower call.upper b h k j ∧
    p.views.sum.read final (Whole.packed g.host b h k j) =
      cumulativeValue plans.selected.model (reblock p.host plans.selected.block) plans.selected.scan logical
        call.hasBias call.softplus call.lower call.upper b h k ⟨j,lt_of_lt_of_le hj g.host.chunk_le_BC⟩

def Success (call : Call α) (logical : Logical α) (rt : Runtime call logical entry old)
    (outcome : Outcome α) : Prop :=
  ∃ g a p plans final, geometry call = .ok g ∧ rt.allocate g = .ok a ∧ rt.inputs g a = p ∧
    rt.setup g a p = .ok plans ∧ outcome = .returned a.sumObject a.outObject a.sum a.out
      [g.host.batches,g.host.heads,g.host.chunks,g.host.chunk] call.dt.device final ∧
    Values call logical p plans final ∧
    (∀ allocation off, ¬ p.views.written p.host allocation off → final allocation off = p.memory allocation off) ∧
    (∀ allocation off, old allocation → final allocation off = entry allocation off)

def HostSpec (call : Call α) (logical : Logical α) (rt : Runtime call logical entry old)
    (outcome : Outcome α) : Prop :=
  (∃ reason, geometry call = .error reason ∧ BoundarySpec call reason ∧ outcome = .boundaryFailure reason) ∨
  (∃ g reason, geometry call = .ok g ∧ rt.allocate g = .error reason ∧ outcome = .allocationFailure reason) ∨
  (∃ g a p reason, geometry call = .ok g ∧ rt.allocate g = .ok a ∧ rt.inputs g a = p ∧
    rt.setup g a p = .error reason ∧ outcome = .setupFailure reason) ∨ Success call logical rt outcome

/-- Full source execution is a conclusion. All autotune trials and the selected
complete grid are composed by the existing unchanged proof. -/
theorem outcomes (call : Call α) (logical : Logical α) (rt : Runtime call logical entry old) :
    HostSpec call logical rt (execute call logical rt) := by
  cases hg : geometry call with
  | error reason => exact Or.inl ⟨reason,hg,geometry_failure call reason hg,by simp [execute,hg]⟩
  | ok g =>
    cases ha : rt.allocate g with
    | error reason => exact Or.inr (Or.inl ⟨g,reason,hg,ha,by simp [execute,hg,ha]⟩)
    | ok a =>
      let p := rt.inputs g a
      cases ht : rt.setup g a p with
      | error reason => exact Or.inr (Or.inr (Or.inl ⟨g,a,p,reason,hg,ha,rfl,ht,by simp [execute,hg,ha,ht,p]⟩))
      | ok plans =>
        obtain ⟨final,ran,values,frame,keptFrame⟩ := tuned_correct p.views p.host p.fresh
          call.hasBias call.softplus call.lower call.upper plans.trials plans.selected p.memory
        apply Or.inr;apply Or.inr;apply Or.inr
        refine ⟨g,a,p,plans,final,hg,ha,rfl,ht,?_,?_,frame,?_⟩
        · simp [execute,hg,ha,ht,ran,p]
        · intro b h k j hb hh hk hj
          have val := values b h k j hb hh hk hj
          rw [p.preparation plans.selected.block plans.selected.model b h k j hb,
            p.cumulative plans.selected.block plans.selected.model plans.selected.scan b h k _ hb hh] at val
          exact val
        · intro allocation off hold
          exact (keptFrame allocation off (p.callerProtected allocation hold)).trans
            ((p.frame allocation off hold).trans (a.frame allocation off hold))

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.HostModel
