import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.RawAutotune
import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.WholeHostCeil

namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.RawBoundary
open VeriTile.Triton FP.Structural Math Program RawObserved Backend RawBackend
noncomputable section
open scoped Classical
open HostBoundary (Binary64Division binary64_ceil_exact)
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000

/-- Ordinary default-FP32 Torch tensor metadata. Nonnegative element strides
need not be contiguous; views may overlap. The call type retains the source's
documented integer chunk and two-element clamp argument interface. -/
structure Tensor where
  shape : List Nat
  stride : Nat → Nat
  view : View
  device : Nat

structure Call (α : Type) where
  dt : Tensor
  rate : Tensor
  bias : Option Tensor
  chunk : Int
  softplus : Bool
  lower : α
  upper : α

def Call.hasBias (call : Call α) := call.bias.isSome
def Call.biasView (call : Call α) := (call.bias.map Tensor.view).getD call.dt.view
def Call.biasStride (call : Call α) := (call.bias.map (fun t => t.stride 0)).getD 0
def Call.words (call : Call α) (entry : PhysicalMemory α) : InputWords α :=
  ⟨call.dt.view.read entry,call.rate.view.read entry,call.biasView.read entry⟩

/-- Explicit signed32 successful-size domain, also when a grid is empty.
This is stronger than a vacuous per-launched-lane condition. -/
structure HostSizes (call : Call α) : Prop where
  dimensions : ∀ n ∈ call.dt.shape, n < 2^31
  chunk : call.chunk.toNat < 2^31

def makeHost (call : Call α) (b s h : Nat) (positive : 0 < call.chunk) : Whole.Host where
  batches := b
  sequence := s
  heads := h
  chunk := call.chunk.toNat
  BH := 1
  dtBatch := call.dt.stride 0
  dtTime := call.dt.stride 1
  dtHead := call.dt.stride 2
  rateHead := call.rate.stride 0
  biasHead := call.biasStride
  chunkPositive := by omega
  legalHeadBlock := by simp

inductive BoundaryFailure where
  | rank
  | rateAssertion
  | biasAssertion
  | zeroDivision
  | negativeAllocation
  deriving DecidableEq

structure Geometry (call : Call α) (fp : Binary64Division) where
  host : Whole.Host
  chunks : Nat
  computedChunks : chunks = Nat.ceil (fp.quotient host.sequence host.chunk)
  chunksExact : chunks = host.chunks
  sourceShape : call.dt.shape = [host.batches,host.sequence,host.heads]
  sourceChunk : host.chunk = call.chunk.toNat
  inputStrides : host.dtBatch = call.dt.stride 0 ∧ host.dtTime = call.dt.stride 1 ∧
    host.dtHead = call.dt.stride 2 ∧ host.rateHead = call.rate.stride 0 ∧ host.biasHead = call.biasStride

/-- The source's ordered rank unpacking, assertions and host division.
Negative chunk sizes reach the first torch.empty with a negative dimension;
zero reaches Python division by zero. No power-of-two chunk premise is used. -/
def prepareGeometry (call : Call α) (sizes : HostSizes call) (fp : Binary64Division) :
    Except BoundaryFailure (Geometry call fp) :=
  match shape : call.dt.shape with
  | [b,s,h] =>
    if call.rate.shape ≠ [h] then .error .rateAssertion
    else if ∃ bias, call.bias = some bias ∧ bias.shape ≠ [h] then .error .biasAssertion
    else if zero : call.chunk = 0 then .error .zeroDivision
    else if neg : call.chunk < 0 then .error .negativeAllocation
    else
      let positive : 0 < call.chunk := by omega
      let c := makeHost call b s h positive
      let nchunks := Nat.ceil (fp.quotient s call.chunk.toNat)
      .ok ⟨c,nchunks,rfl,
        binary64_ceil_exact fp s call.chunk.toNat
          (sizes.dimensions s (by simp [shape])) c.chunkPositive sizes.chunk,
        shape,rfl,⟨rfl,rfl,rfl,rfl,rfl⟩⟩
  | _ => .error .rank

/-- An independent description of every preallocation failure in source order. -/
def BoundarySpec (call : Call α) : BoundaryFailure → Prop
  | .rank => ¬ ∃ b s h, call.dt.shape = [b,s,h]
  | .rateAssertion => ∃ b s h, call.dt.shape = [b,s,h] ∧ call.rate.shape ≠ [h]
  | .biasAssertion => ∃ b s h, call.dt.shape = [b,s,h] ∧ call.rate.shape = [h] ∧
      ∃ bias, call.bias = some bias ∧ bias.shape ≠ [h]
  | .zeroDivision => ∃ b s h, call.dt.shape = [b,s,h] ∧ call.rate.shape = [h] ∧
      (∀ bias, call.bias = some bias → bias.shape = [h]) ∧ call.chunk = 0
  | .negativeAllocation => ∃ b s h, call.dt.shape = [b,s,h] ∧ call.rate.shape = [h] ∧
      (∀ bias, call.bias = some bias → bias.shape = [h]) ∧ call.chunk < 0

theorem geometry_failure (call : Call α) (sizes : HostSizes call) (fp : Binary64Division)
    (reason : BoundaryFailure) (failed : prepareGeometry call sizes fp = .error reason) :
    BoundarySpec call reason := by
  unfold prepareGeometry at failed
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

/-- A successful pair of torch.empty calls. Output storage is arbitrary;
the contract contains only shape/device/freshness and preservation of old
allocations. Input allocation aliases are preserved. -/
structure Allocated (α : Type) (call : Call α) (c : Whole.Host)
    (entry : PhysicalMemory α) (preexisting : Nat → Prop) where
  out : View
  sum : View
  distinct : out.allocation ≠ sum.allocation
  memory : PhysicalMemory α
  dtOld : preexisting call.dt.view.allocation
  rateOld : preexisting call.rate.view.allocation
  biasOld : preexisting call.biasView.allocation
  outFresh : ¬ preexisting out.allocation
  sumFresh : ¬ preexisting sum.allocation
  frame : ∀ allocation off, preexisting allocation → memory allocation off = entry allocation off

def Allocated.views (a : Allocated α call c entry preexisting) : Views :=
  ⟨call.dt.view,call.rate.view,call.biasView,a.out,a.sum,a.distinct⟩

theorem Allocated.fresh (a : Allocated α call c entry preexisting) : FreshPhysical a.views preexisting :=
  ⟨a.dtOld,a.rateOld,a.biasOld,a.outFresh,a.sumFresh⟩

theorem Allocated.words_entry (a : Allocated α call c entry preexisting) :
    a.views.words a.memory = call.words entry :=
  words_frame a.views a.fresh entry a.memory a.frame

structure AutotunePlan (α : Type) (call : Call α) (c : Whole.Host) where
  trials : List (LaunchPlan α c)
  selected : LaunchPlan α c
  device : Nat
  deviceBinding : device = call.dt.device
  key : List Nat
  keyBinding : key = [c.chunk,c.heads]

inductive AllocationFailure where
  | first (reason : String)
  | second (reason : String)

/-- Ordinary allocation and device/compiler/autotuner setup boundaries.
No field supplies a kernel execution, a final memory, or output values. -/
structure Runtime (α : Type) (call : Call α) (entry : PhysicalMemory α) (preexisting : Nat → Prop) where
  division : Binary64Division
  allocate : (g : Geometry call division) → Except AllocationFailure (Allocated α call g.host entry preexisting)
  setup : (g : Geometry call division) → (a : Allocated α call g.host entry preexisting) →
    Except String (AutotunePlan α call g.host)

structure ReturnedTensor where
  view : View
  shape : List Nat
  device : Nat
  dtype : FloatDType := .fp32

def returned (call : Call α) (g : Geometry call fp) (view : View) : ReturnedTensor :=
  ⟨view,[g.host.batches,g.host.heads,g.chunks,g.host.chunk],call.dt.device,.fp32⟩

inductive Outcome (α : Type) where
  | boundaryFailure (reason : BoundaryFailure)
  | allocationFailure (reason : AllocationFailure)
  | runtimeFailure (reason : String)
  | kernelFailure
  | returned (first second : ReturnedTensor) (memory : PhysicalMemory α)

def executeHost (call : Call α) (sizes : HostSizes call)
    (rt : Runtime α call entry preexisting) : Outcome α :=
  match prepareGeometry call sizes rt.division with
  | .error reason => .boundaryFailure reason
  | .ok g => match rt.allocate g with
    | .error reason => .allocationFailure reason
    | .ok a => match rt.setup g a with
      | .error reason => .runtimeFailure reason
      | .ok p => match tunedRun a.views g.host call.hasBias call.softplus call.lower call.upper
          p.trials p.selected a.memory with
        | none => .kernelFailure
        | some final => .returned (returned call g a.sum) (returned call g a.out) final

def ReturnedSpec (call : Call α) (g : Geometry call fp) (a : Allocated α call g.host entry preexisting)
    (p : AutotunePlan α call g.host) (final : PhysicalMemory α) : Prop :=
  (returned call g a.sum).shape = [g.host.batches,g.host.heads,g.host.chunks,g.host.chunk] ∧
  (returned call g a.out).shape = [g.host.batches,g.host.heads,g.host.chunks,g.host.chunk] ∧
  PhysicalValues p.selected.model (reblock g.host p.selected.block) p.selected.scan a.views
    (call.words entry) call.hasBias call.softplus call.lower call.upper final ∧
  (∀ allocation off, ¬ a.views.written g.host allocation off → final allocation off = a.memory allocation off) ∧
  (∀ allocation off, preexisting allocation → final allocation off = entry allocation off)

def HostSpec (call : Call α) (sizes : HostSizes call) (rt : Runtime α call entry preexisting)
    (outcome : Outcome α) : Prop :=
  (∃ reason, prepareGeometry call sizes rt.division = .error reason ∧ BoundarySpec call reason ∧
    outcome = .boundaryFailure reason) ∨
  (∃ g reason, prepareGeometry call sizes rt.division = .ok g ∧ rt.allocate g = .error reason ∧
    outcome = .allocationFailure reason) ∨
  (∃ g a reason, prepareGeometry call sizes rt.division = .ok g ∧ rt.allocate g = .ok a ∧
    rt.setup g a = .error reason ∧ outcome = .runtimeFailure reason) ∨
  (∃ g a p final, prepareGeometry call sizes rt.division = .ok g ∧ rt.allocate g = .ok a ∧
    rt.setup g a = .ok p ∧ outcome = .returned (returned call g a.sum) (returned call g a.out) final ∧
    ReturnedSpec call g a p final)

/-- Every source host branch, including empty dimensions, all autotune trial
launches, the selected full grid, physical output values and the complete
preexisting-allocation frame. The kernel-failure constructor is excluded by
the proved source execution, never by a result-supplying runtime callback. -/
theorem host_outcomes (call : Call α) (sizes : HostSizes call)
    (rt : Runtime α call entry preexisting) :
    HostSpec call sizes rt (executeHost call sizes rt) := by
  cases hg : prepareGeometry call sizes rt.division with
  | error reason =>
    exact Or.inl ⟨reason,hg,geometry_failure call sizes rt.division reason hg,by simp [executeHost,hg]⟩
  | ok g =>
    cases ha : rt.allocate g with
    | error reason => exact Or.inr (Or.inl ⟨g,reason,hg,ha,by simp [executeHost,hg,ha]⟩)
    | ok a =>
      cases hp : rt.setup g a with
      | error reason => exact Or.inr (Or.inr (Or.inl ⟨g,a,reason,hg,ha,hp,by simp [executeHost,hg,ha,hp]⟩))
      | ok p =>
        obtain ⟨final,ran,values,frame,old⟩ := tuned_correct a.views g.host a.fresh
          call.hasBias call.softplus call.lower call.upper p.trials p.selected a.memory
        rw [a.words_entry] at values
        apply Or.inr; apply Or.inr; apply Or.inr
        refine ⟨g,a,p,final,hg,ha,hp,?_,?_,?_,values,frame,?_⟩
        · simp only [executeHost,hg,ha,hp,ran]
        · simp [returned,g.chunksExact]
        · simp [returned,g.chunksExact]
        · exact fun allocation off h => (old allocation off h).trans (a.frame allocation off h)

#guard_msgs (drop info) in
#specNonCircular BoundarySpec avoiding [prepareGeometry,executeHost]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.RawBoundary
