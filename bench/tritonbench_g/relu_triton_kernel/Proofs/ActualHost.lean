import bench.tritonbench_g.relu_triton_kernel.Proofs.ActualGrid
import bench.tritonbench_g.relu_triton_kernel.Proofs.ActualAdmission

namespace VeriTile.Bench.TritonBenchG.ReluTritonKernel.Actual
open VeriTile.Triton
open HostWrapper (Run Result HostState Event call note)
noncomputable section
open Classical
set_option maxHeartbeats 7000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- Actual host metadata is retained, including empty shapes, strides,
nonzero storage offsets, raw data_ptr, storage identity, and owner device. -/
structure Tensor where
  region : RegionName
  dtype : CanonicalDType
  device : String
  shape : List Nat
  strides : List Int
  storageOffset : Nat
  dataPtr : BitVec 64
  storageBase : Nat
  storageBytes : Nat

def Tensor.original (x : Tensor) : HostWrapper.Tensor :=
  ⟨x.region,x.dtype.name,x.device,x.shape,x.strides,x.storageOffset⟩
def Tensor.numel (x : Tensor) : Nat := x.shape.prod

/-- Standard empty_like(float32, x.device) primitive contract. Contents are
arbitrary. Preserve-format strides are the allocator's reported strides,
including when empty; they are never reconstructed by the proof. -/
structure Allocation (x : Tensor) (before : ByteMemory.Memory) where
  output : Tensor
  memory : ByteMemory.Memory
  shape : output.shape=x.shape
  dtype : output.dtype=.fp32
  device : output.device=x.device
  fresh : x.storageBase+x.storageBytes≤output.storageBase ∨
    output.storageBase+output.storageBytes≤x.storageBase
  frame : ∀ address,address<output.storageBase ∨ output.storageBase+output.storageBytes≤address →
    memory address=before address

/-- Device-pointer extraction exposes its actual ABI pointer and a bytewise
translation to canonical backing. No equality to raw data_ptr, ownership
match, kernel values, or final memory is supplied by this primitive. -/
structure PointerBinding (ctx : AccessContext) (t : Tensor) (v : PhysicalView) (width : Nat) : Prop where
  backing : v.base=t.storageBase
  capacity : v.capacity=t.storageBytes
  translated : t.dataPtr≠0 → ∀ byte,byte<v.capacity → ctx.toBacking (v.deviceBase+byte)=v.base+byte
  pointer : t.numel>0 → v.pointer.toNat=v.deviceBase+t.storageOffset*width
  null : t.dataPtr=0 → v.pointer=0

structure Prepared (x output : Tensor) (d : Storage) (current : Nat) where
  ctx : AccessContext
  inputView : PhysicalView
  outputView : PhysicalView
  currentDevice : ctx.currentDevice=current
  inputBinding : PointerBinding ctx x inputView d.bytes
  outputBinding : PointerBinding ctx output outputView 4

def Prepared.pointers {x output : Tensor} {d : Storage} {current : Nat}
    (p : Prepared x output d current) : RegionName → BitVec 64 :=
  fun name => if name="x" then p.inputView.pointer else p.outputView.pointer

/-- These ports cover only ordinary host/runtime operations. Compilation,
pointer extraction and enqueue preparation cannot supply kernel results. -/
structure NativePrimitives (σ : Type) where
  device : (x : Tensor) → Run σ {device : String // device=x.device}
  emptyLike : (x : Tensor) → (dtype : CanonicalDType) → (device : String) →
    (before : ByteMemory.Memory) → Run σ (Allocation x before)
  numel : (output : Tensor) → Run σ {n : Nat // n=output.numel}
  currentDevice : Run σ Nat
  compile : Tensor → Tensor → Storage → Nat → Run σ Unit
  launcherContextAndGridParse : (x output : Tensor) → Nat → Nat →
    Run σ {_u : Unit // grid (Tensor.numel output)≤2^31-1}
  pointers : (x output : Tensor) → (d : Storage) → (current : Nat) → Run σ (Prepared x output d current)
  enqueueReady : Tensor → Tensor → Nat → Nat → Run σ Unit

structure LaunchReady (x output : Tensor) (n : Nat) where
  storage : Storage
  route : dtypeRoute x.dtype=.admitted storage
  constexpr : n<2^64
  current : Nat
  prepared : Prepared x output storage current
  nativeGrid : grid output.numel≤2^31-1

def rejected {σ α : Type} (message : String) : Run σ α := fun s => .error (.native message) s

/-- Pinned binder/frontend routes. Native context and compilation remain
fallible; no host device guard is added. -/
def prepareLaunch {σ : Type} (P : NativePrimitives σ) (x output : Tensor) (n : Nat) : Run σ (LaunchReady x output n) := do
  let current ← P.currentDevice
  match h : dtypeRoute x.dtype with
  | .missingSignatureMap => rejected "dtype is absent from the pinned signature map"
  | .unsupportedSignature => rejected "dtype signature is unsupported by the pinned frontend"
  | .orderedComparisonRejected =>
      if n<2^64 then rejected "ordered comparison is unsupported for this fp8 dtype"
      else rejected "Nonrepresentable constexpr integer"
  | .admitted d =>
      if hn : n<2^64 then do
        let _ ← P.compile x output d n
        let ready ← P.launcherContextAndGridParse x output n (grid n)
        let pointers ← P.pointers x output d current
        let _ ← P.enqueueReady x output n (grid n)
        pure ⟨d,h,hn,current,pointers,ready.property⟩
      else rejected "Nonrepresentable constexpr integer"

structure Ready (x : Tensor) (before : ByteMemory.Memory) where
  allocation : Allocation x before
  n : Nat
  numel : n=allocation.output.numel
  launch : LaunchReady x allocation.output n

/-- Literal source host order, including the unused num_blocks expression
and separate launch-grid expression. No rank, dtype, stride or CUDA assertion. -/
def prepareHost {σ : Type} (P : NativePrimitives σ) (x : Tensor) (before : ByteMemory.Memory) :
    Run (HostState σ) (Ready x before) := do
  let device ← call (.device x.original) (P.device x)
  let allocated ← call (.emptyLike x.original "float32" device.val) (P.emptyLike x .fp32 device.val before)
  let n ← call (.numel allocated.output.original) (P.numel allocated.output)
  let numBlocks := (n.val+1024-1)/1024
  note (.computedNumBlocks n.val numBlocks)
  let launch ← call (.launch x.original allocated.output.original n.val (grid n.val) 1024)
    (prepareLaunch P x allocated.output n.val)
  pure ⟨allocated,n.val,n.property,launch⟩

def Ready.safe {x : Tensor} {before : ByteMemory.Memory} (r : Ready x before) : Prop :=
  let p := r.launch.prepared
  GridSafe p.ctx r.launch.storage p.inputView p.outputView r.n (List.range (grid r.n))

theorem Ready.separated {x : Tensor} {before : ByteMemory.Memory} (r : Ready x before) :
    r.launch.prepared.inputView.Disjoint r.launch.prepared.outputView := by
  have hi := r.launch.prepared.inputBinding
  have ho := r.launch.prepared.outputBinding
  unfold PhysicalView.Disjoint
  rw [hi.backing,hi.capacity,ho.backing,ho.capacity]
  exact r.allocation.fresh

inductive Observation (σ : Type) where
  | nativeFailure (reason : HostWrapper.Error) (state : HostState σ)
  | sourceAccessOutsideModel (output : Tensor) (allocated : ByteMemory.Memory) (state : HostState σ)
  | interpreterFailure (state : HostState σ)
  | returned (output : Tensor) (memory : ByteMemory.Memory) (state : HostState σ)

/-- Outside-model access records neither a native exception nor a successful
return, race winner, rollback, or any final-memory claim. -/
def sourceHost {σ : Type} (P : NativePrimitives σ) (b : Numerics) (x : Tensor)
    (before : ByteMemory.Memory) (state : HostState σ) : Observation σ :=
  match prepareHost P x before state with
  | .error reason t => .nativeFailure reason t
  | .ok r t =>
      if r.safe then
        let p := r.launch.prepared
        match runGrid p.ctx.toBacking b r.launch.storage p.pointers 0 "x" "out" r.n
          (List.range (grid r.n)) r.allocation.memory with
        | none => .interpreterFailure t
        | some final => .returned r.allocation.output final
            {t with trace := t.trace++[.returned r.allocation.output.original]}
      else .sourceAccessOutsideModel r.allocation.output r.allocation.memory t

/-- Independent complete-byte returned-buffer specification. -/
def hostFormula {σ : Type} (P : NativePrimitives σ) (b : Numerics) (x : Tensor)
    (before : ByteMemory.Memory) (state : HostState σ) : Observation σ :=
  match prepareHost P x before state with
  | .error reason t => .nativeFailure reason t
  | .ok r t =>
      if r.safe then
        let p := r.launch.prepared
        .returned r.allocation.output
          (gridFormula p.ctx.toBacking b r.launch.storage p.inputView.pointer p.outputView.pointer
            r.n (List.range (grid r.n)) r.allocation.memory)
          {t with trace := t.trace++[.returned r.allocation.output.original]}
      else .sourceAccessOutsideModel r.allocation.output r.allocation.memory t

theorem original_source_all_outcomes {σ : Type} (P : NativePrimitives σ) (b : Numerics) (x : Tensor)
    (before : ByteMemory.Memory) (state : HostState σ) :
    sourceHost P b x before state=hostFormula P b x before state := by
  unfold sourceHost hostFormula
  cases h : prepareHost P x before state with
  | error reason t => rfl
  | ok r t =>
    by_cases hs : r.safe
    · simp only [if_pos hs]
      rw [source_grid_formula b r.launch.storage r.launch.prepared.pointers 0 "x" "out"
        r.launch.prepared.inputView r.launch.prepared.outputView r.launch.prepared.ctx r.n
        (List.range (grid r.n)) r.allocation.memory (by simp [Prepared.pointers])
        (by simp [Prepared.pointers]) r.separated hs]
    · simp only [if_neg hs]

theorem returned_frame {x : Tensor} {before : ByteMemory.Memory} (r : Ready x before)
    (b : Numerics) (safe : r.safe) (address : Nat)
    (outside : address<r.allocation.output.storageBase ∨
      r.allocation.output.storageBase+r.allocation.output.storageBytes≤address) :
    gridFormula r.launch.prepared.ctx.toBacking b r.launch.storage
      r.launch.prepared.inputView.pointer r.launch.prepared.outputView.pointer
      r.n (List.range (grid r.n)) r.allocation.memory address=before address := by
  have frame := grid_frame b r.launch.storage r.launch.prepared.inputView r.launch.prepared.outputView
    r.launch.prepared.ctx r.n (List.range (grid r.n)) r.allocation.memory safe
  have nativeOutside : address<r.launch.prepared.outputView.base ∨
      r.launch.prepared.outputView.base+r.launch.prepared.outputView.capacity≤address := by
    rw [r.launch.prepared.outputBinding.backing,r.launch.prepared.outputBinding.capacity]
    exact outside
  exact (frame address nativeOutside).trans (r.allocation.frame address outside)

/-- Native grid parsing bounds pid values, without narrowing constexpr N.
All read obligations for the other launched programs are still required. -/
theorem returned_only_first_program {x : Tensor} {before : ByteMemory.Memory} (r : Ready x before)
    (b : Numerics) :
    gridFormula r.launch.prepared.ctx.toBacking b r.launch.storage
      r.launch.prepared.inputView.pointer r.launch.prepared.outputView.pointer
      r.n (List.range (grid r.n)) r.allocation.memory=
      byteFormula (if grid r.n=0 then [] else
        programWrites r.launch.prepared.ctx.toBacking b r.launch.storage
          r.launch.prepared.inputView.pointer r.launch.prepared.outputView.pointer
          r.n 0 r.allocation.memory) r.allocation.memory := by
  unfold gridFormula
  rw [launched_stores_only_first]
  rw [r.numel]
  have := r.launch.nativeGrid
  omega

theorem untouched_bytes {x : Tensor} {before : ByteMemory.Memory} (r : Ready x before)
    (b : Numerics) (address : Nat)
    (outside : ∀ w∈gridWrites r.launch.prepared.ctx.toBacking b r.launch.storage
      r.launch.prepared.inputView.pointer r.launch.prepared.outputView.pointer
      r.n (List.range (grid r.n)) r.allocation.memory,address<w.address ∨ w.address+4≤address) :
    gridFormula r.launch.prepared.ctx.toBacking b r.launch.storage
      r.launch.prepared.inputView.pointer r.launch.prepared.outputView.pointer
      r.n (List.range (grid r.n)) r.allocation.memory address=r.allocation.memory address := by
  rw [gridFormula,←writes_pointwise]
  exact writes_byte_frame _ _ _ outside

theorem any_complete_launch_schedule {x : Tensor} {before : ByteMemory.Memory} (r : Ready x before)
    (b : Numerics) (jobs : List Nat) (complete : jobs.Perm (List.range (grid r.n)))
    (safe : r.safe) :
    runGrid r.launch.prepared.ctx.toBacking b r.launch.storage r.launch.prepared.pointers
      0 "x" "out" r.n jobs r.allocation.memory=
      some (gridFormula r.launch.prepared.ctx.toBacking b r.launch.storage
        r.launch.prepared.inputView.pointer r.launch.prepared.outputView.pointer
        r.n (List.range (grid r.n)) r.allocation.memory) := by
  have safeJobs : GridSafe r.launch.prepared.ctx r.launch.storage
      r.launch.prepared.inputView r.launch.prepared.outputView r.n jobs :=
    fun pid hp => safe pid (complete.mem_iff.mp hp)
  rw [source_grid_formula b r.launch.storage r.launch.prepared.pointers 0 "x" "out"
    r.launch.prepared.inputView r.launch.prepared.outputView r.launch.prepared.ctx r.n jobs
    r.allocation.memory (by simp [Prepared.pointers]) (by simp [Prepared.pointers]) r.separated safeJobs]
  rw [complete_schedule_formula _ _ _ _ _ _ _ jobs _ complete]
  rw [r.numel]
  have := r.launch.nativeGrid
  omega

#specNonCircular hostFormula avoiding [sourceHost,runGrid,runProgram,executeBody,step,applyWrites]
#axiomsClean Ready.separated
#axiomsClean original_source_all_outcomes
#axiomsClean returned_frame
#axiomsClean returned_only_first_program
#axiomsClean untouched_bytes
#axiomsClean any_complete_launch_schedule
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ReluTritonKernel.Actual
