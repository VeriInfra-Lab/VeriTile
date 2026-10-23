import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.Task
import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.KernelScale
import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.KernelProfile

namespace VeriTile.Bench.Repairs.ContextAttnFwd.KernelOnly
open VeriTile VeriTile.Triton FP.Structural NativeGraph
open VeriTile.Torch.FP32 (logicalValue)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 5000000

theorem shared_source (p : Layout) (r : Regions) (scale : ℝ) :
    kernel p r .fp32 scale=
      VeriTile.Bench.TritonBenchG.ContextAttnFwd.Program.kernel p r .fp32 Bool.false scale := rfl

structure Result where
  graph : State Expr
  bytes : Physical.Memory

/-- Actual shared source grid followed by its proved physical stores. -/
def run (c : Host.Call) (B : Native.Bytes extent) (ops : ScalarOps)
    (before : Physical.Memory) : Option Result := do
  let graph ← FPSuccess.runGrid c (backend (NativeSM90.plan c.layout.BM c.layout.D))
    (Returned.order c) (Snapshot.initial c B before)
  return ⟨graph,Physical.stores c B ops graph before⟩

def nativeTile (c : Host.Call) (idx : GridIndex c.grid) (s : State Expr) : State Expr :=
  let plan := NativeSM90.plan c.layout.BM c.layout.D
  let launch := FPSuccess.atIndex s idx
  FPSuccess.scatter c.layout c.regions launch launch
    (FPReference.output c.layout (backend plan) (dots.support (scalar plan) c.layout)
      (KernelScale.value c.args.q.dims) (FPReference.inputs c.layout c.regions launch))

def nativeGrid (c : Host.Call) : List (GridIndex c.grid) → State Expr → State Expr
  | [],s => s
  | idx::rest,s => nativeGrid c rest (nativeTile c idx s)

def nativeReference (c : Host.Call) (B : Native.Bytes extent) (before : Physical.Memory) : State Expr :=
  nativeGrid c (Returned.order c) (Snapshot.initial c B before)

/-- Independent masked recurrence evaluated by the fixed SM90 instruction graph. -/
def nativeTarget (c : Host.Call) (B : Native.Bytes extent) (ops : ScalarOps)
    (before : Physical.Memory) (idx : GridIndex c.grid) (i : TileIndex [c.layout.BM,c.layout.D]) : Word :=
  Physical.outputWord c ops (nativeReference c B before) (idx,i,⟨0,by omega⟩)

theorem nativeReference_eq (c : Host.Call) (B : Native.Bytes extent) (before : Physical.Memory)
    (scale : KernelScale.Bound c) :
    (nativeReference c B before).mem=CheckedTensor.referenceMemory c B before := by
  change Host.scale c=KernelScale.value c.args.q.dims at scale
  have grid (jobs : List (GridIndex c.grid)) (s : State Expr) :
      nativeGrid c jobs s=FPReference.grid c (backend (NativeSM90.plan c.layout.BM c.layout.D))
        (dots.support (scalar (NativeSM90.plan c.layout.BM c.layout.D)) c.layout) jobs s := by
    induction jobs generalizing s with
    | nil => rfl
    | cons idx rest ih =>
      change nativeGrid c rest (nativeTile c idx s)=
        FPReference.grid c _ _ rest (FPReference.tile c _ _ idx s)
      rw [ih]
      congr 1
      unfold nativeTile FPReference.tile
      rw [scale]
  exact congrArg State.mem (grid _ _)

alias input_tensor := CheckedTensor.logical_source

theorem native_tensor (c : Host.Call) (v : Checked.Views) (A : FlatAlloc)
    (B : Native.Bytes A.extent) (start sequence prompt : Nat → Int)
    (checked : Checked.checkInputs c v A.extent start sequence prompt=Bool.true)
    (before : Physical.Memory)
    (observed : Checked.MetadataObservation c B before start sequence prompt)
    (storage : Checked.StorageBinding c v B) (roles : Snapshot.LogicalRoles c)
    (labels : FPSuccess.Separate c) (scale : KernelScale.Bound c)
    (ops : ScalarOps) (out : CheckedTensor.Output v) :
    (∃ abi : Native.ABI c,MachineFacts c B before abi) ∧
    ∃ result,run c B ops before=some result ∧
      FPSuccess.runGrid c (backend (NativeSM90.plan c.layout.BM c.layout.D)) (Returned.order c)
        (Snapshot.initial c B before)=some result.graph ∧
      result.graph.mem=(nativeReference c B before).mem ∧
      (∀ idx i,c.active idx i →
        (logicalValue (CheckedTensor.read result.bytes) out.tensor out.laws (CheckedTensor.flatIndex c idx i)).bits=
          nativeTarget c B ops before idx i) ∧
      (∀ lane : Physical.Lane c,Physical.active c lane → result.bytes (Physical.address c B lane)=
        Physical.byte (nativeTarget c B ops before lane.1 lane.2.1) lane.2.2) ∧
      (∀ a,(¬∃ lane : Physical.Lane c,Physical.active c lane ∧ Physical.address c B lane=a) →
        result.bytes a=before a) ∧
      (∀ rg∈Physical.reads c,∀ off,off<A.extent rg → ∀ j : Fin 4,
        result.bytes (B.base rg+4*off+j.val)=before (B.base rg+4*off+j.val)) ∧
      (∀ rg off,¬FPSuccess.writes c (Returned.order c) rg off →
        result.graph.mem rg off=(Snapshot.initial c B before).mem rg off) ∧
      (¬Returned.Needed c → result.bytes=before) := by
  have guard := (Checked.checkInputs_iff c v A.extent start sequence prompt).mp checked
  obtain ⟨call,machine⟩ := Checked.whole_fp32_bytes c v A B start sequence prompt checked
    before observed storage roles labels ops ⟨0,0,0⟩
  obtain ⟨result,invocation,_,_,_,_,source,_,_,bytes,frame,inputs,graphFrame,empty⟩ := call
  obtain ⟨referenceGraph,referenceRun,referenceEq⟩ := FPReference.whole c
    (backend (NativeSM90.plan c.layout.BM c.layout.D))
    (dots.support (scalar (NativeSM90.plan c.layout.BM c.layout.D)) c.layout)
    (Returned.order c) (Snapshot.initial c B before)
  have same : result.graph=referenceGraph := Option.some.inj (source.symm.trans referenceRun)
  have memory : result.graph.mem=(nativeReference c B before).mem := by
    rw [nativeReference_eq c B before scale]
    exact same ▸ referenceEq
  have byteEq : result.bytes=Physical.stores c B ops result.graph before := by
    unfold Returned.invoke at invocation
    rw [source] at invocation
    exact (congrArg Returned.Result.bytes (Option.some.inj invocation)).symm
  have word (idx : GridIndex c.grid) (i : TileIndex [c.layout.BM,c.layout.D]) (j : Fin 4) :
      Physical.outputWord c ops result.graph (idx,i,j)=nativeTarget c B ops before idx i := by
    simp only [Physical.outputWord,nativeTarget,memory,Physical.owner]
  refine ⟨⟨_,machine⟩,⟨result.graph,result.bytes⟩,?_,source,memory,?_,?_,frame,inputs,graphFrame,empty⟩
  · simp only [run,source,byteEq]
    rfl
  · intro idx i on
    rw [←word idx i ⟨0,by omega⟩]
    exact CheckedTensor.logical_output c v B guard.1 (Checked.geometry c A.extent guard.2.1)
      storage out result.bytes ops result.graph idx i on (fun j => bytes (idx,i,j) on)
  · intro lane on
    rw [←word lane.1 lane.2.1 lane.2.2]
    exact bytes lane on

/-- Initial-array base-2 weighted sum with the actual finite causal sentinel. -/
def mathematical (c : Host.Call) (s : BlockState) (idx : GridIndex c.grid)
    (i : TileIndex [c.layout.BM,c.layout.D]) : ℝ :=
  VeriTile.Bench.TritonBenchG.ContextAttnFwd.RealIO.closedExpected c.layout
    (KernelScale.value c.args.q.dims)
    (VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.launchOf c.args idx)
    (VeriTile.Bench.TritonBenchG.ContextAttnFwd.Program.inputs s c.regions) i

theorem mathematical_real (c : Host.Call) (v : Checked.Views) (A : FlatAlloc)
    (start sequence prompt : Nat → Int)
    (checked : Checked.checkInputs c v A.extent start sequence prompt=Bool.true)
    (s : BlockState) (inputMetadata : Geometry.Metadata c s)
    (labels : FPSuccess.Separate c) (allocation : A.Disjoint)
    (covered : ∀ rg,rg∉A.regions → A.extent rg=0) (scale : KernelScale.Bound c) :
    Spec.Real (∃ final,
      Nonempty (Kernel.GridLaunchedOrdinary
        (VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.flatSource c.args c.regions A Bool.false)
        c.grid (A.flattenState s) final) ∧
      (∀ idx i,VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.active c.args c.regions s idx i →
        final.readMem
          (VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.address c.args c.regions A s idx i).1
          (VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.address c.args c.regions A s idx i).2=
          mathematical c s idx i) ∧
      BlockState.WriteWithin
        (VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.writes c.args c.regions A s)
        (A.flattenState s) final ∧
      (∀ rg off,c.regions.out≠rg → off<A.extent rg →
        final.mem A.flat (A.addr rg off)=(A.flattenState s).mem A.flat (A.addr rg off)) ∧
      (∀ b h row d,b<c.batches → h<c.layout.heads → row<c.length b → d<c.layout.D →
        ∃ idx : GridIndex c.grid,∃ i : TileIndex [c.layout.BM,c.layout.D],
          c.active idx i ∧ VeriTile.Bench.TritonBenchG.ContextAttnFwd.Formula.batch c.layout (c.launchAt idx)=b ∧
          VeriTile.Bench.TritonBenchG.ContextAttnFwd.Formula.head c.layout (c.launchAt idx)=h ∧
          VeriTile.Bench.TritonBenchG.ContextAttnFwd.Formula.queryRow c.layout (c.launchAt idx) i.1.val=row ∧
          i.2.1.val=d)) := by
  obtain ⟨final,source,formula,frame,reads,_⟩ := Checked.whole_mathematical_real c v A
    start sequence prompt checked s inputMetadata labels allocation covered
  refine ⟨final,source,?_,frame,reads,?_⟩
  · intro idx i active
    have value := formula idx i active
    change Host.scale c=KernelScale.value c.args.q.dims at scale
    simpa only [VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.result,mathematical,scale] using value
  · exact RealProof.coverage c A s
      (Checked.geometry c A.extent ((Checked.checkInputs_iff c v A.extent start sequence prompt).mp checked).2.1)
      inputMetadata labels

#specNonCircular nativeTarget avoiding [FP.Structural.exec,FP.Structural.run,FPSuccess.runGrid,kernel,run]
#specNonCircular mathematical avoiding [VeriTile.Triton.exec,kernel,run,Returned.invoke]
#axiomsClean shared_source
#axiomsClean input_tensor
#axiomsClean native_tensor
#axiomsClean mathematical_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnFwd.KernelOnly
