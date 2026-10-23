import bench.repairs.tritonbench_g.context_attn_bloom.Kernels
import bench.repairs.tritonbench_g.context_attn_bloom.Proofs.KernelProfile
import bench.repairs.tritonbench_g.context_attn_bloom.Proofs.Task

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.KernelOnly
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program RealSetup
open FPSuccess (Call)
open Bytes
open VeriTile.Torch.FP32 (logicalValue)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000

/-- Both execution interpretations consume this exact shared source quotation. -/
theorem shared_source (p : Layout) (r : Regions) (scale : ℝ) :
    kernel p r scale=VeriTile.Bench.Repairs.ContextAttnBloomSource.kernel
      r.query r.key r.value scale r.start r.sequence r.out r.tokens r.request r.prompt
      p.qRow p.qHead p.qDim p.kRow p.kHead p.kDim p.vRow p.vHead p.vDim
      p.outRow p.outHead p.outDim p.requestRow p.requestToken p.kvGroup p.headDim p.BM p.D p.BN := rfl

structure Result where
  graph : State Native.Expr
  bytes : Memory

/-- Actual source grid followed by its proved byte stores. The native profile
is fixed here; the caller supplies only individual instruction meanings. -/
def run (c : Call) (B : Windows extent) (ops : Native.WordISA) (before : Memory) : Option Result := do
  let graph ← FPSuccess.runGrid c Native.scalar Native.dots (order c) (initial c B before)
  return ⟨graph,stores c B (KernelNative.profile c.args) ops graph before⟩

/-- The target takes its launch scalar directly from the checked FP32 table. -/
def nativeTile (c : Call) (idx : GridIndex (Host.grid c.args)) (s : State Native.Expr) : State Native.Expr :=
  let launch := FPSuccess.atIndex s idx
  FPSuccess.scatter (Host.layout c.args) c.regions launch launch
    (FPReference.output (Host.layout c.args) Native.scalar Native.dots
      (LaunchScale.value c.args.q.dims) (FPReference.inputs (Host.layout c.args) c.regions launch))

def nativeGrid (c : Call) : List (GridIndex (Host.grid c.args)) → State Native.Expr → State Native.Expr
  | [],s => s
  | idx::rest,s => nativeGrid c rest (nativeTile c idx s)

def nativeReference (c : Call) (B : Windows extent) (before : Memory) : State Native.Expr :=
  nativeGrid c (order c) (initial c B before)

/-- Source-independent normalized tile recurrence, lowered with the fixed
profile to scalar instructions and fixed K8 MMA blocks. -/
def nativeTarget (c : Call) (B : Windows extent) (ops : Native.WordISA)
    (before : Memory) (idx : GridIndex (Host.grid c.args)) (i : RealIO.Lane (Host.layout c.args)) : Word :=
  outputWord c (KernelNative.profile c.args) ops (nativeReference c B before) (idx,i,⟨0,by omega⟩)

theorem nativeReference_eq (c : Call) (B : Windows extent) (before : Memory)
    (launchScalar : LaunchScale.Bound c.args) : nativeReference c B before=reference c B before := by
  change Host.scale c.args=LaunchScale.value c.args.q.dims at launchScalar
  have grid (jobs : List (GridIndex (Host.grid c.args))) (s : State Native.Expr) :
      nativeGrid c jobs s=FPReference.grid c Native.scalar Native.dots jobs s := by
    induction jobs generalizing s with
    | nil => rfl
    | cons idx rest ih =>
      change nativeGrid c rest (nativeTile c idx s)=
        FPReference.grid c Native.scalar Native.dots rest (FPReference.tile c Native.scalar Native.dots idx s)
      rw [ih]
      congr 1
      unfold nativeTile FPReference.tile
      rw [launchScalar]
  exact grid _ _

theorem nativeTarget_eq (c : Call) (B : Windows extent) (ops : Native.WordISA)
    (before : Memory) (launchScalar : LaunchScale.Bound c.args)
    (idx : GridIndex (Host.grid c.args)) (i : RealIO.Lane (Host.layout c.args)) :
    nativeTarget c B ops before idx i=
      referenceWord c B (KernelNative.profile c.args) ops before (idx,i,⟨0,by omega⟩) := by
  simp only [nativeTarget,nativeReference_eq c B before launchScalar,referenceWord]

/-- The physical source read is the caller's logical input Tensor word. -/
theorem input_tensor (c : Call) (v : Checked.Views) (B : Windows extent)
    (tensors : Checked.tensorChecks c.args v) (storage : Checked.StorageBinding c.regions v B)
    (roles : CheckedTensor.LogicalRoles c.regions)
    (rg : RegionName) (x : Checked.View) (role : (rg,x)∈Checked.roleViews c.regions v)
    (payload : x∈[v.q,v.k,v.v]) (logical : rg∈[c.regions.query,c.regions.key,c.regions.value,c.regions.out])
    (input : CheckedTensor.Input x) (before : Memory) (i : Nat) (bound : i < input.tensor.metadata.numel) :
    ((initial c B before).mem rg i).read .real=
      Native.Expr.bits ((logicalValue (CheckedTensor.read before) input.tensor input.laws i).bits) :=
  CheckedTensor.logical_source c v B tensors storage rg x role payload (roles rg logical) input before i bound

theorem native_tensor (c : Call) (v : Checked.Views) (A : FlatAlloc) (B : Windows A.extent)
    (requestedMax : Nat) (checked : Checked.checkInputs c.args c.regions v A.extent c.input requestedMax=Bool.true)
    (before : Memory) (observed : Observed c B before)
    (storage : Checked.StorageBinding c.regions v B) (labels : Checked.Labels c.regions)
    (launchScalar : LaunchScale.Bound c.args) (ops : Native.WordISA) (out : CheckedTensor.Output v) :
    Returned.MachineFacts c v A B before ∧
    ∃ result,run c B ops before=some result ∧
      FPSuccess.runGrid c Native.scalar Native.dots (order c) (initial c B before)=some result.graph ∧
      result.graph.mem=(nativeReference c B before).mem ∧
      (∀ idx i,FPSuccess.laneActive c idx i →
        (logicalValue (CheckedTensor.read result.bytes) out.tensor out.laws (CheckedTensor.flatIndex c idx i)).bits=
          nativeTarget c B ops before idx i) ∧
      (∀ lane : Lane c,active c lane →
        result.bytes (address c B lane)=byte (nativeTarget c B ops before lane.1 lane.2.1) lane.2.2) ∧
      (∀ a,(¬∃ lane : Lane c,active c lane ∧ address c B lane=a) → result.bytes a=before a) ∧
      (∀ rg∈Checked.readRegions c.regions,∀ off,off < A.extent rg → ∀ j : Fin 4,
        result.bytes (B.base rg+4*off+j.val)=before (B.base rg+4*off+j.val)) ∧
      (∀ rg off,¬FPSuccess.writes c (order c) rg off → result.graph.mem rg off=(initial c B before).mem rg off) ∧
      (¬Needed c → result.bytes=before) := by
  have guard := (Checked.checkInputs_iff c.args c.regions v A.extent c.input requestedMax).mp checked
  obtain ⟨result,invocation,_,_,_,_,source,referenceEq,bytes,frame,inputs,graphFrame,empty⟩ :=
    Returned.whole_bytes c v A B requestedMax checked before observed storage labels
      (KernelNative.profile c.args) ops ⟨0,0,0⟩
  have byteEq : result.bytes=stores c B (KernelNative.profile c.args) ops result.graph before := by
    unfold Returned.invoke at invocation
    rw [source] at invocation
    exact (congrArg Returned.Result.bytes (Option.some.inj invocation)).symm
  refine ⟨Returned.machine_facts c v A B requestedMax checked before observed storage labels,
    ⟨result.graph,result.bytes⟩,?_,source,?_,?_,?_,frame,inputs,graphFrame,empty⟩
  · simp only [run,source,byteEq]
    rfl
  · rw [nativeReference_eq c B before launchScalar]
    exact referenceEq
  · intro idx i on
    rw [nativeTarget_eq c B ops before launchScalar]
    exact CheckedTensor.logical_output c v B guard.1 guard.2.1 guard.2.2.1 storage out
      (KernelNative.profile c.args) ops before result.bytes idx i on (fun j => bytes (idx,i,j) on)
  · intro lane on
    rw [nativeTarget_eq c B ops before launchScalar]
    exact bytes lane on

/-- The independent mathematical formula, coverage, and full frame follow
from actual shared source execution, including its flattened memory execution. -/
theorem mathematical_real (a : Host.Arguments) (r : Regions) (v : Checked.Views)
    (A : FlatAlloc) (s : BlockState) (requestedMax : Nat)
    (checked : Checked.checkInputs a r v A.extent (inputs s r) requestedMax=Bool.true)
    (labels : Checked.Labels r) (launchScalar : LaunchScale.Bound a)
    (disjoint : A.Disjoint) (covered : ∀ rg,rg∉A.regions → A.extent rg=0)
    (zero : s.undef=(fun _ _ => 0)) :
    Spec.Real (∃ final,
      RealStream.run a r (HostRun.allPrograms a) s=some final ∧
      RealStream.flatRun a r A (HostRun.allPrograms a) (A.flattenState s)=some (A.flattenState final) ∧
      (∀ idx i,Host.active a r s idx i → final.readMem r.out (RealStream.offset a r s idx i)=
        Checked.mathematical a (inputs s r) (Host.launchOf a idx) i) ∧
      (∀ (b h row d : Nat),b < a.batch → h < a.q.heads →
        (row : Int) < (inputs s r).sequence b-(inputs s r).prompt b → d < a.q.dims →
        ∃ idx : GridIndex (Host.grid a),∃ i : RealIO.Lane (Host.layout a),
          Host.active a r s idx i ∧ (Host.launchOf a idx).batch=b ∧
          (Host.launchOf a idx).head=h ∧ queryRow (Host.layout a) (Host.launchOf a idx) i.1.val=row ∧ i.2.1.val=d) ∧
      (∀ rg off,¬RealStream.writes a r s (HostRun.allPrograms a) rg off → final.mem rg off=s.mem rg off) ∧
      (∀ rg∈Checked.readRegions r,∀ off,final.mem rg off=s.mem rg off)) := by
  obtain ⟨final,source,flat,formula,coverage,frame,reads,_,_,_⟩ :=
    Checked.whole_mathematical_real a r v A s requestedMax checked labels launchScalar disjoint covered zero 0
  exact ⟨final,source,flat,formula,coverage,frame,reads⟩

#specNonCircular nativeTarget avoiding [FP.Structural.exec,FP.Structural.run,FPSuccess.runGrid,kernel,run]
#axiomsClean shared_source
#axiomsClean input_tensor
#axiomsClean native_tensor
#axiomsClean mathematical_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.KernelOnly
