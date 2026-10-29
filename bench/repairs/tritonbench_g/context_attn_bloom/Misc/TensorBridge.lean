import bench.repairs.tritonbench_g.context_attn_bloom.Proofs.PhysicalReal
import bench.repairs.tritonbench_g.context_attn_bloom.Misc.NativeCorrect
import bench.repairs.tritonbench_g.context_attn_bloom.Proofs.PhysicalNative
import bench.repairs.tritonbench_g.context_attn_bloom.Misc.KernelOnly

/-! Complete checked Tensor applications, retained with their original domain,
source executions, native readback and all physical frames. -/
namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.TensorBridge
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program RealSetup
open FPSuccess (Call)
open Bytes
open VeriTile.Torch.FP32 (logicalValue)
noncomputable section

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

theorem correct_real (a : Host.Arguments) (r : Regions) (v : Checked.Views)
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
      (∀ rg∈Checked.readRegions r,∀ off,final.mem rg off=s.mem rg off)) :=
  KernelOnly.mathematical_real a r v A s requestedMax checked labels launchScalar disjoint covered zero

theorem correct_fp32 (c : Call) (v : Checked.Views) (A : FlatAlloc) (B : Windows A.extent)
    (requestedMax : Nat) (checked : Checked.checkInputs c.args c.regions v A.extent c.input requestedMax=Bool.true)
    (before : Memory) (observed : Observed c B before)
    (storage : Checked.StorageBinding c.regions v B) (labels : Checked.Labels c.regions)
    (launchScalar : LaunchScale.Bound c.args) (ops : Native.WordISA) (out : CheckedTensor.Output v) :
    Returned.MachineFacts c v A B before ∧
    ∃ result,KernelOnly.run c B ops before=some result ∧
      FPSuccess.runGrid c Native.scalar Native.dots (order c) (initial c B before)=some result.graph ∧
      result.graph.mem=(KernelOnly.nativeReference c B before).mem ∧
      (∀ idx i,FPSuccess.laneActive c idx i →
        (logicalValue (CheckedTensor.read result.bytes) out.tensor out.laws (CheckedTensor.flatIndex c idx i)).bits=
          KernelOnly.nativeTarget c B ops before idx i) ∧
      (∀ lane : Lane c,active c lane →
        result.bytes (address c B lane)=byte (KernelOnly.nativeTarget c B ops before lane.1 lane.2.1) lane.2.2) ∧
      (∀ a,(¬∃ lane : Lane c,active c lane ∧ address c B lane=a) → result.bytes a=before a) ∧
      (∀ rg∈Checked.readRegions c.regions,∀ off,off < A.extent rg → ∀ j : Fin 4,
        result.bytes (B.base rg+4*off+j.val)=before (B.base rg+4*off+j.val)) ∧
      (∀ rg off,¬FPSuccess.writes c (order c) rg off → result.graph.mem rg off=(initial c B before).mem rg off) ∧
      (¬Needed c → result.bytes=before) :=
  KernelOnly.native_tensor c v A B requestedMax checked before observed storage labels launchScalar ops out

#axiomsClean correct_real
#axiomsClean correct_fp32
#axiomsClean input_tensor
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.TensorBridge
