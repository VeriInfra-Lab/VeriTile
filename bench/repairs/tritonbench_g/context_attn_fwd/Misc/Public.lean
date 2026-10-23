import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.MachineCertificate

namespace VeriTile.Bench.Repairs.ContextAttnFwd
open VeriTile VeriTile.Triton FP.Structural NativeGraph
open Returned
noncomputable section
local instance (c : Host.Call) : Decidable (Needed c) := Classical.propDecidable _

/-- Full normal host result, including the exact supplied Out handle, None,
restored caller device, actual skipped-launch branch, whole graph and bytes. -/
def WholeCall (c : Host.Call) (A : FlatAlloc) (B : Native.Bytes A.extent)
    (ops : ScalarOps) (ctx : Context) (before : Physical.Memory) : Prop :=
    ∃ result,invoke c B ops ctx before=some result ∧
      result.returned=none ∧ result.suppliedOutput=ctx.suppliedOutput ∧
      result.currentDevice=ctx.callerDevice ∧
      result.launchDevice=(if Needed c then some ctx.ownerDevice else none) ∧
      FPSuccess.runGrid c (backend (NativeSM90.plan c.layout.BM c.layout.D)) (order c)
        (Snapshot.initial c B before)=some result.graph ∧
      GoodMemory result.graph ∧ FPSuccess.Metadata c result.graph ∧
      (∀ i : Physical.Lane c,Physical.active c i →
        result.bytes (Physical.address c B i)=Physical.byte (Physical.outputWord c ops result.graph i) i.2.2) ∧
      (∀ a,(¬∃ i : Physical.Lane c,Physical.active c i ∧ Physical.address c B i=a) → result.bytes a=before a) ∧
      (∀ rg,rg∈Physical.reads c → ∀ off,off<A.extent rg → ∀ j : Fin 4,
        result.bytes (B.base rg+4*off+j.val)=before (B.base rg+4*off+j.val)) ∧
      (∀ rg off,¬FPSuccess.writes c (order c) rg off →
        result.graph.mem rg off=(Snapshot.initial c B before).mem rg off) ∧
      (¬Needed c → result.bytes=before)

/-- Whole-source native graph/bytes theorem for the concrete SM90 instruction
plan. Compiler correspondence is checked on the pinned offline profiles;
this theorem does not assert a universal theorem about arbitrary compilers. -/
theorem whole_bytes (c : Host.Call) (A : FlatAlloc) (B : Native.Bytes A.extent)
    (ops : ScalarOps) (ctx : Context) (before : Physical.Memory)
    (hc : Geometry.Checked c A.extent) (hm : Snapshot.Metadata c B before)
    (roles : Snapshot.LogicalRoles c) (sep : FPSuccess.Separate c)
    (physicalSep : Physical.Separate c B) (abi : Native.ABI c) (formed : FormedBounds.Checked c) :
    WholeCall c A B ops ctx before ∧ MachineFacts c B before abi := by
  exact ⟨Returned.whole c A (Snapshot.realInitial c B before) B ops ctx before hc
    (Snapshot.real_metadata c B before hm) hm sep physicalSep,
    machine_facts c B before hc hm roles sep abi formed⟩

/-- Nine device-local all-word instruction laws transport the complete host
result. Host.scale separately retains the three actual Python binary64 scalar
operations in HostArithmetic. There is no native-to-Real equality claim. -/
theorem whole_equiv (c : Host.Call) (A : FlatAlloc) (B : Native.Bytes A.extent)
    (actual reference : ScalarOps) (laws : Numerical.ScalarLaws actual reference)
    (ctx : Context) (before : Physical.Memory)
    (hc : Geometry.Checked c A.extent) (hm : Snapshot.Metadata c B before)
    (roles : Snapshot.LogicalRoles c) (sep : FPSuccess.Separate c)
    (physicalSep : Physical.Separate c B) (abi : Native.ABI c) (formed : FormedBounds.Checked c) :
    WholeCall c A B actual ctx before ∧ WholeCall c A B reference ctx before ∧
      invoke c B actual ctx before=invoke c B reference ctx before ∧ MachineFacts c B before abi := by
  have he := laws.operations_eq actual reference
  subst reference
  obtain ⟨hcall,hmachine⟩ := whole_bytes c A B actual ctx before hc hm roles sep physicalSep abi formed
  exact ⟨hcall,hcall,rfl,hmachine⟩

/-- Independent finite-sentinel Real target for the same call and physical
initial bytes under Snapshot.real32. Nonfinite words have an explicit total
Real abstraction; native exceptional words remain unaltered in whole_bytes. -/
theorem correct (c : Host.Call) (A : FlatAlloc) (B : Native.Bytes A.extent)
    (before : Physical.Memory) (hc : Geometry.Checked c A.extent)
    (hm : Snapshot.Metadata c B before) (roles : Snapshot.LogicalRoles c)
    (sep : FPSuccess.Separate c) (allocation : A.Disjoint)
    (covered : ∀ rg,rg∉A.regions → A.extent rg=0) :
    let s := Snapshot.realInitial c B before
    Spec.Real (∃ final,
      Nonempty (Kernel.GridLaunchedOrdinary
        (VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.flatSource c.args c.regions A Bool.false)
        c.grid (A.flattenState s) final) ∧
      (∀ idx i,VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.active c.args c.regions s idx i →
        final.readMem
          (VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.address c.args c.regions A s idx i).1
          (VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.address c.args c.regions A s idx i).2 =
        VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.result c.args c.regions s idx i) ∧
      BlockState.WriteWithin
        (VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.writes c.args c.regions A s)
        (A.flattenState s) final ∧
      (∀ rg off,c.regions.out≠rg → off<A.extent rg →
        final.mem A.flat (A.addr rg off)=(A.flattenState s).mem A.flat (A.addr rg off)) ∧
      (VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.returned final).2=none) := by
  exact RealProof.complete c A (Snapshot.realInitial c B before) hc
    (Snapshot.real_metadata c B before hm) sep allocation covered

#print axioms correct
#print axioms whole_bytes
#print axioms whole_equiv
end
end VeriTile.Bench.Repairs.ContextAttnFwd
