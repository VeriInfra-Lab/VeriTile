import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.CheckedBridge

namespace VeriTile.Bench.Repairs.ContextAttnFwd.Checked
open VeriTile VeriTile.Triton FP.Structural NativeGraph Returned
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000
local instance (c : Host.Call) : Decidable (Needed c) := Classical.propDecidable _

/-- Arbitrary Real source memory, with the original signed metadata and no
finite-word encoding. The independent target retains the actual finite mask
sentinel, cached prefix, grouped heads, padded keys and host-scale order. -/
theorem whole_mathematical_real (c : Host.Call) (v : Views) (A : FlatAlloc)
    (start sequence prompt : Nat → Int)
    (checked : checkInputs c v A.extent start sequence prompt=Bool.true)
    (s : BlockState) (inputMetadata : Geometry.Metadata c s)
    (labels : FPSuccess.Separate c) (allocation : A.Disjoint)
    (covered : ∀ rg,rg∉A.regions → A.extent rg=0) :
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
  have guard := (checkInputs_iff c v A.extent start sequence prompt).mp checked
  exact RealProof.complete c A s (geometry c A.extent guard.2.1)
    inputMetadata labels allocation covered

theorem empty_whole (c : Host.Call) (A : FlatAlloc) (B : Native.Bytes A.extent)
    (ops : ScalarOps) (ctx : Context) (m : Physical.Memory) (empty : ¬Needed c)
    (metadata : Snapshot.Metadata c B m) : WholeCall c A B ops ctx m := by
  obtain ⟨result,run,bytes,noneRet,object,device,launch,graph⟩ := empty_return c B ops ctx m empty
  have lengths := empty_lengths c empty
  refine ⟨result,run,noneRet,object,device,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩
  · simpa only [if_neg empty] using launch
  · rw [order,if_neg empty,FPSuccess.runGrid,graph]
  · rw [graph]
    exact Snapshot.good c B m
  · rw [graph]
    exact Snapshot.metadata c B m metadata
  · intro i hi
    have h := lengths _ (Host.batch_bound c i.1)
    change _<c.length _ at hi
    omega
  · intro a _; rw [bytes]
  · intro rg _ off _ j; rw [bytes]
  · intro rg off _; rw [graph]
  · intro _; exact bytes

/-- Complete raw FP32 source graph, native index/control/address facts and
caller bytes from the same finite checked contract. Only ordinary input
metadata/storage and static role-label interfaces remain separate premises. -/
theorem whole_fp32_bytes (c : Host.Call) (v : Views) (A : FlatAlloc) (B : Native.Bytes A.extent)
    (start sequence prompt : Nat → Int)
    (checked : checkInputs c v A.extent start sequence prompt=Bool.true)
    (before : Physical.Memory) (observed : MetadataObservation c B before start sequence prompt)
    (storage : StorageBinding c v B) (roles : Snapshot.LogicalRoles c)
    (labels : FPSuccess.Separate c) (ops : ScalarOps) (ctx : Context) :
    WholeCall c A B ops ctx before ∧
      MachineFacts c B before (abi c v B before
        ((checkInputs_iff c v A.extent start sequence prompt).mp checked).2.2.1
        (metadata c B before start sequence prompt
          ((checkInputs_iff c v A.extent start sequence prompt).mp checked).2.2.2 observed)) := by
  have guard := (checkInputs_iff c v A.extent start sequence prompt).mp checked
  have inputMetadata := metadata c B before start sequence prompt guard.2.2.2 observed
  have geometry := geometry c A.extent guard.2.1
  have native := abi c v B before guard.2.2.1 inputMetadata
  have formed := formed c v guard.2.2.1
  constructor
  · by_cases needed : Needed c
    · exact (whole_bytes c A B ops ctx before geometry inputMetadata roles labels
        (physical_separate c v B guard.1 geometry storage needed) native formed).1
    · exact empty_whole c A B ops ctx before needed inputMetadata
  · exact machine_facts c B before geometry inputMetadata roles labels native formed

/-- Instruction-local laws transport the full caller-byte result. The nine
families include fixed k8 TF32 instructions, explicit FMA and exp2 primitives;
no arbitrary dot, reduction, or whole-result equality is a premise. -/
theorem whole_fp32_if_local (c : Host.Call) (v : Views) (A : FlatAlloc) (B : Native.Bytes A.extent)
    (start sequence prompt : Nat → Int)
    (checked : checkInputs c v A.extent start sequence prompt=Bool.true)
    (before : Physical.Memory) (observed : MetadataObservation c B before start sequence prompt)
    (storage : StorageBinding c v B) (roles : Snapshot.LogicalRoles c)
    (labels : FPSuccess.Separate c) (actual reference : ScalarOps)
    (laws : Numerical.ScalarLaws actual reference) (ctx : Context) :
    WholeCall c A B actual ctx before ∧ WholeCall c A B reference ctx before ∧
      invoke c B actual ctx before=invoke c B reference ctx before := by
  have h := (whole_fp32_bytes c v A B start sequence prompt checked before observed storage roles labels actual ctx).1
  have equal := laws.operations_eq actual reference
  subst reference
  exact ⟨h,h,rfl⟩

#axiomsClean whole_mathematical_real
#axiomsClean whole_fp32_bytes
#axiomsClean whole_fp32_if_local
#specNonCircular VeriTile.Bench.TritonBenchG.ContextAttnFwd.Formula.expected
  avoiding [invoke,FPSuccess.runGrid,Physical.stores]
end
end VeriTile.Bench.Repairs.ContextAttnFwd.Checked
