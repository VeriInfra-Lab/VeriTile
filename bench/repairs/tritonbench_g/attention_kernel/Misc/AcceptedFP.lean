import bench.repairs.tritonbench_g.attention_kernel.Proofs.FamilyNativeReturned
import bench.repairs.tritonbench_g.attention_kernel.Proofs.FamilyNativeRouting
namespace VeriTile.Bench.Repairs.AttentionKernel.Whole002
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionKernel.Whole001 Source Whole Machine
noncomputable section

/-- Complete source-to-native-graph execution, caller-output cells and frame,
without numerical-equality assumptions. The finite plan family covers every guarded wrapper configuration. -/
theorem whole_returned_native (p : Parameters) (bases : Bases) (domain : Domain p bases)
    (abi : InputABI p) (plan : NativeRouting.CompilerPlan p abi) (scaleWord : BitVec 32) (before : FamilyReturned.InputWords) (storage : TensorStorage p)
    (fits : storage.capacity=before.capacity) (hbm : 0<p.bm) (work : List Program)
    (legal : (programs (grid p)).Perm work) :
    ∃ result,Physical.invoke (fun _=>(FamilyNative.algebra (plan.layout scaleWord))) (fun _=>(FamilyNative.dotModel (plan.layout scaleWord)))
        p abi (before.symbolic p) work=some result ∧ result.result=() ∧
      result.memory.capacity=before.capacity ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<p.dim),
        ∃ index : Fin (result.memory.capacity p.Out),index.val=logicalAddress p b h n d ∧
          result.memory.cells p.Out index=.mk p.storage.tile (ofFloat p.storage.float
            (answer (FamilyNative.algebra (plan.layout scaleWord)) (FamilyNative.dotModel (plan.layout scaleWord)) p
              (start (grid p) (n/p.bm,b*p.heads+h,0) (before.symbolic p).extend) (lane p hbm n d hd))) ∧
          FamilyReturned.observe (plan.layout scaleWord) p result.memory p.Out index d=FamilyNative.reference (plan.layout scaleWord) p
            (start (grid p) (n/p.bm,b*p.heads+h,0) (before.symbolic p).extend) (lane p hbm n d hd)) ∧
      (∀ r (i : Fin (before.capacity r)),Physical.Exterior p r i.val →
        ∃ j : Fin (result.memory.capacity r),j.val=i.val ∧
          result.memory.cells r j=(before.symbolic p).cells r i) ∧
      (Live p → Access.Safe p storage) ∧ Closed p bases :=
  FamilyReturned.returned_native (plan.layout scaleWord) p bases domain abi before storage fits hbm work legal

/-- Finite caller-output native graph and local-ISA transport. Numerical
conformance is an explicit scalar/fixed-MMA obligation, never a whole-output,
softmax, reduction, dot or execution-equivalence premise. -/
theorem whole_local_numerical_transport {α β : Type} (f : α→β) (A : Native.ISA α) (B : Native.ISA β)
    (laws : Native.LocalTransport f A B) (p : Parameters)
    (bases : Bases) (domain : Domain p bases) (abi : InputABI p) (plan : NativeRouting.CompilerPlan p abi) (scaleWord : BitVec 32)
    (before : FamilyReturned.InputWords) (storage : TensorStorage p) (fits : storage.capacity=before.capacity)
    (hbm : 0<p.bm) (work : List Program) (legal : (programs (grid p)).Perm work) :
    ∃ result,Physical.invoke (fun _=>(FamilyNative.algebra (plan.layout scaleWord))) (fun _=>(FamilyNative.dotModel (plan.layout scaleWord)))
        p abi (before.symbolic p) work=some result ∧ result.result=() ∧
      result.memory.capacity=before.capacity ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<p.dim),
        ∃ index : Fin (result.memory.capacity p.Out),index.val=logicalAddress p b h n d ∧
          f ((FamilyReturned.observe (plan.layout scaleWord) p result.memory p.Out index d).eval A)=
            (FamilyNative.reference (plan.layout scaleWord) p (start (grid p) (n/p.bm,b*p.heads+h,0) (before.symbolic p).extend)
              (lane p hbm n d hd)).eval B) ∧
      (∀ r (i : Fin (before.capacity r)),Physical.Exterior p r i.val →
        ∃ j : Fin (result.memory.capacity r),j.val=i.val ∧ result.memory.cells r j=(before.symbolic p).cells r i) ∧
      (Live p → Access.Safe p storage) ∧ Closed p bases :=
  FamilyReturned.local_numerical_return f A B laws (plan.layout scaleWord) p bases domain abi before storage fits hbm work legal

#axiomsClean whole_returned_native
#axiomsClean whole_local_numerical_transport
#specNonCircular NativeRouting.CompilerPlan avoiding [FP.Structural.exec,FamilyNative.reference,Whole.invoke]
end
end VeriTile.Bench.Repairs.AttentionKernel.Whole002
