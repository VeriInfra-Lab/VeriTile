import bench.repairs.tritonbench_g.attention_kernel.Kernels
import bench.repairs.tritonbench_g.attention_kernel.Proofs.Contract
import bench.repairs.tritonbench_g.attention_kernel.Proofs.AcceptedFP
namespace VeriTile.Bench.Repairs.AttentionKernel.FPEquiv
open VeriTile Triton FP.Structural
open Whole001 Source Whole Machine Whole002
noncomputable section
abbrev reference := FamilyNative.reference
specification attention_native (p : Parameters) (bases : Bases)
    (args : Contract.Arguments) (admitted : Contract.InputContract p bases args) (plan : NativeRouting.CompilerPlan p (Contract.abi p args admitted.1)) (before : FamilyReturned.InputWords) (storage : TensorStorage p)
    (fits : storage.capacity=before.capacity) (work : List Program)
    (legal : (programs (grid p)).Perm work) :
    Spec.Real (Contract.check p bases args=.ok () ∧ ∃ result,Physical.invoke (fun _=>(FamilyNative.algebra (plan.layout args.scaleWord))) (fun _=>(FamilyNative.dotModel (plan.layout args.scaleWord)))
        p (Contract.abi p args admitted.1) (before.symbolic p) work=some result ∧ result.result=() ∧
      result.memory.capacity=before.capacity ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<p.dim),
        ∃ index : Fin (result.memory.capacity p.Out),index.val=logicalAddress p b h n d ∧
          result.memory.cells p.Out index=.mk p.storage.tile (ofFloat p.storage.float
            (answer (FamilyNative.algebra (plan.layout args.scaleWord)) (FamilyNative.dotModel (plan.layout args.scaleWord)) p
              (start (grid p) (n/p.bm,b*p.heads+h,0) (before.symbolic p).extend) (lane p (Contract.positive_block p bases args admitted) n d hd))) ∧
          FamilyReturned.observe (plan.layout args.scaleWord) p result.memory p.Out index d=FamilyNative.reference (plan.layout args.scaleWord) p
            (start (grid p) (n/p.bm,b*p.heads+h,0) (before.symbolic p).extend) (lane p (Contract.positive_block p bases args admitted) n d hd)) ∧
      (∀ r (i : Fin (before.capacity r)),Physical.Exterior p r i.val →
        ∃ j : Fin (result.memory.capacity r),j.val=i.val ∧
          result.memory.cells r j=(before.symbolic p).cells r i) ∧
      (Live p → Access.Safe p storage) ∧ Closed p bases) := by
  refine ⟨(Contract.check_iff p bases args).mpr admitted,?_⟩
  exact FamilyReturned.returned_native (plan.layout args.scaleWord) p bases (Contract.domain p bases args admitted) (Contract.abi p args admitted.1) before storage fits (Contract.positive_block p bases args admitted) work legal


/-- Numerical transport is conditional only on scalar and fixed-MMA instruction laws. -/
theorem attention_local_numerical_transport {α β : Type} (f : α→β) (A : Native.ISA α) (B : Native.ISA β)
    (laws : Native.LocalTransport f A B) (p : Parameters)
    (bases : Bases) (args : Contract.Arguments) (admitted : Contract.InputContract p bases args) (plan : NativeRouting.CompilerPlan p (Contract.abi p args admitted.1))
    (before : FamilyReturned.InputWords) (storage : TensorStorage p) (fits : storage.capacity=before.capacity)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    ∃ result,Physical.invoke (fun _=>(FamilyNative.algebra (plan.layout args.scaleWord))) (fun _=>(FamilyNative.dotModel (plan.layout args.scaleWord)))
        p (Contract.abi p args admitted.1) (before.symbolic p) work=some result ∧ result.result=() ∧
      result.memory.capacity=before.capacity ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<p.dim),
        ∃ index : Fin (result.memory.capacity p.Out),index.val=logicalAddress p b h n d ∧
          f ((FamilyReturned.observe (plan.layout args.scaleWord) p result.memory p.Out index d).eval A)=
            (FamilyNative.reference (plan.layout args.scaleWord) p (start (grid p) (n/p.bm,b*p.heads+h,0) (before.symbolic p).extend)
              (lane p (Contract.positive_block p bases args admitted) n d hd)).eval B) ∧
      (∀ r (i : Fin (before.capacity r)),Physical.Exterior p r i.val →
        ∃ j : Fin (result.memory.capacity r),j.val=i.val ∧ result.memory.cells r j=(before.symbolic p).cells r i) ∧
      (Live p → Access.Safe p storage) ∧ Closed p bases :=
  FamilyReturned.local_numerical_return f A B laws (plan.layout args.scaleWord) p bases (Contract.domain p bases args admitted) (Contract.abi p args admitted.1) before storage fits (Contract.positive_block p bases args admitted) work legal


#axiomsClean attention_local_numerical_transport
#specNonCircular reference avoiding [Physical.invoke,Whole.invoke,FP.Structural.exec]
#axiomsClean attention_native
end
end VeriTile.Bench.Repairs.AttentionKernel.FPEquiv
