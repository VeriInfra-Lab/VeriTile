import bench.repairs.tritonbench_g.attention_kernel.Misc.KernelResults
import bench.repairs.tritonbench_g.attention_kernel.KernelOnly.Endpoints
import bench.repairs.tritonbench_g.attention_kernel.Misc.AcceptedReal
import bench.repairs.tritonbench_g.attention_kernel.Misc.AcceptedFP

/-! Optional checked Tensor applications of the accepted kernel endpoints. -/
namespace VeriTile.Bench.Repairs.AttentionKernel.TensorBridge
open VeriTile Triton FP.Structural
open Whole001 Source Whole Machine Whole002
noncomputable section

namespace Correct
open KernelOnly

theorem empty_kernel {α β : Type} [Inhabited α] (M : Algebra α) (dot : Dot α)
    (c : Call) (hc : InputContract c) (empty : ¬Live (parameters c))
    (before : State α) (work : List Program) (order : (programs (grid (parameters c))).Perm work)
    (decode : α → β) (words : Nat → β) :
    work=[] ∧ runPrograms (fun _=>M) (fun _=>dot) (parameters c) (grid (parameters c)) work before=some before ∧
      ∀ a,observe decode c words before a=words a :=
  KernelOnly.empty_kernel M dot c hc empty before work order decode words

open KernelOnly

theorem kernel_correct (c : Call) (hc : InputContract c) (laws : TensorLaws c)
    (read : Nat → ℝ) (work : List Program)
    (order : (programs (grid (parameters c))).Perm work) :
    ∃ after,runPrograms (fun _=>Real.algebra) (fun _=>Real.dotModel)
        (parameters c) (grid (parameters c)) work (realEntry c read)=some after ∧
      (∀ b<(parameters c).batches,∀ h<(parameters c).heads,∀ n<(parameters c).length,
        ∀ d<(parameters c).dim,
        logical (observe realDecode c read after) c .output b h n d=
          mathematicalTarget c read b h n d) ∧
      RawFrame c (realEntry c read) after ∧
      (∀ a,¬Stored c a → observe realDecode c read after a=read a) ∧
      (∀ r∈reads,∀ b h n d,Valid (parameters c) r b h n d →
        logical (observe realDecode c read after) c r b h n d=logical read c r b h n d) ∧
      Closed (parameters c) (pointers c) :=
  KernelOnly.kernel_real c hc laws read work order

theorem attention_real (p : Parameters) (bases : Bases)
    (args : Contract.Arguments) (admitted : Contract.InputContract p bases args) (before : RealReturned.InputValues) (storage : TensorStorage p)
    (fits : storage.capacity=before.capacity)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    Spec.Real (Contract.check p bases args=.ok () ∧ ∃ result,Physical.invoke (fun _=>Real.algebra) (fun _=>Real.dotModel) p (Contract.abi p args admitted.1) (before.store p) work=some result ∧
      result.result=() ∧ result.memory.capacity=before.capacity ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<p.dim),
        ∃ index : Fin (result.memory.capacity p.Out),index.val=logicalAddress p b h n d ∧
          result.memory.cells p.Out index=.mk p.storage.tile
            (ofFloat p.storage.float (some (RealReturned.target p before.logical b h n d)))) ∧
      (∀ r (i : Fin (before.capacity r)),Physical.Exterior p r i.val →
        ∃ j : Fin (result.memory.capacity r),j.val=i.val ∧ result.memory.cells r j=(before.store p).cells r i) ∧
      (Live p → Access.Safe p storage) ∧ Closed p bases) := by
  refine ⟨(Contract.check_iff p bases args).mpr admitted,?_⟩
  exact RealReturned.returned_real p bases (Contract.domain p bases args admitted) (Contract.abi p args admitted.1) (before.store p) storage fits (Contract.positive_block p bases args admitted) before.logical
    (RealReturned.input_values_reads p before) work legal

#axiomsClean attention_real
end Correct

namespace FPEquiv
abbrev kernel_reference := KernelOnly.nativeReference
open KernelOnly

theorem kernel_native (I : Instructions) (c : Call) (hc : InputContract c) (laws : TensorLaws c)
    (plan : NativeRouting.CompilerPlan (parameters c) (Contract.abi _ _ hc.1))
    (words : Nat → BitVec 16) (work : List Program)
    (order : (programs (grid (parameters c))).Perm work) :
    let layout := plan.layout c.scaleWord
    let before := entry c (Specification.nativeReader c words)
    ∃ after,runPrograms (fun _=>FamilyNative.algebra layout) (fun _=>FamilyNative.dotModel layout)
        (parameters c) (grid (parameters c)) work before=some after ∧
      (∀ b<(parameters c).batches,∀ h<(parameters c).heads,∀ n<(parameters c).length,
        ∀ d (hd : d<(parameters c).dim),
        logical (observe (nativeDecode I layout d) c words after) c .output b h n d=
          nativeReference I layout c words b h n d (Contract.positive_block _ _ _ hc) hd) ∧
      RawFrame c before after ∧
      (∀ d a,¬Stored c a → observe (nativeDecode I layout d) c words after a=words a) ∧
      (∀ owner r, r∈reads → ∀ b h n d,Valid (parameters c) r b h n d →
        logical (observe (nativeDecode I layout owner) c words after) c r b h n d=logical words c r b h n d) ∧
      Closed (parameters c) (pointers c) :=
  KernelOnly.kernel_native I c hc laws plan words work order

theorem attention_native (p : Parameters) (bases : Bases)
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

#axiomsClean attention_native
#axiomsClean attention_local_numerical_transport
end FPEquiv
end
end VeriTile.Bench.Repairs.AttentionKernel.TensorBridge
