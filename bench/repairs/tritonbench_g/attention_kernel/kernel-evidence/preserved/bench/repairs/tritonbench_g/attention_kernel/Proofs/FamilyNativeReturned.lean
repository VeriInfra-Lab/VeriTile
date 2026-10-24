import bench.repairs.tritonbench_g.attention_kernel.Proofs.Physical
import bench.repairs.tritonbench_g.attention_kernel.Proofs.FamilyNativeStaging
namespace VeriTile.Bench.Repairs.AttentionKernel.Whole002.FamilyReturned
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionKernel.Whole001 Source Whole Machine
noncomputable section
set_option maxHeartbeats 4000000

/-- Actual input storage words. Read-only aliases share the same region cells. -/
structure InputWords where
  capacity : RegionName→ Nat
  bits : (r : RegionName)→ Fin (capacity r)→ BitVec 16

def InputWords.symbolic (p : Parameters) (m : InputWords) : Physical.Store FamilyNative.Expr :=
  ⟨m.capacity,(fun r i=>.mk p.storage.tile (ofFloat p.storage.float
    (fun _=>.ready (.bits ((m.bits r i).zeroExtend 32))))),
    .mk p.storage.tile (ofFloat p.storage.float (fun _=>.ready (.bits 0)))⟩

def observe (plan : FamilyNative.Layout) (p : Parameters) (memory : Physical.Store FamilyNative.Expr) (r : RegionName)
    (i : Fin (memory.capacity r)) (d : Nat) : Native.Graph :=
  Native.lower (toFloat p.storage.float ((memory.cells r i).read p.storage.tile) (FamilyNative.physicalOwner plan d))

theorem float_roundtrip {α : Type} (dtype : FloatDType) (a : α) :
    toFloat dtype (ofFloat dtype a)=a := by cases dtype <;> rfl

theorem returned_native (plan : FamilyNative.Layout) (p : Parameters) (bases : Bases) (domain : Domain p bases)
    (abi : InputABI p) (before : InputWords) (storage : TensorStorage p)
    (fits : storage.capacity=before.capacity) (hbm : 0<p.bm) (work : List Program)
    (legal : (programs (grid p)).Perm work) :
    ∃ result,Physical.invoke (fun _=>(FamilyNative.algebra plan)) (fun _=>(FamilyNative.dotModel plan))
        p abi (before.symbolic p) work=some result ∧ result.result=() ∧
      result.memory.capacity=before.capacity ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<p.dim),
        ∃ index : Fin (result.memory.capacity p.Out),index.val=logicalAddress p b h n d ∧
          result.memory.cells p.Out index=.mk p.storage.tile (ofFloat p.storage.float
            (answer (FamilyNative.algebra plan) (FamilyNative.dotModel plan) p
              (start (grid p) (n/p.bm,b*p.heads+h,0) (before.symbolic p).extend) (lane p hbm n d hd))) ∧
          observe plan p result.memory p.Out index d=FamilyNative.reference plan p
            (start (grid p) (n/p.bm,b*p.heads+h,0) (before.symbolic p).extend) (lane p hbm n d hd)) ∧
      (∀ r (i : Fin (before.capacity r)),Physical.Exterior p r i.val →
        ∃ j : Fin (result.memory.capacity r),j.val=i.val ∧
          result.memory.cells r j=(before.symbolic p).cells r i) ∧
      (Live p → Access.Safe p storage) ∧ Closed p bases := by
  obtain ⟨result,hr,hn,hcap,hv,hf,ha,hc⟩:=Physical.finite_return
    (fun _=>(FamilyNative.algebra plan)) (fun _=>(FamilyNative.dotModel plan)) p bases domain abi (before.symbolic p) storage fits hbm work legal
  refine ⟨result,hr,hn,hcap,?_,hf,ha,hc⟩
  intro b hb h hh n hn d hd
  obtain ⟨index,hi,hvalue⟩:=hv b hb h hh n hn d hd
  refine ⟨index,hi,hvalue,?_⟩
  simp only [observe,hvalue,Cell.read_same,float_roundtrip,FamilyNative.reference,lane]

theorem local_numerical_return {α β : Type} (f : α→β) (A : Native.ISA α) (B : Native.ISA β)
    (laws : Native.LocalTransport f A B) (plan : FamilyNative.Layout)
    (p : Parameters) (bases : Bases) (domain : Domain p bases) (abi : InputABI p)
    (before : InputWords) (storage : TensorStorage p) (fits : storage.capacity=before.capacity)
    (hbm : 0<p.bm) (work : List Program) (legal : (programs (grid p)).Perm work) :
    ∃ result,Physical.invoke (fun _=>(FamilyNative.algebra plan)) (fun _=>(FamilyNative.dotModel plan))
        p abi (before.symbolic p) work=some result ∧ result.result=() ∧
      result.memory.capacity=before.capacity ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<p.dim),
        ∃ index : Fin (result.memory.capacity p.Out),index.val=logicalAddress p b h n d ∧
          f ((observe plan p result.memory p.Out index d).eval A)=
            (FamilyNative.reference plan p (start (grid p) (n/p.bm,b*p.heads+h,0) (before.symbolic p).extend)
              (lane p hbm n d hd)).eval B) ∧
      (∀ r (i : Fin (before.capacity r)),Physical.Exterior p r i.val →
        ∃ j : Fin (result.memory.capacity r),j.val=i.val ∧ result.memory.cells r j=(before.symbolic p).cells r i) ∧
      (Live p → Access.Safe p storage) ∧ Closed p bases := by
  obtain ⟨result,hr,hn,hcap,hv,hf,ha,hc⟩:=returned_native plan p bases domain abi before storage fits hbm work legal
  refine ⟨result,hr,hn,hcap,?_,hf,ha,hc⟩
  intro b hb h hh n hn d hd
  obtain ⟨index,hi,hvalue,hgraph⟩:=hv b hb h hh n hn d hd
  refine ⟨index,hi,?_⟩
  rw [hgraph]
  exact Native.transport f A B laws _

#axiomsClean returned_native
#axiomsClean local_numerical_return
#specNonCircular InputWords avoiding [FP.Structural.exec,Whole.invoke,Physical.invoke,Whole.answer]
end
end VeriTile.Bench.Repairs.AttentionKernel.Whole002.FamilyReturned
