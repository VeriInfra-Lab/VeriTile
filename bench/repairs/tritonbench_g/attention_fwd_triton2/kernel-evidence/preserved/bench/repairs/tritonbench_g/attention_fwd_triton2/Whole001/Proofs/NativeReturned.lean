import bench.repairs.tritonbench_g.attention_fwd_triton2.Whole001.Proofs.Physical
import bench.repairs.tritonbench_g.attention_fwd_triton2.Whole001.Proofs.NativeStaging
namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.Whole001.NativeReturned
open VeriTile Triton FP.Structural Source Whole Machine
noncomputable section
set_option maxHeartbeats 4000000

/-- Native input leaves are actual FP32 words, not arbitrary graph or dot oracles.
Read-only aliases are represented by the same region words. -/
structure InputWords where
  capacity : RegionName → Nat
  bits : (r : RegionName) → Fin (capacity r) → BitVec 32

def InputWords.symbolic (m : InputWords) : Physical.Store Native.Expr :=
  ⟨m.capacity,(fun r i=>.mk .real (fun _=>.ready (.bits (m.bits r i)))),
    .mk .real (fun _=>.ready (.bits 0))⟩

def observe (memory : Physical.Store Native.Expr) (r : RegionName)
    (i : Fin (memory.capacity r)) (d : Nat) : Native.Graph :=
  Native.lower ((memory.cells r i).read .bf16 (Native.physicalOwner d))

theorem returned_native (p : Parameters) (bases : Bases) (domain : Domain p bases)
    (abi : InputABI p) (before : InputWords) (fits : Physical.InputFits p before.symbolic)
    (alloc : Allocation p abi before.symbolic.extend) (work : List Program)
    (legal : (programs (grid p)).Perm work) :
    ∃ result,Physical.invoke (fun _=>Native.algebra) (fun _=>Native.dotModel)
        p abi before.symbolic alloc work=some result ∧ result.tensor=alloc.output ∧
      result.memory.capacity p.Out=words p ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<96),
        ∃ index : Fin (result.memory.capacity p.Out),index.val=logicalAddress p b h n d ∧
          result.memory.cells p.Out index=.mk .bf16
            (answer Native.algebra Native.dotModel p
              (start (grid p) (n/128,b*p.heads+h,0) alloc.state) (lane n d hd)) ∧
          observe result.memory p.Out index d=Native.reference p
            (start (grid p) (n/128,b*p.heads+h,0) alloc.state) (lane n d hd)) ∧
      (∀ r (i : Fin (before.capacity r)),r≠p.Out → r≠alloc.unusedM.region →
        ∃ j : Fin (result.memory.capacity r),j.val=i.val ∧
          result.memory.cells r j=before.symbolic.cells r i) ∧
      Access.Safe p ∧ Closed p bases ∧ Physical.InputAccess p before.symbolic := by
  obtain ⟨result,hr,ht,hcap,hv,hf,ha,hc,hinput⟩:=Physical.finite_return
    (fun _=>Native.algebra) (fun _=>Native.dotModel) p bases domain abi before.symbolic fits alloc work legal
  refine ⟨result,hr,ht,hcap,?_,hf,ha,hc,hinput⟩
  intro b hb h hh n hn d hd
  obtain ⟨index,hi,hvalue⟩:=hv b hb h hh n hn d hd
  refine ⟨index,hi,hvalue,?_⟩
  simp only [observe,hvalue,Cell.read_same,Native.reference,lane]

theorem local_numerical_return {α β : Type} (f : α→β) (A : Native.ISA α) (B : Native.ISA β)
    (laws : Native.LocalTransport f A B)
    (p : Parameters) (bases : Bases) (domain : Domain p bases) (abi : InputABI p)
    (before : InputWords) (fits : Physical.InputFits p before.symbolic)
    (alloc : Allocation p abi before.symbolic.extend) (work : List Program)
    (legal : (programs (grid p)).Perm work) :
    ∃ result,Physical.invoke (fun _=>Native.algebra) (fun _=>Native.dotModel)
        p abi before.symbolic alloc work=some result ∧ result.tensor=alloc.output ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<96),
        ∃ index : Fin (result.memory.capacity p.Out),index.val=logicalAddress p b h n d ∧
          f ((observe result.memory p.Out index d).eval A)=
            (Native.reference p (start (grid p) (n/128,b*p.heads+h,0) alloc.state) (lane n d hd)).eval B) ∧
      Access.Safe p ∧ Closed p bases ∧ Physical.InputAccess p before.symbolic := by
  obtain ⟨result,hr,ht,hcap,hv,hf,ha,hc,hinput⟩:=returned_native p bases domain abi before fits alloc work legal
  refine ⟨result,hr,ht,?_,ha,hc,hinput⟩
  intro b hb h hh n hn d hd
  obtain ⟨index,hi,hvalue,hgraph⟩:=hv b hb h hh n hn d hd
  refine ⟨index,hi,?_⟩
  rw [hgraph]
  exact Native.transport f A B laws _

#axiomsClean returned_native
#axiomsClean local_numerical_return
#specNonCircular InputWords avoiding [FP.Structural.exec,Whole.invoke,Physical.invoke,Whole.answer]
end
end VeriTile.Bench.Repairs.AttentionFwdTriton2.Whole001.NativeReturned
