import bench.repairs.tritonbench_g.attention_fwd_triton2.Whole001.Proofs.NativeReturned
namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.Whole001
open VeriTile Triton FP.Structural Source Whole Machine
noncomputable section
/-- Same typed source/control/index/mask/state structure as Correct, specialized
to FP32 input-word leaves and the concrete SM80 numerical instruction graph.
No caller-provided dot, reduction, execution or output equality is required. -/
theorem whole_returned_native (p : Parameters) (bases : Bases) (domain : Domain p bases)
    (abi : InputABI p) (before : NativeReturned.InputWords) (fits : Physical.InputFits p before.symbolic)
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
          NativeReturned.observe result.memory p.Out index d=Native.reference p
            (start (grid p) (n/128,b*p.heads+h,0) alloc.state) (lane n d hd)) ∧
      (∀ r (i : Fin (before.capacity r)),r≠p.Out → r≠alloc.unusedM.region →
        ∃ j : Fin (result.memory.capacity r),j.val=i.val ∧
          result.memory.cells r j=before.symbolic.cells r i) ∧
      Access.Safe p ∧ Closed p bases ∧ Physical.InputAccess p before.symbolic := NativeReturned.returned_native p bases domain abi before fits alloc work legal

/-- Fourteen local ISA transport families, explicitly conditional. This does
not assert that native floating arithmetic equals exact Real attention. -/
theorem whole_local_numerical_transport {α β : Type} (f : α→β) (A : Native.ISA α) (B : Native.ISA β)
    (laws : Native.LocalTransport f A B)
    (p : Parameters) (bases : Bases) (domain : Domain p bases) (abi : InputABI p)
    (before : NativeReturned.InputWords) (fits : Physical.InputFits p before.symbolic)
    (alloc : Allocation p abi before.symbolic.extend) (work : List Program)
    (legal : (programs (grid p)).Perm work) :
    ∃ result,Physical.invoke (fun _=>Native.algebra) (fun _=>Native.dotModel)
        p abi before.symbolic alloc work=some result ∧ result.tensor=alloc.output ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<96),
        ∃ index : Fin (result.memory.capacity p.Out),index.val=logicalAddress p b h n d ∧
          f ((NativeReturned.observe result.memory p.Out index d).eval A)=
            (Native.reference p (start (grid p) (n/128,b*p.heads+h,0) alloc.state) (lane n d hd)).eval B) ∧
      Access.Safe p ∧ Closed p bases ∧ Physical.InputAccess p before.symbolic := NativeReturned.local_numerical_return f A B laws p bases domain abi before fits alloc work legal

#axiomsClean whole_returned_native
#axiomsClean whole_local_numerical_transport
end
end VeriTile.Bench.Repairs.AttentionFwdTriton2.Whole001
