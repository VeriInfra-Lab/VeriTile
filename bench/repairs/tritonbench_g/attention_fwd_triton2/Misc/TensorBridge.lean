import bench.repairs.tritonbench_g.attention_fwd_triton2.Proofs.RealCompatibility
import bench.repairs.tritonbench_g.attention_fwd_triton2.Proofs.NativeCompatibility
import bench.repairs.tritonbench_g.attention_fwd_triton2.Whole001.Correct
import bench.repairs.tritonbench_g.attention_fwd_triton2.Whole001.FPEquiv

/-! Optional historical checked Tensor applications and local numerical transport.
The primary entries retain the complete selected kernel endpoints. -/
namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.TensorBridge
open VeriTile Triton FP.Structural Checked Whole001 Whole001.Source Whole001.Whole Whole001.Machine
noncomputable section

/-- Original arbitrary-state raw grid consequences, including formed-address
closure and the complete tagged memory frame. -/
alias kernel_raw_correct := RealCompatibility.kernel_raw_correct
alias kernel_raw_native := NativeCompatibility.kernel_raw_native


/-- Actual full-source exact-Real execution, allocations, arbitrary legal grid order,
finite returned BF16-tagged tensor and frame, with an independent initial-input target.
This is the mathematical interpretation, not a claim of zero native rounding error. -/
theorem checked_real (p : Parameters) (bases : Bases) (denseOK : denseChecks p=Bool.true)
    (output : Live p → p.o=dense p) (setup : setupChecks p bases=Bool.true)
    (args : Arguments p) (inputs : inputChecks p args=Bool.true) (before : RealReturned.InputValues) (fits : Physical.InputFits p before.store)
    (alloc : Allocation p (admittedABI p args inputs) before.store.extend) (work : List Program)
    (legal : (programs (grid p)).Perm work) :
    ∃ result,Physical.invoke (fun _=>Real.algebra) (fun _=>Real.dotModel) p (admittedABI p args inputs) before.store alloc work=some result ∧
      result.tensor=alloc.output ∧ result.memory.capacity p.Out=words p ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<96),
        ∃ index : Fin (result.memory.capacity p.Out),index.val=logicalAddress p b h n d ∧
          result.memory.cells p.Out index=.mk .bf16 (some (RealReturned.target p before.logical b h n d))) ∧
      (∀ r (i : Fin (before.capacity r)),r≠p.Out → r≠alloc.unusedM.region →
        ∃ j : Fin (result.memory.capacity r),j.val=i.val ∧ result.memory.cells r j=before.store.cells r i) ∧
      Access.Safe p ∧ Closed p bases ∧ Physical.InputAccess p before.store :=
  RealReturned.returned_real p bases (admittedDomain p bases denseOK output setup) (admittedABI p args inputs) before.store fits alloc before.logical
    (RealReturned.input_values_reads p before) work legal


export KernelOnly (kernel_real kernel_native empty_kernel)
abbrev kernel_reference := KernelOnly.nativeReference

#axiomsClean checked_real

/-- Same typed source/control/index/mask/state structure as Correct, specialized
to FP32 input-word leaves and the concrete SM80 numerical instruction graph.
No caller-provided dot, reduction, execution or output equality is required. -/
theorem checked_native (p : Parameters) (bases : Bases) (denseOK : denseChecks p=Bool.true)
    (output : Live p → p.o=dense p) (setup : setupChecks p bases=Bool.true)
    (args : Arguments p) (inputs : inputChecks p args=Bool.true) (before : NativeReturned.InputWords) (fits : Physical.InputFits p before.symbolic)
    (alloc : Allocation p (admittedABI p args inputs) before.symbolic.extend) (work : List Program)
    (legal : (programs (grid p)).Perm work) :
    ∃ result,Physical.invoke (fun _=>Native.algebra) (fun _=>Native.dotModel)
        p (admittedABI p args inputs) before.symbolic alloc work=some result ∧ result.tensor=alloc.output ∧
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
      Access.Safe p ∧ Closed p bases ∧ Physical.InputAccess p before.symbolic := NativeReturned.returned_native p bases (admittedDomain p bases denseOK output setup) (admittedABI p args inputs) before fits alloc work legal

/-- Fourteen local ISA transport families, explicitly conditional. This does
not assert that native floating arithmetic equals exact Real attention. -/
theorem checked_numerical_transport {α β : Type} (f : α→β) (A : Native.ISA α) (B : Native.ISA β)
    (laws : Native.LocalTransport f A B)
    (p : Parameters) (bases : Bases) (denseOK : denseChecks p=Bool.true)
    (output : Live p → p.o=dense p) (setup : setupChecks p bases=Bool.true) (args : Arguments p) (inputs : inputChecks p args=Bool.true)
    (before : NativeReturned.InputWords) (fits : Physical.InputFits p before.symbolic)
    (alloc : Allocation p (admittedABI p args inputs) before.symbolic.extend) (work : List Program)
    (legal : (programs (grid p)).Perm work) :
    ∃ result,Physical.invoke (fun _=>Native.algebra) (fun _=>Native.dotModel)
        p (admittedABI p args inputs) before.symbolic alloc work=some result ∧ result.tensor=alloc.output ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<96),
        ∃ index : Fin (result.memory.capacity p.Out),index.val=logicalAddress p b h n d ∧
          f ((NativeReturned.observe result.memory p.Out index d).eval A)=
            (Native.reference p (start (grid p) (n/128,b*p.heads+h,0) alloc.state) (lane n d hd)).eval B) ∧
      Access.Safe p ∧ Closed p bases ∧ Physical.InputAccess p before.symbolic := NativeReturned.local_numerical_return f A B laws p bases (admittedDomain p bases denseOK output setup) (admittedABI p args inputs) before fits alloc work legal

#axiomsClean checked_native
#axiomsClean checked_numerical_transport
end
end VeriTile.Bench.Repairs.AttentionFwdTriton2.TensorBridge
