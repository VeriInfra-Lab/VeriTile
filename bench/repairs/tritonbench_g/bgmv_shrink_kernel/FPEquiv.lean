import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Kernels
import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Proofs.Direct.Physical

namespace VeriTile.Bench.Repairs.BgmvShrinkKernel.FPEquiv
open VeriTile.Triton Contract Direct.NativeGraph Direct.Native
noncomputable section

abbrev nativeReference := @Direct.Native.reference

/-- Source-derived atomic events, scalar-tree arithmetic at the actual half
format, exact runtime order, same supplied output, four concrete stored bytes,
and every untouched byte. No whole-output numerical equation is a premise. -/
theorem checked_fp (plans : Direct.ScalarTree.Plans) (S : Selection) (I : Instructions)
    (binding : CompilerBinding plans S) (x w y i : TensorInfo) (s : Scale) (indices : Nat → Int)
    (ok : Admitted x w y i s indices) (orders : Direct.Workset.Orders (call x w y i s indices ok))
    (a : Binding x w y i) (before : Raw.State) (run : Run plans (call x w y i s indices ok) orders a before)
    (metadata : ∀ b,b < i.numel → (data a before).indices b=indices b) :
    let h:=call x w y i s indices ok
    let result:=invoke binding I x w y i s indices ok orders a before run
    result.tensor=y ∧ result.returned=() ∧ Direct.FPWorkset.SourceWorkset algebra (primitives plans) h (data a before) ∧
    (∀ c : Direct.Native.Cell h,indices c.1.val≠-1 → ∀ byte : Fin 4,
      result.bytes a.output.window.owner ((a.output.window.key (address h c)).address+byte.val)=
        Raw.unpack (nativeReference plans S I h orders a before c) byte) ∧
    (∀ owner ptr,(∀ c : Direct.Native.Cell h,indices c.1.val≠-1 → ¬(a.output.window.key (address h c)).covers owner ptr) →
      result.bytes owner ptr=before owner ptr) :=
  Direct.Native.whole plans S I binding x w y i s indices ok orders a before run metadata

theorem checked_caller_bytes (binding : CompilerBinding plans S) (I : Instructions)
    (x w y i : TensorInfo) (s : Scale) (indices : Nat → Int) (ok : Admitted x w y i s indices)
    (orders : Direct.Workset.Orders (call x w y i s indices ok)) (a : Binding x w y i) (before : Raw.State)
    (run : Run plans (call x w y i s indices ok) orders a before) (owner ptr : Nat)
    (caller : a.input.backing.covers owner ptr ∨ a.weights.backing.covers owner ptr ∨ a.indices.backing.covers owner ptr) :
    (invoke binding I x w y i s indices ok orders a before run).bytes owner ptr=before owner ptr :=
  Direct.Native.caller_frame binding I x w y i s indices ok orders a before run owner ptr caller

#axiomsClean checked_fp
#axiomsClean checked_caller_bytes
#axiomsClean Direct.Workset.fiber
#axiomsClean Direct.NativeGraph.atomic_is_distinct
#axiomsClean Direct.Native.active_addresses
#axiomsClean Direct.Native.output_address
#axiomsClean Direct.Native.metadata_span
#axiomsClean Direct.Native.failure_frame
#axiomsClean Direct.Native.empty
#specNonCircular nativeReference avoiding [Direct.FPWorkset.SourceWorkset,Direct.Kernels.source,FP.Structural.exec,sourceBytes]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BgmvShrinkKernel.FPEquiv
