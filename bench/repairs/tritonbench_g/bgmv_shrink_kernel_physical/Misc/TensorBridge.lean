import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Misc.NativeCorrect
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Kernels
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.RealTensor
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.Physical
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.RealKernelResult
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.NativeKernelResult
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.IOConnection

namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.TensorBridge
open VeriTile.Triton Contract NativeGraph Native
noncomputable section

theorem source_workset (x w y i : TensorInfo) (s : Scale) (indices : Nat → Int)
    (ok : Admitted x w y i s indices) (u : BlockState) :
    FPWorkset.SourceWorkset Real.algebra Real.primitives (call x w y i s indices ok) (Real.data (call x w y i s indices ok) u indices) := by
  intro b n split active current
  have valid := (selected_valid x w y i s indices ok b.val n.val b.isLt n.isLt).resolve_left active
  exact FPWorkset.source_event Real.algebra Real.primitives _ (Real.data (call x w y i s indices ok) u indices) b n split valid.1 active current

/-- Complete shared quotation, original caller tensor and independent dense
Real GEMV. The only run premise is ordinary per-address atomic linearization. -/
theorem checked_real (x w y i : TensorInfo) (s : Scale) (indices : Nat → Int)
    (ok : Admitted x w y i s indices) (u : BlockState)
    (orders : VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset.Orders (call x w y i s indices ok))
    (run : Real.Run (call x w y i s indices ok) orders u indices) :
    let result := RealTensor.invoke x w y i s indices ok orders u run
    result.tensor=y ∧ result.returned=() ∧
    FPWorkset.SourceWorkset Real.algebra Real.primitives (call x w y i s indices ok) (Real.data (call x w y i s indices ok) u indices) ∧
    (∀ c : RealTensor.Cell (call x w y i s indices ok),
      result.cells "Y" (RealTensor.address (call x w y i s indices ok) c)=
        .mk .fp32 (Target.real (call x w y i s indices ok) u indices c.1 c.2)) :=
  ⟨rfl,rfl,source_workset x w y i s indices ok u,RealTensor.all_values x w y i s indices ok orders u run⟩

theorem checked_real_frame (x w y i : TensorInfo) (s : Scale) (indices : Nat → Int)
    (ok : Admitted x w y i s indices) (u : BlockState)
    (orders : VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset.Orders (call x w y i s indices ok))
    (run : Real.Run (call x w y i s indices ok) orders u indices) (r : RegionName) (off : Nat)
    (outside : ¬(r="Y" ∧ ∃ c : RealTensor.Cell (call x w y i s indices ok),
      indices c.1.val ≠ -1 ∧ off=RealTensor.address _ c)) :
    (RealTensor.invoke x w y i s indices ok orders u run).cells r off=
      (Real.data (call x w y i s indices ok) u indices).cells (call x w y i s indices ok) r off :=
  RealTensor.frame x w y i s indices ok orders u run r off outside

abbrev nativeReference := @Native.reference

/-- Source-derived atomic events, scalar-tree arithmetic at the actual half
format, exact runtime order, same supplied output, four concrete stored bytes,
and every untouched byte. No whole-output numerical equation is a premise. -/
theorem checked_fp (plans : ScalarTree.Plans) (S : Selection) (I : Instructions)
    (binding : CompilerBinding plans S) (x w y i : TensorInfo) (s : Scale) (indices : Nat → Int)
    (ok : Admitted x w y i s indices) (orders : VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset.Orders (call x w y i s indices ok))
    (a : Binding x w y i s) (before : Raw.State) (run : Run plans (call x w y i s indices ok) orders a before)
    (metadata : ∀ b,b < i.numel → (data a before).indices b=indices b) :
    let h:=call x w y i s indices ok
    let result:=invoke binding I x w y i s indices ok orders a before run
    result.tensor=y ∧ result.returned=() ∧ FPWorkset.SourceWorkset algebra (primitives plans) h (data a before) ∧
    (∀ c : Native.Cell h,indices c.1.val≠-1 → ∀ byte : Fin 4,
      result.bytes a.output.window.owner ((a.output.window.key (address h c)).address+byte.val)=
        Raw.unpack (nativeReference plans S I h orders a before c) byte) ∧
    (∀ owner ptr,(∀ c : Native.Cell h,indices c.1.val≠-1 → ¬(a.output.window.key (address h c)).covers owner ptr) →
      result.bytes owner ptr=before owner ptr) :=
  Native.whole plans S I binding x w y i s indices ok orders a before run metadata

theorem checked_caller_bytes (binding : CompilerBinding plans S) (I : Instructions)
    (x w y i : TensorInfo) (s : Scale) (indices : Nat → Int) (ok : Admitted x w y i s indices)
    (orders : VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset.Orders (call x w y i s indices ok)) (a : Binding x w y i s) (before : Raw.State)
    (run : Run plans (call x w y i s indices ok) orders a before) (owner ptr : Nat)
    (caller : a.input.backing.covers owner ptr ∨ a.weights.backing.covers owner ptr ∨ a.indices.backing.covers owner ptr) :
    (invoke binding I x w y i s indices ok orders a before run).bytes owner ptr=before owner ptr :=
  Native.caller_frame binding I x w y i s indices ok orders a before run owner ptr caller

#axiomsClean source_workset
#axiomsClean checked_real
#axiomsClean checked_real_frame
#axiomsClean checked_fp
#axiomsClean checked_caller_bytes
#specNonCircular Target.real avoiding [Real.Run,Surface.source,FP.Structural.exec,RealTensor.invoke]
#specNonCircular nativeReference avoiding [FPWorkset.SourceWorkset,Surface.source,FP.Structural.exec,sourceBytes]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.TensorBridge
