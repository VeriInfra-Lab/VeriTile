import bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Proofs.RawKernel
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelSchedule
namespace VeriTile.Bench.Repairs.KsoftmaxGridYChecked.RawNative
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.Ksoftmax
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
noncomputable section

theorem kernel_native_raw (rest : Algebra Graph.Expr) (platform : KernelCompiler.Platform)
    (op : Operation) (B Q K D : Nat) (hD : 0<D) (_covered : K≤D)
    (dst leftPtr rightPtr : BitVec 64) (code : RawKernel.Selected platform op Q K D dst leftPtr rightPtr)
    (s : State Graph.Expr) (left right : Nat → Graph.Word)
    (loads : RawKernel.InputWords op B Q K D s left right) :
    ∃ final,RawKernel.run (Graph.algebra rest code.lowering.profile) op B Q K D s=some final ∧
      (∀ (b : Fin B) (r : Fin Q) (i : Fin D),i.val<K →
        Graph.eval (platform.instructions code.object.primitives)
          ((final.mem "dst" (Native.flat Q K b r i.val)).read .real)=
          Graph.reference rest code.lowering.profile (platform.instructions code.object.primitives)
            op Q K b r 0 1 left right i) ∧
      (∀ (region : RegionName) j,region≠"dst" ∨ B*Q*K≤j → final.mem region j=s.mem region j) := by
  obtain ⟨final,execution,values,frame⟩ := RawKernel.complete (Graph.algebra rest code.lowering.profile) op B Q K D hD s
  refine ⟨final,execution,?_,frame⟩
  intro b r i hi
  rw [values b r i hi]
  unfold Graph.reference
  apply congrArg (fun c : Cell Graph.Expr => Graph.eval (platform.instructions code.object.primitives) (c.read .real))
  exact gridValue_congr _ _ _ _ _ _ _ _ _ _ (loads.1 b r) (loads.2 b r) i

theorem forward_kernel_native_raw (rest : Algebra Graph.Expr) (platform : KernelCompiler.Platform)
    (log causal : Bool) (mode : Forward.MaskMode) (B Q K D : Nat) (hD : 0<D) (_covered : K≤D)
    (dst leftPtr rightPtr : BitVec 64) (code : RawKernel.Selected platform (.forward log causal mode) Q K D dst leftPtr rightPtr)
    (s : State Graph.Expr) (left right : Nat → Graph.Word)
    (loads : RawKernel.InputWords (.forward log causal mode) B Q K D s left right) :
    ∃ final,RawKernel.run (Graph.algebra rest code.lowering.profile) (.forward log causal mode) B Q K D s=some final ∧
      (∀ (b : Fin B) (r : Fin Q) (i : Fin D),i.val<K →
        Graph.eval (platform.instructions code.object.primitives)
          ((final.mem "dst" (Native.flat Q K b r i.val)).read .real)=
          Graph.reference rest code.lowering.profile (platform.instructions code.object.primitives)
            (.forward log causal mode) Q K b r 0 1 left right i) ∧
      (∀ (region : RegionName) j,region≠"dst" ∨ B*Q*K≤j → final.mem region j=s.mem region j) :=
  kernel_native_raw rest platform (.forward log causal mode) B Q K D hD _covered dst leftPtr rightPtr code s left right loads

theorem backward_kernel_native_raw (rest : Algebra Graph.Expr) (platform : KernelCompiler.Platform)
    (log causal : Bool) (B Q K D : Nat) (hD : 0<D) (_covered : K≤D)
    (dst leftPtr rightPtr : BitVec 64) (code : RawKernel.Selected platform (.backward log causal) Q K D dst leftPtr rightPtr)
    (s : State Graph.Expr) (left right : Nat → Graph.Word)
    (loads : RawKernel.InputWords (.backward log causal) B Q K D s left right) :
    ∃ final,RawKernel.run (Graph.algebra rest code.lowering.profile) (.backward log causal) B Q K D s=some final ∧
      (∀ (b : Fin B) (r : Fin Q) (i : Fin D),i.val<K →
        Graph.eval (platform.instructions code.object.primitives)
          ((final.mem "dst" (Native.flat Q K b r i.val)).read .real)=
          Graph.reference rest code.lowering.profile (platform.instructions code.object.primitives)
            (.backward log causal) Q K b r 0 1 left right i) ∧
      (∀ (region : RegionName) j,region≠"dst" ∨ B*Q*K≤j → final.mem region j=s.mem region j) :=
  kernel_native_raw rest platform (.backward log causal) B Q K D hD _covered dst leftPtr rightPtr code s left right loads

theorem empty_kernel_raw {α : Type} [Inhabited α] (model : Algebra α) (op : Operation)
    (B Q K D : Nat) (s : State α) (empty : B*Q*K=0) :
    RawKernel.run model op B Q K D s=some s := by simp [RawKernel.run,empty]

theorem reversed_schedule_equivalent {α : Type} [Inhabited α] (model : Algebra α) (op : Operation)
    (B Q K D : Nat) (hD : 0<D) (covered : K≤D) (s : State α) :
    ∃ original reordered,
      KernelExecution.rows model op B Q K D (programs B Q) s=some original ∧
      Ksoftmax.KernelSchedule.candidate model op B Q K D s=some reordered ∧ reordered.mem=original.mem :=
  Ksoftmax.KernelSchedule.reversed_schedule_equivalent model op B Q K D hD covered s

#axiomsClean forward_kernel_native_raw
#axiomsClean backward_kernel_native_raw
#axiomsClean empty_kernel_raw
#axiomsClean reversed_schedule_equivalent
end
end VeriTile.Bench.Repairs.KsoftmaxGridYChecked.RawNative
