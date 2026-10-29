import bench.repairs.tritonbench_g.cross_entropy1.Proofs.NativeByteTrace

namespace VeriTile.Bench.Repairs.CrossEntropy1.Physical
open VeriTile.Triton Source Host Floating
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

/-- A source-indexed code certificate and the actual final scalar store.
After the store, preservation is derived from a trace of byte footprints,
not supplied as equality to a requested loss or gradient. -/
structure SelectedReturn {σ : Type} {selection : Selection σ}
    (layout : Layout σ) (observe : σ → Nat → Nat → Byte)
    (events : ByteEvents σ) (trace : PrimitiveTrace σ)
    (readWord : σ → Tensor → Nat → Floating.Word)
    (expression : Expr) (state : σ) (tensor : Tensor) (index : Nat) where
  plan : SelectedPlan selection expression
  code : CompiledExecution plan trace readWord
  site : Occurrence σ
  stored : σ
  store : trace.store site code.output stored tensor index
  stable : layout.buffer state tensor.region=layout.buffer stored tensor.region
  writes : List ByteAddress
  tail : ByteTrace events stored state writes
  untouched : ∀ byte : Fin 4,
    ((layout.buffer stored tensor.region).device,
      (layout.buffer stored tensor.region).base+4*tensor.index index+byte.val)∉writes

theorem SelectedReturn.bytes {σ : Type} {selection : Selection σ}
    {layout : Layout σ} {observe events trace readWord expression state tensor index}
    (api : PrimitiveContracts trace native readWord (tensorByte layout observe))
    (memory : ByteContracts events observe selection)
    (ports : Obligations native reference) (laws : Peepholes native) (constants : ConstantPorts native)
    (result : SelectedReturn (selection:=selection) layout observe events trace readWord expression state tensor index)
    (byte : Fin 4) : tensorByte layout observe state tensor index byte=
      byteOf (if tensor.isNeg then flipSign (result.plan.reference reference) else result.plan.reference reference) byte := by
  have stored := api.store result.site result.code.output result.stored tensor index result.store byte
  have tail := result.tail.frame memory _ _ (result.untouched byte)
  rw [result.code.reference_value api ports laws constants] at stored
  unfold tensorByte at stored ⊢
  rw [result.stable]
  exact tail.trans stored

theorem SelectedReturn.object {σ : Type} {selection : Selection σ}
    {layout : Layout σ} {observe events trace readWord expression state tensor index allowed loaded}
    (calls : PlatformCalls selection allowed loaded) (links : TraceCalls selection trace)
    (result : SelectedReturn (selection:=selection) layout observe events trace readWord expression state tensor index) :
    Nonempty (ActualObject allowed loaded result.site.call) ∧ tensor∈result.site.call.outputs :=
  actual_store_object calls links result.store

#axiomsClean SelectedReturn.bytes
#axiomsClean SelectedReturn.object
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Physical
