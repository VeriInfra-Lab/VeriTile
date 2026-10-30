import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.ProducerIO

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.PublicIO.Producer
open VeriTile.Triton FP.Structural
open KernelProducerGrid (Geometry)
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

private theorem fold_regs {ι : Type} (xs : List ι)
    (active : ι→Prop) [DecidablePred active] (r : RegionName)
    (off : ι→Nat) (v : ι→FP.Structural.Cell α) (s : State α) :
    (xs.foldl (fun t i=>if active i then t.write r (off i) (v i) else t) s).regs=s.regs := by
  induction xs generalizing s with
  | nil=>rfl
  | cons i xs ih=>simp only [List.foldl_cons,ih];split <;> rfl
private theorem fold_pids {ι : Type} (xs : List ι)
    (active : ι→Prop) [DecidablePred active] (r : RegionName)
    (off : ι→Nat) (v : ι→FP.Structural.Cell α) (s : State α) :
    (xs.foldl (fun t i=>if active i then t.write r (off i) (v i) else t) s).pids=s.pids := by
  induction xs generalizing s with
  | nil=>rfl
  | cons i xs ih=>simp only [List.foldl_cons,ih];split <;> rfl

/-- Both actual producer loops, masked FP32 reads, signed INT8 stores and the
unmasked FP16 row scale store are checked in their actual predecessor states. -/
theorem source_safe (M : Algebra α) (integer8 : α→BitVec 8) (p : Geometry)
    (r : Fin p.rows) (s : State α) (bounds : RegionName→Nat)
    (hx : ∀ c : Fin p.width,c.val<p.columns → r.val*p.columns+c.val<bounds "x")
    (hq : ∀ c : Fin p.width,c.val<p.columns → r.val*p.columns+c.val<bounds "q")
    (hs : r.val<bounds "scales") :
    TraceSafe (KernelProducer.backend M integer8) bounds
      (Source.quantize (KernelProducer.parameters (KernelProducerGrid.parameters p)))
      (KernelProducerGrid.atRow s r) := by
  simp (maxSteps := 1500000) [TraceSafe,Source.quantize,KernelProducer.parameters,
    KernelProducerGrid.parameters,KernelProducerGrid.atRow,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.rangeSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    run,step,range,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,numericLt,
    numericEq,natLt,bop,KernelProducer.backend,State.setReg,ComputeDType.eraseDType,
    ofFloat,toFloat,Region.cast,Option.bind,TileShape.axisDim,p.positive,fold_regs,fold_pids,
    integralFloorDiv,integralMod,Nat.mod_eq_of_lt r.isLt,hs]
  exact ⟨hx,hq⟩

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.PublicIO.Producer
