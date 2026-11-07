import bench.tritonbench_g.int8_matmul_quantization.Proofs.ProducerSafety
namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelProducer.SourceSafety
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 1000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

private theorem append_safe (M : Algebra α) (bounds : RegionName→Nat)
    (xs ys : List ComputeStmt) (s t : State α)
    (ran : run M xs s=some t) (head : Safety.traceSafeList M bounds xs s)
    (tail : Safety.traceSafeList M bounds ys t) : Safety.traceSafeList M bounds (xs++ys) s := by
  rw [Safety.traceSafeList_append]
  refine ⟨head,?_⟩
  intro next h
  rw [ran] at h
  cases Option.some.inj h
  exact tail

private theorem singleton_safe (M : Algebra α) (bounds : RegionName→Nat)
    (st : ComputeStmt) (s : State α) (safe : Safety.stepSafe M bounds st s) :
    Safety.traceSafeList M bounds [st] s :=
  (Safety.traceSafeList_cons M bounds st [] s).mpr ⟨safe,fun t _=>Safety.traceSafeList_nil M bounds t⟩

private theorem limit_safe (M : Algebra α) (bounds : RegionName→Nat)
    (p : Parameters) (state : State α) : Safety.safeOp M none bounds state (limit p) := by
  simp only [limit,maxLoop,code,KernelSource.quantize,quantizeKernel,parameters,
    ComputeKernel.surfaceBody,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some]
  simp [Safety.safeOp_unfold]

private theorem loop_safe (M : Algebra α) (bounds : RegionName→Nat)
    (p : Parameters) (body : List ComputeStmt) (s : State α)
    (checked : Safety.rangeIntSafe M bounds "k" 0 (tripCount p : Int) 1 body s) :
    Safety.stepSafe M bounds (.forRangeInt "k" (.castNatToInt (.constNat 0))
      (limit p) (.constNat 1) body) s := by
  rw [Safety.stepSafe_unfold]
  have evalZero : evalOp M none (Op.castNatToInt (Op.constNat 0)) s=some (fun _=>(0 : Int)) := rfl
  have evalOne : evalOp M none (Op.constNat 1) s=some (fun _=>(1 : Nat)) := rfl
  refine ⟨?_,limit_safe M bounds p s,?_,?_⟩
  · simp [Safety.safeOp_unfold]
  · simp [Safety.safeOp_unfold]
  · rw [evalZero,limit_eval,evalOne]
    exact checked

/-- Every phase is checked at its actual predecessor, including all iterations
of both original source loops. Numeric correctness is not a safety premise. -/
theorem source_safe (M : Algebra α) (integer8 : α→BitVec 8)
    (p : Parameters) (s : State α) (width : 0<p.width)
    (bounds : RegionName→Nat) (bounded : Bounds p s bounds) :
    TraceSafe (backend M integer8) bounds (KernelSource.quantize (parameters p)) s := by
  let start:=prefixExecution M integer8 p s
  have startInv:=MaxPass.initial M integer8 p s
  obtain ⟨maxed,maxRan,maxInv⟩:=MaxPass.loop M integer8 p s start.val 0 (tripCount p)
    (by omega) startInv width
  let reset:=resetExecution M integer8 p s maxed maxInv
  have resetInv:=resetInvariant M integer8 p s maxed maxInv
  obtain ⟨quantized,quantRan,quantInv⟩:=QuantizePass.loop M integer8 p s reset.val
    (finalScale M p s) 0 (tripCount p) (by omega) resetInv
  change Safety.traceSafeList (backend M integer8) bounds (code p) s
  rw [code_decomposition]
  simp only [List.append_assoc]
  apply append_safe _ _ _ _ s start.val start.property (prefix_safe M integer8 p s bounds)
  apply append_safe _ _ [maxLoop p] _ start.val maxed
    (by rw [maxLoop_run];exact maxRan)
  · apply singleton_safe
    rw [maxLoop_shape]
    exact loop_safe _ bounds p _ start.val (max_loop_safe M integer8 p s start.val 0
      (tripCount p) startInv width bounds bounded)
  apply append_safe _ _ _ _ maxed reset.val reset.property (reset_safe M integer8 p maxed bounds)
  apply append_safe _ _ [quantizeLoop p] _ reset.val quantized
    (by rw [quantizeLoop_run];exact quantRan)
  · apply singleton_safe
    rw [quantizeLoop_shape]
    exact loop_safe _ bounds p _ reset.val (quantize_loop_safe M integer8 p s reset.val
      (finalScale M p s) 0 (tripCount p) resetInv bounds bounded)
  exact tail_safe M integer8 p s quantized (finalScale M p s) (tripCount p) quantInv bounds bounded

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelProducer.SourceSafety
