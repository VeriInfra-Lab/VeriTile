import bench.tritonbench_g.int8_matmul_quantization.Proofs.ProducerLoopSafety
import bench.tritonbench_g.int8_matmul_quantization.Proofs.ProducerPassConnection
import VeriTile.Triton.Float.StructuralSafety
namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelProducer.SourceSafety
open VeriTile.Triton FP.Structural ControlArithmetic
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false
attribute [local simp 2000] FP.Structural.eval_truncDiv FP.Structural.eval_truncMod

theorem pure_safe {α : Type} [Inhabited α] (M : Algebra α) (bounds : RegionName→Nat)
    (code : List ComputeStmt) (pure : ∀ st∈code,∀ state : State α,Safety.stepSafe M bounds st state)
    (state : State α) : Safety.traceSafeList M bounds code state := by
  induction code generalizing state with
  | nil => exact Safety.traceSafeList_nil M bounds state
  | cons statement rest ih =>
    exact (Safety.traceSafeList_cons M bounds statement rest state).mpr
      ⟨pure statement (by simp) state,fun next _=>ih (fun st member=>pure st (by simp [member])) next⟩

structure Bounds (p : Parameters) (original : State α) (bounds : RegionName→Nat) : Prop where
  input : ∀ k (i : MaxPass.Index p), i.2.1.val<p.columns-k*p.width →
    MaxPass.inputOffset p original k i<bounds p.input
  output : ∀ k (i : QuantizePass.Index p), i.2.1.val<p.columns-k*p.width →
    QuantizePass.outputOffset p original k i<bounds p.output
  scale : original.pids 0*p.scaleStride<bounds p.scales

theorem max_body_safe {α : Type} [Inhabited α] (M : Algebra α) (integer8 : α→BitVec 8)
    (p : Parameters) (original current : State α) (k : Nat)
    (h : MaxPass.Invariant M p original k current) (bounds : RegionName→Nat)
    (bounded : Bounds p original bounds) :
    Safety.traceSafeList (backend M integer8) bounds (maxBody p)
      (current.setReg "k" .int [] (fun _=>(k : Int))) := by
  by_cases width : 0<p.width
  all_goals simp only [maxBody,code,KernelSource.quantize,quantizeKernel,parameters,
    ComputeKernel.surfaceBody,List.getElem?_cons_succ,List.getElem?_cons_zero]
  all_goals simp [Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,numeric,numericLt,natLt,
    bop,backend,ComputeDType.eraseDType,Option.bind,NumericDType.sub,
    h.pointers,h.offsk,h.maximum,TileShape.axisDim,width,ofFloat,toFloat]
  all_goals exact fun column rest active => bounded.input k (0,column,rest) active

theorem quantize_body_safe {α : Type} [Inhabited α] (M : Algebra α) (integer8 : α→BitVec 8)
    (p : Parameters) (original current : State α) (scale : α) (k : Nat)
    (h : QuantizePass.Invariant p original scale k current) (bounds : RegionName→Nat)
    (bounded : Bounds p original bounds) :
    Safety.traceSafeList (backend M integer8) bounds (quantizeBody p)
      (current.setReg "k" .int [] (fun _=>(k : Int))) := by
  simp only [quantizeBody,code,KernelSource.quantize,quantizeKernel,parameters,
    ComputeKernel.surfaceBody,List.getElem?_cons_succ,List.getElem?_cons_zero]
  simp [Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,numericLt,natLt,
    bop,backend,ComputeDType.eraseDType,Option.bind,NumericDType.sub,fold_regs,fold_pids,
    h.input,h.output,h.offsk,h.scale]
  refine ⟨fun column rest active => bounded.input k (0,column,rest) active,?_⟩
  repeat' constructor
  all_goals first
    | exact fun column rest active => bounded.output k (0,column,rest) active
    | (split <;> trivial)

theorem prefix_safe {α : Type} [Inhabited α] (M : Algebra α) (integer8 : α→BitVec 8)
    (p : Parameters) (state : State α) (bounds : RegionName→Nat) :
    Safety.traceSafeList (backend M integer8) bounds (prelude p) state := by
  apply pure_safe
  simp only [prelude,code,KernelSource.quantize,quantizeKernel,parameters,
    ComputeKernel.surfaceBody,List.take_succ_cons,List.take_zero]
  intro st member state
  simp only [List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold]

theorem reset_safe {α : Type} [Inhabited α] (M : Algebra α) (integer8 : α→BitVec 8)
    (p : Parameters) (state : State α) (bounds : RegionName→Nat) :
    Safety.traceSafeList (backend M integer8) bounds (scaleAndReset p) state := by
  apply pure_safe
  simp only [scaleAndReset,code,KernelSource.quantize,quantizeKernel,parameters,
    ComputeKernel.surfaceBody,List.drop_succ_cons,List.drop_zero,List.take_succ_cons,List.take_zero]
  intro st member state
  simp only [List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl | rfl
  all_goals simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold]

theorem max_loop_safe {α : Type} [Inhabited α] (M : Algebra α) (integer8 : α→BitVec 8)
    (p : Parameters) (original current : State α) (start stop : Nat)
    (h : MaxPass.Invariant M p original start current) (width : 0<p.width)
    (bounds : RegionName→Nat) (bounded : Bounds p original bounds) :
    Safety.rangeIntSafe (backend M integer8) bounds "k" (start : Int) (stop : Int) 1 (maxBody p) current := by
  refine loop_from_invariant (backend M integer8) bounds (maxBody p) (MaxPass.Invariant M p original)
    (fun k state inv=>max_body_safe M integer8 p original state k inv bounds bounded) ?_ start stop current h
  intro k state inv
  let next:=MaxPass.iteration M integer8 p original state k inv width
  exact ⟨next.val,next.property,MaxPass.iteration_invariant M integer8 p original state k inv width⟩

theorem quantize_loop_safe {α : Type} [Inhabited α] (M : Algebra α) (integer8 : α→BitVec 8)
    (p : Parameters) (original current : State α) (scale : α) (start stop : Nat)
    (h : QuantizePass.Invariant p original scale start current)
    (bounds : RegionName→Nat) (bounded : Bounds p original bounds) :
    Safety.rangeIntSafe (backend M integer8) bounds "k" (start : Int) (stop : Int) 1 (quantizeBody p) current := by
  refine loop_from_invariant (backend M integer8) bounds (quantizeBody p) (QuantizePass.Invariant p original scale)
    (fun k state inv=>quantize_body_safe M integer8 p original state scale k inv bounds bounded) ?_ start stop current h
  intro k state inv
  let next:=QuantizePass.iteration M integer8 p original state scale k inv
  exact ⟨next.val,next.property,QuantizePass.iteration_invariant M integer8 p original state scale k inv⟩

theorem tail_safe {α : Type} [Inhabited α] (M : Algebra α) (integer8 : α→BitVec 8)
    (p : Parameters) (original current : State α) (scale : α) (iterations : Nat)
    (h : QuantizePass.Invariant p original scale iterations current) (bounds : RegionName→Nat)
    (bounded : Bounds p original bounds) :
    Safety.traceSafeList (backend M integer8) bounds (epilogue p) current := by
  have fin_one (i : Fin 1) : i=0 := Subsingleton.elim _ _
  simp only [epilogue,code,KernelSource.quantize,quantizeKernel,parameters,
    ComputeKernel.surfaceBody,List.drop_succ_cons,List.drop_zero]
  simp [Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,numericLt,natLt,
    bop,backend,ComputeDType.eraseDType,Option.bind,NumericDType.sub,h.pidm,h.scale,
    ofFloat,toFloat,fin_one]
  simp only [← Int.natCast_mul,← Int.natCast_add,Int.toNat_natCast,Int.mul_one,Int.add_zero]
  exact bounded.scale

#axiomsClean max_body_safe
#axiomsClean quantize_body_safe
#axiomsClean prefix_safe
#axiomsClean reset_safe
#axiomsClean max_loop_safe
#axiomsClean quantize_loop_safe
#axiomsClean tail_safe
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelProducer.SourceSafety
