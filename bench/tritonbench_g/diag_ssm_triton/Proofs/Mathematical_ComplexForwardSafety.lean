import bench.tritonbench_g.diag_ssm_triton.Proofs.Mathematical_ComplexForwardLoop
import bench.tritonbench_g.diag_ssm_triton.Proofs.SignedRealLoop
import VeriTile.Triton.Memory.TraceRewrite
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexForwardSafety
open VeriTile.Triton ComplexForwardStep ComplexForwardLoop
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

/-- Every active component address must fit its allocation. The final
component bound includes the preceding real component. Static reads remain
required for a zero-length sequence; no bound on block versus width is used. -/
structure Bounds (p : Layout) (u : BlockState) (length : Nat) (bounds : RegionBounds) : Prop where
  initial : ∀ j, mask p u j=Bool.true → columns p u j*2+1 < bounds p.s
  transition : ∀ j, mask p u j=Bool.true → (columns p u j%p.dim)*2+1 < bounds p.l
  input : ∀ t, t<length → ∀ j, mask p u j=Bool.true → offset p u t j*2+1 < bounds p.x
  output : ∀ t, t<length → ∀ j, mask p u j=Bool.true → offset p u t j*2+1 < bounds p.y

theorem body_safe (bounds : RegionBounds) (x y : RegionName) (batch dim B t : Nat) (u : BlockState)
    (col : TileIndex [B] → Nat) (mask : TileIndex [B] → Bool)
    (state transition : TileIndex [B] → ℂ)
    (ht : u.regs .int [] "t" = some (Tile.scalar (t : Int)))
    (hc : u.regs .int [B] "col_offsets" = some ⟨fun j=>(col j : Int)⟩)
    (hm : u.regs .bool [B] "mask" = some ⟨mask⟩)
    (hsr : u.regs .real [B] "s_real" = some (realTile state))
    (hsi : u.regs .real [B] "s_imag" = some (imagTile state))
    (hlr : u.regs .real [B] "lambda_real" = some (realTile transition))
    (hli : u.regs .real [B] "lambda_imag" = some (imagTile transition))
    (hx : ∀ j, mask j=Bool.true → ComplexForwardStep.offset batch dim t col j*2+1 < bounds x)
    (hy : ∀ j, mask j=Bool.true → ComplexForwardStep.offset batch dim t col j*2+1 < bounds y) :
    Stmt.TraceSafeList bounds (ComplexForwardSource.body x y batch dim B) u := by
  simp only [ComplexForwardSource.body,Erasure.complexForward,TypedSource.complexForward,VeriTile.Bench.Repairs.DiagSsmTriton.Kernels.complexForwardTyped,VeriTile.Bench.Repairs.DiagSsmTriton.complexForwardKernel,VeriTile.Bench.Repairs.DiagSsmTriton.Kernels.complexForwardIndexCommuted,ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeStmt.listToAlgorithm?_map_alg,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,Kernel.eraseDType,Stmt.eraseDTypeList.eq_def,Stmt.eraseDType.eq_def,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Except.bind,bind,
    ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeKernel.surfaceBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeStmt.listToAlgorithm?_map_alg,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Except.bind,bind,Bool.false_eq_true,↓reduceIte,List.drop]
  simp only [Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt.eq_def,
    MemAccess.SafeAt,MaskOpt.SafeAt,MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
    and_true,true_and,and_self,stepStmts,stepStmt,evalOp,evalOp.eq_def,Option.bind,bind,pure,
    Option.map,Option.pure_def,Broadcast.leftIndex,Broadcast.rightIndex,Tile.vec,
    String.reduceEq,Tile.bop,Tile.cop,Tile.scalar,NumericDType.mul,NumericDType.add,NumericDType.sub,
    WithBot.realMul,WithBot.realAdd,WithBot.realSub,Option.map₂_some_some,
    Region.cast,BlockState.writeMemTyped_real,BlockState.foldl_writeMem_masked_regs,
    BlockState.setReg_same,BlockState.setReg_ne_name,BlockState.setReg_pids,
    BlockState.setReg_readMemValue,BlockState.setReg_readMem,BlockState.readMemValue_real,
    Option.bind_some,Option.map_some,Bool.true_eq_false,↓reduceIte,ne_eq,not_false_eq_true,
    ht,hc,hm,hsr,hsi,hlr,hli,realTile,imagTile,Nat.mul_assoc,
    FloatDType.storeValue,FloatDType.toWithBot,WithBot.unbotD_some]
  have twice (n : Nat) : (n : Int)*2=((n*2 : Nat) : Int) := by omega
  have plusOne (n : Nat) : (n : Int)+1=((n+1 : Nat) : Int) := by omega
  simp only [Int.ofNat_eq_natCast,←Int.natCast_mul,←Int.natCast_add,twice,plusOne,
    Int.toNat_natCast,Nat.mul_assoc,IntegerExpr.same_left,IntegerExpr.same_right]
  simp only [Option.some.injEq]
  refine ⟨?_,?_,?_,?_⟩
  · intro offsets hoff i ⟨masks,hmask,hact⟩
    subst offsets; subst masks
    exact Nat.lt_of_le_of_lt (Nat.le_add_right _ 1) (hx i hact)
  · intro offsets hoff i ⟨masks,hmask,hact⟩
    subst offsets; subst masks
    exact hx i hact
  · intro offsets hoff i ⟨masks,hmask,hact⟩
    subst offsets; subst masks
    exact Nat.lt_of_le_of_lt (Nat.le_add_right _ 1) (hy i hact)
  · intro offsets hoff i ⟨masks,hmask,hact⟩
    subst offsets; subst masks
    exact hy i hact


theorem prelude_safe (p : Layout) (u : BlockState) (length : Nat) (bounds : RegionBounds)
    (hb : Bounds p u length bounds) :
    Stmt.TraceSafeList bounds (ComplexForwardSource.prelude p.s p.l length p.batch p.dim p.B) u := by
  let s:=p.s
  let l:=p.l
  let batch:=p.batch
  let dim:=p.dim
  let B:=p.B
  simp only [ComplexForwardSource.prelude,Erasure.complexForward,TypedSource.complexForward,VeriTile.Bench.Repairs.DiagSsmTriton.Kernels.complexForwardTyped,VeriTile.Bench.Repairs.DiagSsmTriton.complexForwardKernel,VeriTile.Bench.Repairs.DiagSsmTriton.Kernels.complexForwardIndexCommuted,ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeStmt.listToAlgorithm?_map_alg,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,Kernel.eraseDType,Stmt.eraseDTypeList.eq_def,Stmt.eraseDType.eq_def,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Except.bind,bind,
    ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeKernel.surfaceBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeStmt.listToAlgorithm?_map_alg,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Except.bind,bind,Bool.false_eq_true,↓reduceIte,List.take]
  simp only [Kernel.TraceSafe,Stmt.TraceSafeList,Stmt.TraceSafe,Stmt.forLoopTraceSafe,
    Op.SafeAt.eq_def,MemAccess.SafeAt,MaskOpt.SafeAt,MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe,MaskOpt.Active,
    stepStmts,stepStmt,↓eval_truncMod,evalOp,evalOp.eq_def,Option.bind,bind,pure,
    Option.map,Option.pure_def,Broadcast.leftIndex,Broadcast.rightIndex,Tile.vec,
    String.reduceEq,FlatAlloc.shiftBroadcast,
    Tile.bop,Tile.cop,Tile.scalar,NumericDType.mul,NumericDType.add,
    IntegralDType.mod,ComparableDType.lt,IntegerExpr.same_left,IntegerExpr.same_right,stepForLoopAux.forLoop_unfold,
    stepForLoopAux.step_ge,Region.cast,BlockState.setReg_same,
    BlockState.setReg_ne_name,BlockState.setReg_pids,BlockState.setReg_readMem,
    BlockState.setReg_readMemValue,Option.bind_some,Option.map_some,
    Bool.true_eq_false,↓reduceIte,ne_eq,not_false_eq_true,Nat.le_refl,Nat.not_lt_zero,False.elim,forall_const,true_and,and_true]
  have twice (n : Nat) : (n : Int)*2=((n*2 : Nat) : Int) := by omega
  have plusOne (n : Nat) : (n : Int)+1=((n+1 : Nat) : Int) := by omega
  simp only [Int.ofNat_eq_natCast,←Int.natCast_mul,←Int.natCast_add,←Int.ofNat_tmod,
    twice,plusOne,Int.toNat_natCast,Int.ofNat_lt]
  simp only [Option.some.injEq,forall_eq_apply_imp_iff,exists_eq_left,decide_eq_true_eq]
  refine ⟨?_,?_,?_,?_⟩
  all_goals intro offsets hoff i ⟨masks,hm,hact⟩; subst offsets; subst masks
  · exact Nat.lt_of_le_of_lt (Nat.le_add_right _ 1) (hb.initial i hact)
  · exact hb.initial i hact
  · exact Nat.lt_of_le_of_lt (Nat.le_add_right _ 1) (hb.transition i hact)
  · exact hb.transition i hact

theorem trace_safe (p : Layout) (u : BlockState) (length : Nat) (bounds : RegionBounds)
    (hb : Bounds p u length bounds) :
    (Erasure.complexForward Bool.false p.s p.x p.y p.l length p.batch p.dim p.B).toAlgKernel.TraceSafe bounds u := by
  unfold Kernel.TraceSafe
  rw [ComplexForwardSource.source_body]
  apply TraceProofs.append_safe _ u (prelude_safe p u length bounds hb)
  intro v he
  rw [ComplexForwardInit.prefix_step] at he
  obtain rfl := Option.some.inj he
  refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => Stmt.TraceSafeList.nil_intro)
  simp only [Stmt.TraceSafe,evalOp.eq_def,Tile.scalar,Option.bind_some,pure,bind,Int.ofNat_eq_natCast]
  let P := fun v => ∃ state : Lane p → ℂ, Registers v (columns p u) (mask p u) state (transition p u)
  have hs := VeriTile.Bench.Repairs.DiagSsmTriton.SignedRealLoop.safe bounds "t" length
    (ComplexForwardSource.body p.x p.y p.batch p.dim p.B) P ?_ ?_
    0 (Nat.zero_le _) _ ⟨initial p u,ComplexForwardInit.result_registers u p.s p.l length p.batch p.dim p.B⟩
  · simpa only [Nat.cast_zero,Op.SafeAt.eq_def,true_and] using hs
  · intro i v hi hv
    obtain ⟨state,hr⟩ := hv
    have ht := hr.setCounter i
    exact body_safe bounds p.x p.y p.batch p.dim p.B i (v.setReg "t" .int [] (Tile.scalar (i : Int)))
      (columns p u) (mask p u) state (transition p u) (by simp)
      ht.columns ht.active ht.realState ht.imagState ht.realTransition ht.imagTransition
      (hb.input i hi) (hb.output i hi)
  · intro i v w _ hv he
    obtain ⟨state,hr⟩ := hv
    have ht := hr.setCounter i
    have hrun := body_step p.x p.y p.batch p.dim p.B i (v.setReg "t" .int [] (Tile.scalar (i : Int)))
      (columns p u) (mask p u) state (transition p u) (by simp)
      ht.columns ht.active ht.realState ht.imagState ht.realTransition ht.imagTransition
    rw [hrun] at he
    obtain rfl := Option.some.inj he
    exact ⟨_,ComplexForwardStep.result_registers _ _ _ _ _ _ _ _ ht⟩

theorem flatten_ok (p : Layout) (length : Nat) :
    (Erasure.complexForward Bool.false p.s p.x p.y p.l length p.batch p.dim p.B).toAlgKernel.FlattenOk := by
  simp only [Kernel.FlattenOk,Erasure.complexForward,TypedSource.complexForward,VeriTile.Bench.Repairs.DiagSsmTriton.Kernels.complexForwardTyped,VeriTile.Bench.Repairs.DiagSsmTriton.complexForwardKernel,VeriTile.Bench.Repairs.DiagSsmTriton.Kernels.complexForwardIndexCommuted,ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeStmt.listToAlgorithm?_map_alg,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,Kernel.eraseDType,Stmt.eraseDTypeList.eq_def,Stmt.eraseDType.eq_def,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Except.bind,bind,
    ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeKernel.surfaceBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeStmt.listToAlgorithm?_map_alg,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Except.bind,bind,Bool.false_eq_true,↓reduceIte]
  simp [StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexForwardSafety
