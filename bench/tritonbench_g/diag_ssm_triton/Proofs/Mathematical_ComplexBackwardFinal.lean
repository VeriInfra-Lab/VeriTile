import bench.tritonbench_g.diag_ssm_triton.Proofs.Mathematical_ComplexBackwardStep
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexBackwardFinal
open VeriTile.Triton ComplexForwardStep
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def result {B : Nat} (u : BlockState) (gs gl : RegionName) (col : TileIndex [B] → Nat)
    (mask : TileIndex [B] → Bool) (carry acc : TileIndex [B] → ℂ) : BlockState :=
  ComplexMemory.store gl col (fun j => starRingEnd ℂ (acc j)) mask
    (ComplexMemory.store gs col (fun j => starRingEnd ℂ (carry j)) mask u)

theorem post_step {B : Nat} (u : BlockState) (gs gl : RegionName) (col : TileIndex [B] → Nat)
    (mask : TileIndex [B] → Bool) (carry acc transition : TileIndex [B] → ℂ)
    (h : ComplexBackwardStep.Registers u col mask carry acc transition) :
    stepStmts (ComplexBackwardSource.postlude gs gl B) u = some (result u gs gl col mask carry acc) := by
  rcases h with ⟨hc,hm,hcr,hci,har,hai,hlr,hli⟩
  simp only [ComplexBackwardSource.postlude,Erasure.complexBackward,TypedSource.complexBackward,VeriTile.Bench.Repairs.DiagSsmTriton.Kernels.complexBackwardTyped,VeriTile.Bench.Repairs.DiagSsmTriton.Kernels.complexBackwardIndexCommuted,ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Except.bind,bind,
    ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeKernel.surfaceBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeKernel.eraseDType,ComputeKernel.fromKernelBody,ComputeDType.eraseDType,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,
    ComparableDType.eraseDType,IntegralDType.eraseDType,
    List.map_cons,List.map_nil,Op.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType,
    Except.bind,bind,Bool.false_eq_true,↓reduceIte,List.drop]
  simp only [eq_mp_eq_cast, eq_mpr_eq_cast, cast_eq, id_eq]
  simp only [stepStmts,stepStmt,evalOp,evalOp.eq_def,Option.bind,bind,pure,
    FloatDType.toTileDType,FloatDType.ofWithBot,FloatDType.toWithBot,WithBot.map,Option.map,Option.pure_def,Broadcast.leftIndex,Broadcast.rightIndex,Tile.vec,
    String.reduceEq,Tile.bop,Tile.cop,Tile.scalar,NumericDType.mul,NumericDType.add,NumericDType.sub,
    WithBot.realSub,Option.map₂_some_some,Region.cast,BlockState.writeMemTyped_real,
    BlockState.foldl_writeMem_masked_regs,BlockState.setReg_same,BlockState.setReg_ne_name,
    hc,hm,hcr,hci,har,hai,realTile,imagTile,
    FloatDType.storeValue,FloatDType.toWithBot,WithBot.unbotD_some]
  simp only [result,ComplexMemory.store,Complex.conj_re,Complex.conj_im]

theorem read_carry {B : Nat} (u : BlockState) (gs gl : RegionName) (col : TileIndex [B] → Nat)
    (mask : TileIndex [B] → Bool) (carry acc : TileIndex [B] → ℂ)
    (hne : gs≠gl) (hi : Function.Injective col) (j : TileIndex [B]) (hm : mask j=Bool.true) :
    ComplexMemory.read (result u gs gl col mask carry acc) gs (col j) = starRingEnd ℂ (carry j) := by
  unfold result
  have h0 := ComplexMemory.store_frame gl col (fun j => starRingEnd ℂ (acc j)) mask
    (ComplexMemory.store gs col (fun j => starRingEnd ℂ (carry j)) mask u) gs (col j*2) (Or.inl hne)
  have h1 := ComplexMemory.store_frame gl col (fun j => starRingEnd ℂ (acc j)) mask
    (ComplexMemory.store gs col (fun j => starRingEnd ℂ (carry j)) mask u) gs (col j*2+1) (Or.inl hne)
  rw [ComplexMemory.read_of_mem h0 h1]
  exact ComplexMemory.store_read _ _ _ _ _ hi j hm

theorem read_acc {B : Nat} (u : BlockState) (gs gl : RegionName) (col : TileIndex [B] → Nat)
    (mask : TileIndex [B] → Bool) (carry acc : TileIndex [B] → ℂ)
    (hi : Function.Injective col) (j : TileIndex [B]) (hm : mask j=Bool.true) :
    ComplexMemory.read (result u gs gl col mask carry acc) gl (col j) = starRingEnd ℂ (acc j) := by
  unfold result
  exact ComplexMemory.store_read _ _ _ _ _ hi j hm

theorem frame {B : Nat} (u : BlockState) (gs gl : RegionName) (col : TileIndex [B] → Nat)
    (mask : TileIndex [B] → Bool) (carry acc : TileIndex [B] → ℂ) (r : RegionName) (o : Nat)
    (hs : r≠gs ∨ ∀ j, mask j=Bool.true → col j*2≠o ∧ col j*2+1≠o)
    (hl : r≠gl ∨ ∀ j, mask j=Bool.true → col j*2≠o ∧ col j*2+1≠o) :
    (result u gs gl col mask carry acc).mem r o=u.mem r o := by
  unfold result
  rw [ComplexMemory.store_frame _ _ _ _ _ r o hl,ComplexMemory.store_frame _ _ _ _ _ r o hs]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexBackwardFinal
