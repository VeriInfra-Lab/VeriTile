import bench.tritonbench_g.diag_ssm_triton.Proofs.Mathematical_ComplexBackwardStep
import bench.tritonbench_g.diag_ssm_triton.Proofs.Mathematical_ComplexForwardInit
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexBackwardInit
open VeriTile.Triton ComplexForwardStep
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def result (u : BlockState) (l : RegionName) (length batch dim B : Nat) : BlockState :=
  let shaped := ((u.setReg "length" .int [] (Tile.scalar (length : Int))).setReg
    "batch_size" .int [] (Tile.scalar (batch : Int))).setReg "dim" .int [] (Tile.scalar (dim : Int))
  let indexed := shaped.setReg "col_idx" .int [] (Tile.scalar ((u.pids 0*B : Nat) : Int))
  let columnsSet := indexed.setReg "col_offsets" .int [B] ⟨fun j=>(ComplexForwardInit.columns u B j : Int)⟩
  let masked := columnsSet.setReg "mask" .bool [B] ⟨ComplexForwardInit.active u batch dim B⟩
  let lr := masked.setReg "lambda_real" .real [B] (realTile (ComplexForwardInit.transition u l batch dim B))
  let li := lr.setReg "lambda_imag" .real [B] (imagTile (ComplexForwardInit.transition u l batch dim B))
  let cr := li.setReg "grad_s_real" .real [B] (realTile (fun _ => 0))
  let ci := cr.setReg "grad_s_imag" .real [B] (imagTile (fun _ => 0))
  let ar := ci.setReg "grad_lambda_real" .real [B] (realTile (fun _ => 0))
  ar.setReg "grad_lambda_imag" .real [B] (imagTile (fun _ => 0))

theorem prefix_step (u : BlockState) (l : RegionName) (length batch dim B : Nat) :
    stepStmts (ComplexBackwardSource.prelude l length batch dim B) u = some (result u l length batch dim B) := by
  simp only [ComplexBackwardSource.prelude,Erasure.complexBackward,TypedSource.complexBackward,VeriTile.Bench.Repairs.DiagSsmTriton.Kernels.complexBackwardTyped,VeriTile.Bench.Repairs.DiagSsmTriton.complexBackwardKernel,VeriTile.Bench.Repairs.DiagSsmTriton.Kernels.complexBackwardIndexCommuted,ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,Kernel.eraseDType,Stmt.eraseDTypeList.eq_def,Stmt.eraseDType.eq_def,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Except.bind,bind,
    ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeKernel.surfaceBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeKernel.eraseDType,ComputeKernel.fromKernelBody,ComputeDType.eraseDType,
    Kernel.eraseDType,Stmt.eraseDTypeList.eq_def,Stmt.eraseDType.eq_def,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,
    ComparableDType.eraseDType,IntegralDType.eraseDType,
    List.map_cons,List.map_nil,Op.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType,
    Except.bind,bind,Bool.false_eq_true,↓reduceIte,List.take]
  simp only [stepStmts,stepStmt,↓eval_truncMod,evalOp,evalOp.eq_def,Option.bind,bind,pure,
    Option.map,Option.pure_def,Broadcast.leftIndex,Broadcast.rightIndex,Tile.vec,
    String.reduceEq,Tile.bop,Tile.cop,Tile.scalar,NumericDType.mul,NumericDType.add,
    ComparableDType.lt,IntegralDType.mod,IntegerExpr.same_left,IntegerExpr.same_right,Region.cast,
    BlockState.setReg_same,BlockState.setReg_ne_name,BlockState.setReg_pids,
    BlockState.setReg_readMemValue,BlockState.setReg_readMem,BlockState.readMemValue_real,
    Option.bind_some,Option.map_some,Bool.true_eq_false,↓reduceIte,ne_eq,not_false_eq_true,
    FloatDType.storeValue,FloatDType.toWithBot,WithBot.unbotD_some]
  simp only [result,ComplexForwardInit.transition,ComplexForwardInit.active,
    ComplexForwardInit.columns,input,realTile,imagTile]
  have twice (n : Nat) : (n : Int)*2=((n*2 : Nat) : Int) := by omega
  have plusOne (n : Nat) : (n : Int)+1=((n+1 : Nat) : Int) := by omega
  simp only [apply_ite,Int.ofNat_eq_natCast,←Int.natCast_mul,←Int.natCast_add,←Int.ofNat_tmod,
    twice,plusOne,Int.toNat_natCast,Int.ofNat_lt]
  norm_num only [Complex.zero_re,Complex.zero_im]
  all_goals rfl

theorem result_registers (u : BlockState) (l : RegionName) (length batch dim B : Nat) :
    ComplexBackwardStep.Registers (result u l length batch dim B) (ComplexForwardInit.columns u B)
      (ComplexForwardInit.active u batch dim B) (fun _ => 0) (fun _ => 0)
        (ComplexForwardInit.transition u l batch dim B) := by
  constructor <;> simp [result]

@[simp] theorem result_memory (u : BlockState) (l : RegionName) (length batch dim B : Nat) :
    (result u l length batch dim B).mem = u.mem := rfl
@[simp] theorem result_pids (u : BlockState) (l : RegionName) (length batch dim B : Nat) :
    (result u l length batch dim B).pids = u.pids := rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexBackwardInit
