import bench.tritonbench_g.diag_ssm_triton.Proofs.Mathematical_ComplexBackwardStep
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexBackwardLoads
open VeriTile.Triton ComplexForwardStep
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def source (gy : RegionName) (length batch dim B : Nat) : List Stmt :=
  (ComplexBackwardSource.body "s" "y" "gx" gy length batch dim B).take 5

def result {B : Nat} (u : BlockState) (gy : RegionName) (batch dim t : Nat)
    (col : TileIndex [B] → Nat) (mask : TileIndex [B] → Bool) : BlockState :=
  let off := offset batch dim t col
  let values := fun j => starRingEnd ℂ (input u gy off mask j)
  let timed := u.setReg "t" .int [] (Tile.scalar (t : Int))
  let indexed := timed.setReg "offsets" .int [B] ⟨fun j => ((off j*2 : Nat) : Int)⟩
  let realLoaded := indexed.setReg "grad_y_real" .real [B] (realTile values)
  let captured := realLoaded.setReg "__veritile_load_0" .real [B] (imagTile (input u gy off mask))
  captured.setReg "grad_y_imag" .real [B] (imagTile values)

theorem step (gy : RegionName) (length batch dim B k : Nat) (u : BlockState) (bound : k<length)
    (col : TileIndex [B] → Nat) (mask : TileIndex [B] → Bool)
    (hi : u.regs .int [] "i"=some (Tile.scalar (k : Int)))
    (hc : u.regs .int [B] "col_offsets"=some ⟨fun j=>(col j : Int)⟩)
    (hm : u.regs .bool [B] "mask"=some ⟨mask⟩) :
    stepStmts (source gy length batch dim B) u = some (result u gy batch dim (length-1-k) col mask) := by
  have indexcast : (length : Int)-1-k=((length-1-k : Nat) : Int) := by omega
  simp only [source,ComplexBackwardSource.body,Erasure.complexBackward,TypedSource.complexBackward,VeriTile.Bench.Repairs.DiagSsmTriton.Kernels.complexBackwardTyped,VeriTile.Bench.Repairs.DiagSsmTriton.complexBackwardKernel,VeriTile.Bench.Repairs.DiagSsmTriton.Kernels.complexBackwardIndexCommuted,ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,
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
    Except.bind,bind,Bool.false_eq_true,↓reduceIte,List.drop,List.take]
  simp only [eq_mp_eq_cast, eq_mpr_eq_cast, cast_eq, id_eq]
  simp only [stepStmts,stepStmt,evalOp,evalOp.eq_def,Option.bind,bind,pure,
    Option.map,Option.pure_def,FloatDType.ofWithBot,FloatDType.toWithBot,
    FloatDType.toTileDType,WithBot.map,Broadcast.leftIndex,Broadcast.rightIndex,Tile.vec,
    String.reduceEq,Tile.bop,Tile.cop,Tile.scalar,NumericDType.mul,NumericDType.add,NumericDType.sub,
    WithBot.realSub,Option.map₂_some_some,Region.cast,
    BlockState.setReg_same,BlockState.setReg_ne_name,BlockState.setReg_pids,
    BlockState.setReg_readMemValue,BlockState.setReg_readMem,BlockState.readMemValue_real,
    Option.bind_some,Option.map_some,Bool.true_eq_false,↓reduceIte,ne_eq,not_false_eq_true,
    hi,hc,hm,Nat.mul_assoc]
  simp only [Int.ofNat_eq_natCast,indexcast,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,IntegerExpr.same_left,IntegerExpr.same_right]
  simp only [result,realTile,imagTile,input,offset,Complex.conj_re,Complex.conj_im]
  simp only [apply_ite,Option.map₂_some_some]
  norm_num only [zero_sub]
  have twice (n : Nat) : (n : Int)*2=((n*2 : Nat) : Int) := by omega
  have plusOne (n : Nat) : (n : Int)+1=((n+1 : Nat) : Int) := by omega
  simp only [twice,plusOne,Int.toNat_natCast,Nat.mul_assoc]
  congr 1
  congr 1
  congr 1
  funext j
  cases hj : mask j <;> simp only [hj,Bool.false_eq_true,if_false,if_true,neg_zero]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexBackwardLoads
