import bench.tritonbench_g.diag_ssm_triton.Proofs.Mathematical_ComplexBackwardStep
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexBackwardPrevious
open VeriTile.Triton ComplexForwardStep
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def source (s y : RegionName) (batch dim B : Nat) : List Stmt :=
  ((ComplexBackwardSource.body s y "gx" "gy" 0 batch dim B).drop 5).take 1

def result {B : Nat} (u : BlockState) (s y : RegionName) (batch dim t : Nat)
    (col : TileIndex [B] → Nat) (mask : TileIndex [B] → Bool) : BlockState :=
  let prev := ComplexBackwardStep.previous u s y batch dim t col mask
  let re := u.setReg "s_real" .real [B] (realTile prev)
  re.setReg "s_imag" .real [B] (imagTile prev)

theorem previous_address_imag (batch dim t c : Nat) (ht : 0<t) :
    (Int.ofNat ((t*(batch*dim)+c)*2) - 2*Int.ofNat batch*Int.ofNat dim+1).toNat =
      ((t-1)*(batch*dim)+c)*2+1 := by
  exact ComplexBackwardStep.previous_address batch dim t c 1 ht

theorem step (s y : RegionName) (batch dim B t : Nat) (u : BlockState)
    (col : TileIndex [B] → Nat) (mask : TileIndex [B] → Bool)
    (ht : u.regs .int [] "t"=some (Tile.scalar (t : Int)))
    (hc : u.regs .int [B] "col_offsets"=some ⟨fun j=>(col j : Int)⟩)
    (ho : u.regs .int [B] "offsets"=some ⟨fun j => ((offset batch dim t col j*2 : Nat) : Int)⟩)
    (hm : u.regs .bool [B] "mask"=some ⟨mask⟩) :
    stepStmts (source s y batch dim B) u = some (result u s y batch dim t col mask) := by
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
  have twice (n : Nat) : (n : Int)*2=((n*2 : Nat) : Int) := by omega
  have twiceLeft (n : Nat) : 2*(n : Int)=((2*n : Nat) : Int) := by omega
  have plusOne (n : Nat) : (n : Int)+1=((n+1 : Nat) : Int) := by omega
  cases t with
  | zero =>
    simp only [stepStmts,stepStmt,evalOp,evalOp.eq_def,Option.bind,bind,pure,
      Option.map,Option.pure_def,Broadcast.leftIndex,Broadcast.rightIndex,Tile.vec,
      String.reduceEq,Tile.bop,Tile.cop,Tile.scalar,NumericDType.mul,NumericDType.add,
      NumericDType.sub,ComparableDType.gt,
      Region.cast,BlockState.setReg_same,BlockState.setReg_ne_name,
      BlockState.setReg_readMemValue,BlockState.readMemValue_real,
      ht,hc,hm,ho,Bool.false_eq_true,decide_eq_true_eq,ite_false,ite_true,↓reduceIte]
    norm_num only
    simp only [ite_false,ite_true,Option.bind,bind,pure,Option.pure_def,
      BlockState.setReg_ne_name,BlockState.setReg_readMem,hc,hm,ho,
      String.reduceEq,ne_eq,not_false_eq_true]
    simp only [result,ComplexBackwardStep.previous,realTile,imagTile,input,↓reduceIte]
    simp only [apply_ite,Nat.mul_comm 2,Int.ofNat_eq_natCast,twice,twiceLeft,plusOne,Int.toNat_natCast]
    rfl
  | succ t =>
    have hp : 0<t+1 := by omega
    have hip : (0 : Int)<((t+1 : Nat) : Int) := by omega
    have address0 (c : Nat) :
        (((((t+1)*(batch*dim)+c)*2 : Nat) : Int) - ((2*batch : Nat) : Int)*(dim : Int)).toNat =
          (t*(batch*dim)+c)*2 := by
      simpa only [Int.ofNat_eq_natCast,Nat.cast_mul,Nat.cast_ofNat,Nat.add_sub_cancel] using
        ComplexBackwardStep.previous_address_zero batch dim (t+1) c hp
    have address1 (c : Nat) :
        (((((t+1)*(batch*dim)+c)*2 : Nat) : Int) - ((2*batch : Nat) : Int)*(dim : Int)+1).toNat =
          (t*(batch*dim)+c)*2+1 := by
      simpa only [Int.ofNat_eq_natCast,Nat.cast_mul,Nat.cast_ofNat,Nat.add_sub_cancel] using
        previous_address_imag batch dim (t+1) c hp
    simp only [stepStmts,stepStmt,evalOp,evalOp.eq_def,Option.bind,bind,pure,
      Option.map,Option.pure_def,Broadcast.leftIndex,Broadcast.rightIndex,Tile.vec,
      String.reduceEq,Tile.bop,Tile.cop,Tile.scalar,NumericDType.mul,NumericDType.add,
      NumericDType.sub,ComparableDType.gt,
      Region.cast,BlockState.setReg_same,BlockState.setReg_ne_name,
      BlockState.setReg_readMemValue,BlockState.readMemValue_real,
      ht,hc,hm,ho,Bool.false_eq_true,decide_eq_true_eq,ite_false,ite_true,↓reduceIte,hp]
    norm_num only
    simp only [ite_false,ite_true,Option.bind,bind,pure,Option.pure_def,
      BlockState.setReg_ne_name,BlockState.setReg_readMem,hc,hm,ho,
      String.reduceEq,ne_eq,not_false_eq_true]
    simp only [result,ComplexBackwardStep.previous,Nat.succ_ne_zero,realTile,imagTile,input,offset,↓reduceIte]
    simp only [Int.ofNat_eq_natCast]
    try simp only [ComplexBackwardStep.previous_address,ComplexBackwardStep.previous_address_zero,previous_address_imag,hp,Nat.succ_pos]
    simp only [apply_ite,twice,twiceLeft,plusOne,Int.toNat_natCast,hip,if_true,address0,address1,Nat.add_sub_cancel]
    rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexBackwardPrevious
