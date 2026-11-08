import bench.tritonbench_g.rms_rbe_matmul.Proofs.KernelModel
import bench.tritonbench_g.rms_matmul_rbe.Proofs.PrefixValues

namespace VeriTile.Bench.Repairs.RmsRbeMatmul.KernelSource
open VeriTile.Triton FP.Structural
open RmsMatmulRbeScalarDomain (signed_nat_negative signed_columns signed_column_expression
  signed_lane16_negative signed_lane64_negative signed_remainder signed_nat_remainder
  signed_nat_remainder_product)
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 35000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

private theorem cast_columns (n : Nat) :
    ((n:Int)+63)/64=(((n+63)/64:Nat):Int) := by omega

def xPointer (p : Parameters) (s : State α) (step : Nat) : Values α .ptr [16,64] :=
  fun i => ("X", xOffset p s step i)
def wPointer (p : Parameters) (s : State α) (step : Nat) : Values α .ptr [64,64] :=
  fun i => ("W", wOffset p s step i)
def rPointer (step : Nat) : Values α .ptr [1,64] := fun i => ("RMS", rOffset step i)
structure Ready (p : Parameters) (s t : State α) (step : Nat) (acc : Accumulator α) : Prop where
  mem : t.mem=s.mem
  pids : t.pids=s.pids
  undef : t.undef=s.undef
  batch : t.regs .int [] "pid_batch"=some (fun _ => (s.pids 0:Int))
  tileM : t.regs .int [] "pid_m"=some (fun _ => (pidM p s:Int))
  tileN : t.regs .int [] "pid_n"=some (fun _ => (pidN p s:Int))
  offk : t.regs .int [64] "offs_k"=some (fun i => (i.1.val:Int))
  xptr : t.regs .ptr [16,64] "x_ptrs"=some (xPointer p s step)
  wptr : t.regs .ptr [64,64] "w_ptrs"=some (wPointer p s step)
  rptr : t.regs .ptr [1,64] "rms_w_ptrs"=some (rPointer step)
  product : t.regs .real [16,64] "accumulator"=some acc.product
  squares : t.regs .real [16,64] "x_sum"=some acc.squares

theorem prefix_run (a : Operations α) (fp8 : Bool) (p : Parameters) (s : State α) :
    ∃ t, run (engine a) (prelude fp8 p) s=some t ∧ Ready p s t 0 (initial a) := by
  have source_zero : (0.0:ℝ)=0 := by norm_num
  cases fp8 <;>
    simp [prelude,kernel,TypedSource.kernel,TypedSource.source,arguments,
      ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,ComputeDType.eraseDType,numeric,numericEq,FP.Structural.bop,natLt,numericLt,
      integralFloorDiv,integralMod,Region.cast,ofFloat,toFloat,State.setReg,
      engine,Algebra.withDefaultPrecision,resolvePrecision,source_zero,signed_nat_negative,
      signed_column_expression,Nat.add_sub_assoc]
  all_goals constructor
  all_goals try simp [State.setReg,pidM,pidN,xPointer,wPointer,rPointer,xOffset,wOffset,rOffset,
    row,col,initial,zero]
  all_goals first | rfl | (funext i; try simp [FP.Structural.bop,initial,zero,xPointer,wPointer,rPointer,
    xOffset,wOffset,rOffset,row,col,pidM,pidN])
  all_goals
    try simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
      signed_nat_negative,if_false,signed_column_expression]
    try simp only [←Int.natCast_add,←Int.natCast_mul,←Int.natCast_ediv,
      signed_nat_negative,if_false,signed_nat_remainder,Int.toNat_natCast]
    try simp only [signed_nat_remainder_product,←Int.natCast_mul _ 16,←Int.natCast_mul _ 64,
      ←Int.natCast_add,signed_nat_negative,if_false,signed_nat_remainder,
      ←Int.natCast_mul,Int.toNat_natCast,Nat.zero_add,Nat.add_sub_assoc]
    try simp only [signed_lane16_negative,signed_lane64_negative,if_false,signed_remainder]
    try norm_cast
    try simp [Nat.add_sub_assoc]
    try simp only [signed_remainder]
    try simp only [cast_columns,←Int.natCast_ediv,←Int.natCast_emod,
      signed_lane16_negative,signed_lane64_negative,if_false]
    try simp only [←Int.natCast_add,←Int.natCast_mul,←Int.natCast_ediv,
      ←Int.natCast_emod,signed_nat_negative,if_false,signed_nat_remainder,
      signed_nat_remainder_product,Int.toNat_natCast]
    try simp only [signed_remainder,←Int.natCast_add,←Int.natCast_mul,
      ←Int.natCast_emod,Int.toNat_natCast]
    try rfl


#axiomsClean prefix_run
end
end VeriTile.Bench.Repairs.RmsRbeMatmul.KernelSource
