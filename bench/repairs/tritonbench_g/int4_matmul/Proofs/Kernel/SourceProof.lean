import bench.repairs.tritonbench_g.int4_matmul.Proofs.Kernel.Reference

namespace VeriTile.Bench.Repairs.Int4Matmul.KernelExecution
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false

private theorem min_lt_form (a b : Nat) : min a b = if a<b then a else b := by
  by_cases h : a<b
  · rw [if_pos h,min_eq_left (Nat.le_of_lt h)]
  · rw [if_neg h,min_eq_right (Nat.le_of_not_gt h)]

structure Invariant [Inhabited α] (o : Ops α)
    (p : Parameters) (initial : RegisterState α) (k : Nat) (current : RegisterState α) : Prop where
  memory : current.mem=initial.mem
  pidm : current.regs .nat [] "pid_m"=some (fun _ => rowBlock p initial)
  pidn : current.regs .nat [] "pid_n"=some (fun _ => columnBlock p initial)
  offsk : current.regs .nat [p.blockInner] "offs_k"=some (fun i => inputInner p initial i.1)
  offsn : current.regs .nat [p.blockColumns] "offs_bn"=some (fun i => inputColumn p initial i.1)
  aptrs : current.regs .ptr [p.blockRows,p.blockInner] "a_ptrs"=
    some (fun i => (p.input,aOffset p initial k i))
  bptrs : current.regs .ptr [p.blockInner,p.blockColumns] "b_ptrs"=
    some (fun i => (p.weight,bOffset p initial k i))
  acc : current.regs .real [p.blockRows,p.blockColumns] "accumulator"=
    some (accumulation o p initial k)

def preludeExecution [Inhabited α] (o : Ops α) (rest : Algebra α)
    (p : Parameters) (s : RegisterState α) :
    {t : RegisterState α // run o rest (prelude p) s=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [prelude,kernel,ComputeKernel.surfaceBody,
    run,step,localStep,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,
    bop,natLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
    FloatDType.numericDType,FloatDType.toTileDType,
    Region.cast,Option.bind,integralFloorDiv,integralMod,algebra]
  rfl

theorem preludeInvariant [Inhabited α] (o : Ops α) (rest : Algebra α)
    (p : Parameters) (s : RegisterState α) :
    Invariant o p s 0 (preludeExecution o rest p s).val := by
  constructor <;>
    simp [preludeExecution,accumulation,blocksM,blocksN,firstRow,
      rowsInGroup,rowBlock,columnBlock,inputInner,inputColumn,inputRow,
      aOffset,bOffset,min_lt_form,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals funext i
  all_goals simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]

def bodyExecution [Inhabited α] (o : Ops α) (rest : Algebra α)
    (p : Parameters) (initial current : RegisterState α) (k : Nat)
    (h : Invariant o p initial k current) :
    {t : RegisterState α // run o rest (body p)
      (current.setReg "k" .nat [] (fun _ => k))=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [body,kernel,ComputeKernel.surfaceBody,run,step,localStep_dot]
  simp [body,kernel,ComputeKernel.surfaceBody,
    run,step,localStep,localStep_dot,accumulate,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,
    bop,natLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
    FloatDType.numericDType,FloatDType.toTileDType,
    Region.cast,Option.bind,integralFloorDiv,integralMod,
    h.offsk,h.offsn,h.aptrs,h.bptrs,h.acc,h.memory,algebra,defaultValue]
  rfl

theorem bodyInvariant [Inhabited α] (o : Ops α) (rest : Algebra α)
    (p : Parameters) (initial current : RegisterState α) (k : Nat)
    (h : Invariant o p initial k current) :
    Invariant o p initial (k+1) (bodyExecution o rest p initial current k h).val := by
  have hmask (n : Nat) : n &&& 15<4294967296 :=
    lt_of_le_of_lt Nat.and_le_right (by decide)
  have hmaskInt (n : Nat) : ((n &&& 15 : Nat) : Int)%4294967296=((n &&& 15 : Nat) : Int) := by
    apply Int.emod_eq_of_lt (by omega)
    exact_mod_cast hmask n
  constructor <;>
    simp [bodyExecution,h.memory,h.pidm,h.pidn,h.offsk,h.offsn,
      h.aptrs,h.bptrs,h.acc,accumulation,aValue,bValue,difference,
      aOffset,bOffset,scaleOffset,zeroOffset,innerIndex,
      signedNibble,VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.nibble,
      BitVec.toInt_eq_toNat_cond,BitVec.toNat_and,BitVec.toNat_ofInt,
      Int.shiftRight_eq_div_pow,Nat.shiftLeft_eq,Int.shiftLeft_eq,
      Nat.mod_eq_of_lt (hmask _),hmaskInt,
      FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Nat.add_mul,Nat.add_assoc]
  all_goals funext i
  all_goals try simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,aValue,aOffset,innerIndex,Nat.add_assoc]
  all_goals rcases i with ⟨r,c,u⟩
  all_goals cases u
  all_goals congr 1
  all_goals funext j
  all_goals simp [aValue,bValue,difference,aOffset,bOffset,scaleOffset,zeroOffset,innerIndex,
    signedNibble,VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.nibble,
    BitVec.toInt_eq_toNat_cond,BitVec.toNat_and,BitVec.toNat_ofInt,
    Int.shiftRight_eq_div_pow,Nat.mod_eq_of_lt (hmask _),hmaskInt,Nat.add_assoc]

#axiomsClean preludeExecution
#axiomsClean preludeInvariant
#axiomsClean bodyExecution
#axiomsClean bodyInvariant
end
end VeriTile.Bench.Repairs.Int4Matmul.KernelExecution
