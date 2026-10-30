import bench.tritonbench_g.matmul_dequantize_int4.Proofs.FPCompiledBits
namespace VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.FPFused
open VeriTile.Triton FP.Structural FPTyped
open Int4Matmul.FPSoftware (Word)
open FPCompiled (DotAcc)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

/-- The fused graph keeps the actual loopPrefix through signed nibble extraction,
and the actual suffix starting at live-carry dot. Only the middle dequantization
assignment is a new scalar-FMA instruction; it is not a shared DSL node. -/
def loopPrefix (p : Parameters) := (FPCompiled.body p).take 4
def suffix (p : Parameters) := (FPCompiled.body p).drop 5

/-- The fused instruction starts after exactly the unchanged four source
statements. The zero product has already been rounded by this prefix. -/
theorem original_prefix_boundary (p : Parameters) :
    loopPrefix p = (FPTyped.body p).take 4 := by rfl

/-- Only the one dequantization assignment is replaced by `fmaAssign`.
The suffix includes the explicit compiled live-carry dot and both original
pointer increments. This is a graph partition, not an unfused/fused equality. -/
theorem compiled_body_partition (p : Parameters) :
    FPCompiled.body p = loopPrefix p ++ ((FPCompiled.body p).drop 4).take 1 ++ suffix p := by rfl

def bValue (M : Algebra Word) (p : Parameters) (s : State Word) (step : Nat) (i : BIndex p) : Word :=
  QuantizeKvTransform.MixedBits.embed (FPCompiled.HalfFMA.fma .ieeeValue
    (MatmulDequantInt4.Binary16.round .ieeeValue (weightNibble p s step i : Rat))
    (QuantizeKvTransform.MixedBits.half (scaleValue p s step (i.2.1,PUnit.unit)))
    (QuantizeKvTransform.MixedBits.half (zeroValue M p s step (i.2.1,PUnit.unit))).neg)

def accumulation (M : Algebra Word) (D : DotAcc Word)
    (p : Parameters) (s : State Word) : Nat → CIndex p → Word
  | 0 => fun _ => M.literal (some .fp32) .real 0
  | step + 1 => D.apply (some .fp32) {} (batch := []) (aValue M p s step)
      (fun j => M.cast (some .fp32) .fp16 .fp16 (bValue M p s step j))
      (accumulation M D p s step)

def outputValue (M : Algebra Word) (D : DotAcc Word)
    (p : Parameters) (s : State Word) (i : CIndex p) : Word :=
  M.cast none .real .fp16 (accumulation M D p s (steps p) i)

structure Invariant (M : Algebra Word) (D : DotAcc Word)
    (p : Parameters) (initial : State Word) (k : Nat) (current : State Word) : Prop where
  memory : current.mem = initial.mem
  pidm : current.regs .nat [] "pid_m" = some (fun _ => rowBlock p initial)
  pidn : current.regs .nat [] "pid_n" = some (fun _ => columnBlock p initial)
  numk : current.regs .nat [] "num_pid_k" = some (fun _ => steps p)
  bits : current.regs .nat [] "bits" = some (fun _ => 4)
  inf : current.regs .nat [] "infearure_per_bits" = some (fun _ => 8)
  shift : current.regs .nat [p.blockInner] "shifter" = some (fun i => i.1.val % 8 * 4)
  zshift : current.regs .nat [p.blockColumns] "zeros_shifter" =
    some (fun i => inputColumn p initial i.1 % 8 * 4)
  amask : current.regs .bool [p.blockRows,1] "a_mask" =
    some (fun i => decide (inputRow p initial i.1 < p.rows))
  aptrs : current.regs .ptr [p.blockRows,p.blockInner] "a_ptrs" =
    some (fun i => (p.input, aOffset p initial k i))
  bptrs : current.regs .ptr [p.blockInner,p.blockColumns] "b_ptrs" =
    some (fun i => (p.weight, bOffset p initial k i))
  sptrs : current.regs .ptr [p.blockColumns] "scales_ptrs" =
    some (fun i => (p.scale, inputColumn p initial i.1 * p.scaleColumnStride))
  zptrs : current.regs .ptr [p.blockColumns] "zeros_ptrs" =
    some (fun i => (p.zero, inputColumn p initial i.1 / 8 * p.zeroColumnStride))
  scales : p.noGroups = Bool.true → current.regs .fp16 [p.blockColumns] "scales" =
    some (scaleValue p initial 0)
  zeros : p.noGroups = Bool.true → current.regs .fp16 [p.blockColumns] "zeros" =
    some (zeroValue M p initial 0)
  acc : current.regs .real [p.blockRows,p.blockColumns] "accumulator" =
    some (accumulation M D p initial k)


def preludeExecution (M : Algebra Word) (D : DotAcc Word) (p : Parameters) (s : State Word) :=
  FPCompiled.preludeExecution M D p s

theorem preludeInvariant (M : Algebra Word) (D : DotAcc Word) (p : Parameters) (s : State Word) :
    Invariant M D p s 0 (preludeExecution M D p s).val := by
  have h := FPCompiled.preludeInvariant M D p s
  exact ⟨h.memory,h.pidm,h.pidn,h.numk,h.bits,h.inf,h.shift,h.zshift,h.amask,
    h.aptrs,h.bptrs,h.sptrs,h.zptrs,h.scales,h.zeros,h.acc⟩

private theorem min_lt_form (a b : Nat) : min a b = if a < b then a else b := by
  by_cases h : a < b
  · rw [if_pos h,min_eq_left (Nat.le_of_lt h)]
  · rw [if_neg h,min_eq_right (Nat.le_of_not_gt h)]

def prefixExecution (M : Algebra Word) (D : DotAcc Word)
    (p : Parameters) (initial current : State Word) (k : Nat)
    (h : Invariant M D p initial k current) :
    {t : State Word // run (FPCompiled.backend M D) (loopPrefix p)
      (current.setReg "k" .nat [] (fun _ => k)) = some t} := by
  by_cases hno : p.noGroups = Bool.true <;>
    (refine ⟨?_, ?_⟩
     swap
     simp [loopPrefix,FPCompiled.body,FPCompiled.source,FPCompiled.typed,ComputeKernel.surfaceBody,
       run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,
       bop,natLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
       FloatDType.numericDType,FloatDType.toTileDType,
       Region.cast,Option.bind,integralFloorDiv,integralMod,
       h.aptrs,h.bptrs,h.acc,h.memory,h.amask,h.sptrs,h.zptrs,h.shift,h.zshift,
       h.bits,h.inf,h.scales,h.zeros,hno,FPCompiled.backend]
     rfl)

structure PrefixFacts (M : Algebra Word) (D : DotAcc Word)
    (p : Parameters) (initial : State Word) (k : Nat) (current : State Word) : Prop where
  control : Invariant M D p initial k current
  a : current.regs .fp16 [p.blockRows,p.blockInner] "a" = some (aValue M p initial k)
  b : current.regs .int [p.blockInner,p.blockColumns] "b" = some (weightNibble p initial k)
  scales : current.regs .fp16 [p.blockColumns] "scales" = some (scaleValue p initial k)
  zeros : current.regs .fp16 [p.blockColumns] "zeros" = some (zeroValue M p initial k)

theorem prefixFacts (M : Algebra Word) (D : DotAcc Word)
    (p : Parameters) (initial current : State Word) (k : Nat)
    (h : Invariant M D p initial k current) :
    PrefixFacts M D p initial k (prefixExecution M D p initial current k h).val := by
  have hmask (n : Nat) : n &&& 15 < 4294967296 :=
    lt_of_le_of_lt Nat.and_le_right (by decide)
  have hmaskInt (n : Nat) : ((n &&& 15 : Nat) : Int) % 4294967296 = ((n &&& 15 : Nat) : Int) := by
    apply Int.emod_eq_of_lt (by omega)
    exact_mod_cast hmask n
  cases hno : p.noGroups
  all_goals refine ⟨?_,?_,?_,?_,?_⟩
  all_goals try apply Invariant.mk
  all_goals simp [prefixExecution,h.memory,h.pidm,h.pidn,h.numk,h.bits,h.inf,h.shift,h.zshift,
    h.amask,h.aptrs,h.bptrs,h.sptrs,h.zptrs,h.scales,h.zeros,h.acc,hno,
    FPCompiled.backend,accumulation,aValue,weightNibble,zeroNibble,scaleValue,zeroValue,
    aOffset,bOffset,scaleOffset,zeroOffset,groupIndex,inputInner,
    MatmulDequantInt4.FPTyped.nibble,BitVec.toInt_eq_toNat_cond,BitVec.toNat_and,
    BitVec.toNat_ofInt,Int.shiftRight_eq_div_pow,Nat.shiftLeft_eq,Int.shiftLeft_eq,
    Nat.mod_eq_of_lt (hmask _),hmaskInt,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Nat.add_mul,Nat.add_assoc]

  all_goals funext i
  all_goals simp [aValue,weightNibble,zeroNibble,scaleValue,zeroValue,
    aOffset,bOffset,scaleOffset,zeroOffset,groupIndex,inputInner,hno,
    MatmulDequantInt4.FPTyped.nibble,BitVec.toInt_eq_toNat_cond,BitVec.toNat_and,
    BitVec.toNat_ofInt,Int.shiftRight_eq_div_pow,Nat.shiftLeft_eq,Int.shiftLeft_eq,
    Nat.mod_eq_of_lt (hmask _),hmaskInt,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Nat.add_mul,Nat.add_assoc]
  all_goals norm_num

#axiomsClean original_prefix_boundary
#axiomsClean compiled_body_partition
#axiomsClean preludeInvariant
#axiomsClean prefixExecution
#axiomsClean prefixFacts
end
end VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.FPFused
