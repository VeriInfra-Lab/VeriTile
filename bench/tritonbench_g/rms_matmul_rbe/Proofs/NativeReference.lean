import bench.tritonbench_g.rms_matmul_rbe.Proofs.NativeRotate

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open RmsMatmulRbe (Parameters RowInput DotPlan Reduction steps masked scaled dotBlock)
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 10000000

def nativeAccumulation (a : Ops α) (p : Parameters α) (x : RowInput α)
    (plans : Nat → DotPlan) (n : Nat) : Nat → α
  | 0 => a.zero
  | t+1 => dotBlock a.toOps p x (plans t) n t (nativeAccumulation a p x plans n t)

/-- Independent selected-native reference: one flat ordered instruction
stream carries the accumulator across K64 block boundaries. It does not
call the source recurrence or any selected-source evaluator. -/
def nativeReferenceAccumulation (a : Ops α) (p : Parameters α) (x : RowInput α)
    (plans : Nat → DotPlan) (n count : Nat) : α :=
  ((List.range count).flatMap (fun t => (plans t).instructions.map (fun lanes => (t,lanes)))).foldl
    (fun seed instruction => a.mma8
      (fun j => scaled a.toOps p x instruction.1 (instruction.2 j))
      (fun j => masked a.toOps p (x.weight n) instruction.1 (instruction.2 j)) seed) a.zero

theorem native_reference_step (a : Ops α) (p : Parameters α) (x : RowInput α)
    (plans : Nat → DotPlan) (n count : Nat) :
    nativeReferenceAccumulation a p x plans n (count+1)=
      dotBlock a.toOps p x (plans count) n count
        (nativeReferenceAccumulation a p x plans n count) := by
  simp [nativeReferenceAccumulation,List.range_succ,List.flatMap_append,
    List.foldl_append,List.foldl_map,dotBlock,RmsMatmulRbe.instruction]

theorem native_accumulation_reference (a : Ops α) (p : Parameters α) (x : RowInput α)
    (plans : Nat → DotPlan) (n count : Nat) :
    nativeAccumulation a p x plans n count=nativeReferenceAccumulation a p x plans n count := by
  induction count with
  | zero => rfl
  | succ t ih => rw [nativeAccumulation,native_reference_step,ih]

def nativeReferenceProjection (a : Ops α) (p : Parameters α) (x : RowInput α)
    (plans : Nat → Nat → DotPlan) (r : Reduction) (n : Nat) : α :=
  a.mul (nativeReferenceAccumulation a p x (plans n) n (steps p.K)) (referenceNorm a p x r)

def nativeScalarPair (a : Arithmetic α) (plan : RotaryPlan) (real imag phase : α) : α × α :=
  let cosine := a.cos phase
  let sine := a.sin phase
  (orderedFma a plan.swapFactors real cosine
      (a.neg (orderedMul a plan.swapFactors imag sine)),
    if plan.fuseRealSin then
      orderedFma a plan.swapFactors real sine (orderedMul a plan.swapFactors imag cosine)
    else orderedFma a plan.swapFactors imag cosine (orderedMul a plan.swapFactors real sine))

def nativeReferenceValue (a : Arithmetic α) (plan : RotaryPlan) (p : Parameters α)
    (x : RowInput α) (rotary : Bool) (row n : Nat) : α :=
  let project := nativeReferenceProjection a.toScalar p x (fun _ _ => a.plan) a.reduction
  if rotary then
    let even := n-n%2
    let phase := a.divide (a.ofInt (p.start+(row:Int)))
      (a.pow p.theta (a.divide (a.ofInt ((even%p.headDim:Nat):Int)) (a.literalNat p.headDim)))
    let pair := nativeScalarPair a plan (project even) (project (even+1)) phase
    if n%2=0 then pair.1 else pair.2
  else project n

#axiomsClean native_reference_step
#axiomsClean native_accumulation_reference
#specNonCircular nativeReferenceValue avoiding [nativeAccumulation,accumulation,
  RmsMatmulRbe.accumulation,nativeRotationExecution,nativeRotated]
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
