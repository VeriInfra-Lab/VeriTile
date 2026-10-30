import bench.tritonbench_g.bmm_optimized.Proofs.Invariant

namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.Execution
open VeriTile.Triton FP.Structural Source
noncomputable section
set_option maxHeartbeats 18000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

private theorem aValue_function (o : BmmOptimized.Scalar α) (p : Parameters)
    (initial : RegisterState α) (k : Nat) : aValue o p initial k=
      (fun i=>if (p.divM=Bool.true ∨ row p initial i.1<p.M) ∧
        (p.divK=Bool.true ∨ inner p k i.2.1<p.K) then
        (initial.mem p.A (aOffset p initial k i)).read .real else o.zero) := rfl
private theorem bValue_function (o : BmmOptimized.Scalar α) (p : Parameters)
    (initial : RegisterState α) (k : Nat) : bValue o p initial k=
      (fun i=>if (p.divK=Bool.true ∨ inner p k i.1<p.K) ∧
        (p.divN=Bool.true ∨ column p initial i.2.1<p.N) then
        (initial.mem p.B (bOffset p initial k i)).read .real else o.zero) := rfl

def maskAResult (p : Parameters) (initial current : RegisterState α) (k : Nat) : RegisterState α :=
  match p.divM,p.divK with
  | Bool.false,Bool.false => current.setReg "mask_a" .bool [p.TM,p.TK]
      (fun i=>decide (row p initial i.1<p.M ∧ inner p k i.2.1<p.K))
  | Bool.false,Bool.true => current.setReg "mask_a" .bool [p.TM,1]
      (fun i=>decide (row p initial i.1<p.M))
  | Bool.true,Bool.false => current.setReg "mask_a" .bool [1,p.TK]
      (fun i=>decide (inner p k i.2.1<p.K))
  | Bool.true,Bool.true => current

def masksResult (p : Parameters) (initial current : RegisterState α) (k : Nat) : RegisterState α :=
  let t := if p.divK then current else current.setReg "mask_k" .bool [p.TK]
    (fun i=>decide (inner p k i.1<p.K))
  let t := maskAResult p initial t k
  match p.divN,p.divK with
  | Bool.false,Bool.false => t.setReg "mask_b" .bool [p.TK,p.TN]
      (fun i=>decide (inner p k i.1<p.K ∧ column p initial i.2.1<p.N))
  | Bool.false,Bool.true => t.setReg "mask_b" .bool [1,p.TN]
      (fun i=>decide (column p initial i.2.1<p.N))
  | Bool.true,Bool.false => t.setReg "mask_b" .bool [p.TK,1]
      (fun i=>decide (inner p k i.1<p.K))
  | Bool.true,Bool.true => t

def bodyResult (o : BmmOptimized.Scalar α) (p : Parameters)
    (initial current : RegisterState α) (k : Nat) : RegisterState α :=
  let t := masksResult p initial current k
  let t := t.setReg "a" .real [p.TM,p.TK] (aValue o p initial k)
  let t := t.setReg "b" .real [p.TK,p.TN] (bValue o p initial k)
  let t := t.setReg "offs_k" .nat [p.TK] (fun i=>inner p (k+1) i.1)
  let t := t.setReg "a_ptrs" .ptr [p.TM,p.TK] (fun i=>(p.A,aOffset p initial (k+1) i))
  let t := t.setReg "b_ptrs" .ptr [p.TK,p.TN] (fun i=>(p.B,bOffset p initial (k+1) i))
  t.setReg "o" .real [p.TM,p.TN] (accumulation o p initial (k+1))

theorem bodyExecution (o : BmmOptimized.Scalar α) (rest : Algebra α)
    (p : Parameters) (initial current : RegisterState α) (k : Nat)
    (h : Invariant o p initial k current) :
    run o rest (body p) current=some (bodyResult o p initial current k) := by
  have hz : (0.0 : ℝ)=0 := by norm_num
  have hm:=h.maskM
  have hn:=h.maskN
  cases dm : p.divM <;> cases dn : p.divN <;> cases dk : p.divK
  all_goals
    simp only [dm,dn,Bool.false_eq_true,false_implies,true_implies] at hm hn
    simp only [body,parts,kernel,VeriTile.Triton.Source.dynamicRange?,ComputeKernel.surfaceBody,
      dm,dn,dk,Option.map,Option.getD,beq_self_eq_true,↓reduceIte]
    simp only [run,step,localStep_dot]
    simp [bodyResult,masksResult,maskAResult,dm,dn,dk,
      run,step,localStep,accumulate,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
      evalOffsets,store,numeric,FP.Structural.bop,natLt,numericLt,numericEq,ComputeDType.eraseDType,
      ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,Region.cast,
      Option.bind,integralFloorDiv,integralMod,algebra,Broadcast.leftIndex,Broadcast.rightIndex,
      h.memory,h.aptrs,h.bptrs,h.offsk,h.acc,hm,hn,hz,aValue_function,bValue_function,aValue,bValue,inner,accumulation,
      aOffset,bOffset,Nat.add_mul,Nat.mul_add,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
    congr 2
    funext i
    apply Prod.ext
    · rfl
    · omega

theorem bodyInvariant (o : BmmOptimized.Scalar α) (p : Parameters)
    (initial current : RegisterState α) (k : Nat) (h : Invariant o p initial k current) :
    Invariant o p initial (k+1) (bodyResult o p initial current k) := by
  cases dm : p.divM <;> cases dn : p.divN <;> cases dk : p.divK
  all_goals constructor <;>
    simp [bodyResult,masksResult,maskAResult,dm,dn,dk,
      h.memory,h.pids,h.numPids,h.maskM,h.maskN,h.optrs,h.iterations]

#axiomsClean bodyExecution
#axiomsClean bodyInvariant
end
end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.Execution
