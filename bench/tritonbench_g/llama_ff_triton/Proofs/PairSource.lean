import bench.tritonbench_g.llama_ff_triton.Proofs.Source
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.PairSource
open VeriTile.Triton FP.Structural Execution
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 6000000

def candidate (p : Parameters) : ComputeKernel :=
  Source.candidate p.byte p.rows p.columns p.inner p.epsilon

def indexStmt (B : Nat) (commuted : Bool) : ComputeStmt :=
  .assign .int [B] "offs_outm" (.alg (.add .int .scalarL
    (if commuted then .mul .int .nil (.constInt (Int.ofNat B)) (.ref .int [] "pid_m")
     else .mul .int .nil (.ref .int [] "pid_m") (.constInt (Int.ofNat B)))
    (.castNatToInt (.arange B))))

def before (p : Parameters) : List ComputeStmt := (source p).surfaceBody.take 31
def after (p : Parameters) : List ComputeStmt := (source p).surfaceBody.drop 32

theorem original_partition (p : Parameters) :
    (source p).surfaceBody = before p ++ [indexStmt 16 Bool.false] ++ after p := by
  rcases p with ⟨M,N,K,fp8,eps⟩
  cases fp8 <;> rfl

theorem candidate_partition (p : Parameters) :
    (candidate p).surfaceBody = before p ++ [indexStmt 16 Bool.true] ++ after p := by
  rcases p with ⟨M,N,K,fp8,eps⟩
  cases fp8 <;> rfl

/-- Exact signed integer multiplication; arbitrary FP operation models and all
input payloads are admitted. No numeric rule or finite-value condition. -/
theorem index_step {α : Type} [Inhabited α] (a : Algebra α) (s : State α) (B : Nat) :
    step a (indexStmt B Bool.false) s = step a (indexStmt B Bool.true) s := by
  simp only [indexStmt, Bool.false_eq_true, ite_false, ite_true, step, evalExpr,
    evalOp_unfold, numeric, bop]
  cases h : s.regs .int [] "pid_m" with
  | none => rfl
  | some v =>
    change some (s.setReg "offs_outm" .int [B]
      (fun i => v PUnit.unit * (B : Int) + (i.1.val : Int))) =
      some (s.setReg "offs_outm" .int [B]
        (fun i => (B : Int) * v PUnit.unit + (i.1.val : Int)))
    congr 2
    funext i
    rw [Int.mul_comm]

theorem exact {α : Type} [Inhabited α] (a : Algebra α) (p : Parameters) (s : State α) :
    exec a (source p) s = exec a (candidate p) s := by
  change run a (source p).surfaceBody s = run a (candidate p).surfaceBody s
  rw [original_partition, candidate_partition]
  simp only [run_append]
  cases h : run a (before p) s with
  | none => rfl
  | some t =>
    simp only [Option.bind_some, run, index_step]

/-- Both variants have the same actual memory access safety. The changed
statement computes only signed indices, with no memory subexpression. -/
theorem index_safe {α : Type} [Inhabited α] (a : Algebra α) (s : State α)
    (B : Nat) (bounds : RegionName → Nat) (variant : Bool) :
    Safety.stepSafe a bounds (indexStmt B variant) s := by
  cases variant <;> simp [indexStmt, Safety.stepSafe, Safety.safeExpr, Safety.safeOp]

theorem safe_iff {α : Type} [Inhabited α] (a : Algebra α) (p : Parameters)
    (s : State α) (bounds : RegionName → Nat) :
    TraceSafe a bounds (source p) s ↔
      TraceSafe a bounds (candidate p) s := by
  change Safety.traceSafeList a bounds (source p).surfaceBody s ↔
    Safety.traceSafeList a bounds (candidate p).surfaceBody s
  rw [original_partition, candidate_partition]
  simp only [List.append_assoc, List.singleton_append, Safety.traceSafeList_append,
    Safety.traceSafeList_cons, index_safe, index_step, true_and]

#axiomsClean original_partition
#axiomsClean candidate_partition
#axiomsClean index_step
#axiomsClean exact
#axiomsClean safe_iff
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.PairSource
