import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.Widened
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.NativeReturnedMath
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealPhysical

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.Whole002.Execution
open VeriTile Triton Geometry Physical FPStorage FP.Structural
open WholeGrid (Job jobs)
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 16000

def kernel (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool) : ComputeKernel :=
  let p := parameters c offset size scale window complement
  TypedSource.widened p.Q p.K p.M p.Out
    p.queryBatch p.queryHead p.queryRow p.queryDim p.keyBatch p.keyHead p.keyRow p.keyDim
    p.outputBatch p.outputHead p.outputInner p.batches p.heads p.kvHeads p.queryContext p.roundedContext
    p.keyContext c.B c.D c.B p.offset p.size p.scale p.window p.complement p.evenM p.evenN rfl

theorem kernel_eq (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool) :
    kernel c offset size scale window complement=DenseSource.kernel c offset size scale window complement := by
  simp only [kernel,TypedSource.widened_eq,DenseSource.kernel,RealSource.source,Bool.false_eq_true,ite_false]

namespace ExactReal

def invoke (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool)
    (job : Job c) (s : BlockState) : Option BlockState :=
  Triton.exec (kernel c offset size scale window complement).eraseDType.toAlgKernel
    (DenseSource.programState c s job.2.2.val job.1.val job.2.1.val)

def run (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool) :
    List (Job c) → BlockState → Option BlockState
  | [],s => some s
  | job::rest,s => (invoke c offset size scale window complement job s).bind
      (run c offset size scale window complement rest)

theorem run_eq (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool)
    (order : List (Job c)) (s : BlockState) :
    run c offset size scale window complement order s=WholeGrid.run c offset size scale window complement order s := by
  induction order generalizing s with
  | nil => rfl
  | cons job rest ih => simp only [run,WholeGrid.run,invoke,WholeGrid.invoke,kernel_eq,ih]
      <;> rfl

def hostRun (c : Shape) (s : BlockState) (offset size : Int) (scale : ℝ)
    (window complement : Bool) : Option BlockState :=
  let initialized := RealReturned.zeroOutput c s
  if c.N=0 ∨ c.outputSize=0 then some initialized
  else run c offset size scale window complement (jobs c) initialized

def execute (c : Shape) (abi : ABI c) (physical : Nat → ℝ) (offset size : Int) (scale : ℝ)
    (window complement : Bool) : Option (Nat → ℝ) :=
  (hostRun c (RealPhysical.initial abi physical) offset size scale window complement).map
    (fun t => flush (abi.alloc "out") (t.readMem "out") physical)

theorem execute_eq (c : Shape) (abi : ABI c) (physical : Nat → ℝ) (offset size : Int) (scale : ℝ)
    (window complement : Bool) : execute c abi physical offset size scale window complement=
      RealPhysical.execute c abi physical offset size scale window complement := by
  simp only [execute,RealPhysical.execute,hostRun,RealReturned.hostRun,run_eq]
end ExactReal

namespace Native
variable {α : Type} [Inhabited α]

def invoke (M : Algebra α) (dot : FPEngine.Dot α) (c : Shape) (offset size : Int) (scale : ℝ)
    (window complement : Bool) (job : Job c) (s : State α) : Option (State α) :=
  FP.Structural.exec (FP.ProfiledRewrite.engine (FPEngine.withDot M dot))
    (kernel c offset size scale window complement) (FPGrid.programState c s job)

def run (M : Algebra α) (dot : FPEngine.Dot α) (c : Shape) (offset size : Int) (scale : ℝ)
    (window complement : Bool) : List (Job c) → State α → Option (State α)
  | [],s => some s
  | job::rest,s => (invoke M dot c offset size scale window complement job s).bind
      (run M dot c offset size scale window complement rest)

theorem run_eq (M : Algebra α) (dot : FPEngine.Dot α) (c : Shape) (offset size : Int) (scale : ℝ)
    (window complement : Bool) (order : List (Job c)) (s : State α) :
    run M dot c offset size scale window complement order s=FPGrid.run M dot c offset size scale window complement order s := by
  induction order generalizing s with
  | nil => rfl
  | cons job rest ih =>
      simp only [run,FPGrid.run,invoke,FPGrid.invoke,kernel_eq,DenseSource.kernel,FPEngine.kernel,ih]

def hostRun (M : Algebra α) (dot : FPEngine.Dot α) (c : Shape) (s : State α) (offset size : Int)
    (scale : ℝ) (window complement : Bool) : Option (State α) :=
  let initialized := FPReturned.zeroOutput M c s
  if c.N=0 ∨ c.outputSize=0 then some initialized
  else run M dot c offset size scale window complement (jobs c) initialized

def physicalRun (M : Algebra α) (dot : FPEngine.Dot α) (c : Shape) (abi : ABI c) (physical : Nat → α)
    (offset size : Int) (scale : ℝ) (window complement : Bool) : Option (Nat → α) :=
  (hostRun M dot c (Physical.initial abi physical) offset size scale window complement).map
    (fun t => flush (abi.alloc "out") (exportMemory t "out") physical)

def execute (c : Shape) (abi : ABI c) (physical : Nat → BitVec 32)
    (offset size : Int) (scale : ℝ) (window complement : Bool) : Option (Nat → NativeInstructions.Graph) :=
  (physicalRun NativeStaging.algebra NativeStaging.dotModel c abi (NativePhysical.input physical)
    offset size scale window complement).map (fun result a => NativeStaging.lower (result a))

theorem execute_eq (c : Shape) (abi : ABI c) (physical : Nat → BitVec 32)
    (offset size : Int) (scale : ℝ) (window complement : Bool) :
    execute c abi physical offset size scale window complement=
      NativePhysical.execute c abi physical offset size scale window complement := by
  simp only [execute,NativePhysical.execute,physicalRun,Physical.execute,hostRun,FPReturned.hostRun,run_eq]
end Native

#axiomsClean kernel_eq
#axiomsClean ExactReal.execute_eq
#axiomsClean Native.execute_eq
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.Whole002.Execution
