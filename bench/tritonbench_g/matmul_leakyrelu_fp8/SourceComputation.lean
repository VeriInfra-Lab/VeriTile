import bench.tritonbench_g.matmul_leakyrelu_fp8.SourceGrid

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.Execution
open VeriTile.Triton FP.Structural
open MatmulLeakyrelu.KernelOnly MatmulLeakyreluFp8 MatmulLeakyreluFp8.Checked
noncomputable section
open Classical
set_option maxHeartbeats 6000000

def replay {α : Type} [Inhabited α] (M : Algebra α) (ops : Operations α)
    (p : Problem) (b : Buffers) (scratch : State α) : Nat → Memory α → Option (Memory α)
  | 0,memory => some memory
  | count+1,memory => (grid M ops p b scratch memory).bind (replay M ops p b scratch count)

theorem replay_source {α : Type} [Inhabited α] (M : Algebra α) (ops : Operations α)
    (law : ScalarLaws M ops) (p : Problem) (b : Buffers) (layout : Layout p b)
    (scratch : State α) (count : Nat) (memory : Memory α) :
    replay M ops p b scratch count memory =
      some (Source.replay (sourceArithmetic ops) p b count memory) := by
  induction count generalizing memory with
  | zero => rfl
  | succ count ih =>
    simp only [replay,grid_source M ops law p b layout,Option.bind_some,ih,Source.replay]

/-- Empty and zero-K preserve the actual no-launch source branches. Positive K
executes the shared typed quotation for every PID and every finite autotuner
trial plus its final launch. Zero-fill is the existing individual-store code;
there is no host lifecycle or allocation-success gate. -/
def computation {α : Type} [Inhabited α] (M : Algebra α) (ops : Operations α)
    (r : Request) (c : OutputTensor) (trials : Nat) (scratch : State α) (memory : Memory α) :
    Option (Memory α) :=
  if r.A.rows = 0 ∨ r.B.columns = 0 then some memory
  else if r.A.columns = 0 then some ((zeroCode (outputDescriptor r c)).eval ops memory).2
  else replay M ops (arguments r).problem ⟨(arguments r).A,(arguments r).B,outputDescriptor r c⟩
    scratch (trials+1) memory

theorem computation_source {α : Type} [Inhabited α] (M : Algebra α) (ops : Operations α)
    (law : ScalarLaws M ops) (r : Request) (c : OutputTensor) (h : Contract r c)
    (trials : Nat) (scratch : State α) (memory : Memory α) :
    computation M ops r c trials scratch memory =
      some (resultMemory ops (arguments r) (outputDescriptor r c) memory) := by
  unfold computation resultMemory
  change (if r.A.rows = 0 ∨ r.B.columns = 0 then _ else if r.A.columns = 0 then _ else _) =
    some (if r.A.rows = 0 ∨ r.B.columns = 0 then _ else if r.A.columns = 0 then _ else _)
  split_ifs
  · rfl
  · rw [eval_zero]
  · rw [replay_source M ops law _ _ (layout r c h),Source.replay_positive _ _ _ (layout r c h)]

/-- The requested source-interpreter→event connection is a theorem. Both sides
execute their programs; neither side is supplied as a caller hypothesis. -/
theorem source_event {α : Type} [Inhabited α] (M : Algebra α) (ops : Operations α)
    (law : ScalarLaws M ops) (r : Request) (c : OutputTensor) (h : Contract r c)
    (trials : Nat) (scratch : State α) (read : Nat → α) (before : Memory α) :
    computation M ops r c trials scratch (initial r read before) =
      some (eventRun ops r c trials read before).2 := by
  rw [computation_source M ops law r c h,event_complete ops r c h]

#specNonCircular computation avoiding [reference,logicalReference,resultMemory]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.Execution
