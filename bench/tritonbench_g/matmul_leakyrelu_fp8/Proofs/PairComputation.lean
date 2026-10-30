import bench.tritonbench_g.matmul_leakyrelu_fp8.Proofs.Pair

/-! The same source-local change through the retained physical-memory adapter,
all finite trial replays and the unchanged no-launch wrapper branches. -/
namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.Pair
open VeriTile.Triton FP.Structural
open MatmulLeakyrelu.KernelOnly MatmulLeakyreluFp8 MatmulLeakyreluFp8.Checked Execution
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 40000
variable {α : Type} [Inhabited α]

def candidatePrograms (M : Algebra α) (ops : Operations α) (p : Problem) (b : Buffers)
    (scratch : State α) : List Nat → Memory α → Option (Memory α)
  | [],memory => some memory
  | pid::rest,memory =>
    (exec (backend M (selectedDot ops)) (candidate p b)
      (programState b memory pid scratch)).bind
        (fun final => candidatePrograms M ops p b scratch rest (project b memory final))

theorem programs_equal (M : Algebra α) (ops : Operations α) (h : MulCommutes M)
    (p : Problem) (b : Buffers) (scratch : State α) (ids : List Nat) (memory : Memory α) :
    runPrograms M ops p b scratch ids memory = candidatePrograms M ops p b scratch ids memory := by
  induction ids generalizing memory with
  | nil => rfl
  | cons pid rest ih =>
    simp only [runPrograms,candidatePrograms,← kernel_equal M (selectedDot ops) h]
    cases hr : exec (backend M (selectedDot ops)) (source p b)
        (programState b memory pid scratch) with
    | none => rfl
    | some final => exact ih _

def candidateGrid (M : Algebra α) (ops : Operations α) (p : Problem) (b : Buffers)
    (scratch : State α) (memory : Memory α) : Option (Memory α) :=
  candidatePrograms M ops p b scratch (Source.tiles p) memory

theorem grid_equal (M : Algebra α) (ops : Operations α) (h : MulCommutes M)
    (p : Problem) (b : Buffers) (scratch : State α) (memory : Memory α) :
    grid M ops p b scratch memory=candidateGrid M ops p b scratch memory :=
  programs_equal M ops h p b scratch _ memory

def candidateReplay (M : Algebra α) (ops : Operations α) (p : Problem) (b : Buffers)
    (scratch : State α) : Nat → Memory α → Option (Memory α)
  | 0,memory => some memory
  | count+1,memory => (candidateGrid M ops p b scratch memory).bind
      (candidateReplay M ops p b scratch count)

theorem replay_equal (M : Algebra α) (ops : Operations α) (h : MulCommutes M)
    (p : Problem) (b : Buffers) (scratch : State α) (count : Nat) (memory : Memory α) :
    replay M ops p b scratch count memory=candidateReplay M ops p b scratch count memory := by
  induction count generalizing memory with
  | zero => rfl
  | succ count ih =>
    simp only [replay,candidateReplay,← grid_equal M ops h]
    cases hr : grid M ops p b scratch memory with
    | none => rfl
    | some final => exact ih _

/-- Exact same empty and K=0 branches as the selected wrapper. Positive K
executes the candidate quotation through the unchanged physical adapter. -/
def candidateComputation (M : Algebra α) (ops : Operations α)
    (r : Request) (c : OutputTensor) (trials : Nat) (scratch : State α) (memory : Memory α) :
    Option (Memory α) :=
  if r.A.rows=0 ∨ r.B.columns=0 then some memory
  else if r.A.columns=0 then some ((zeroCode (outputDescriptor r c)).eval ops memory).2
  else candidateReplay M ops (arguments r).problem
    ⟨(arguments r).A,(arguments r).B,outputDescriptor r c⟩ scratch (trials+1) memory

theorem computation_equal (M : Algebra α) (ops : Operations α) (h : MulCommutes M)
    (r : Request) (c : OutputTensor) (trials : Nat) (scratch : State α) (memory : Memory α) :
    computation M ops r c trials scratch memory=
      candidateComputation M ops r c trials scratch memory := by
  unfold computation candidateComputation
  split_ifs
  · rfl
  · rfl
  · exact replay_equal M ops h _ _ scratch _ memory

#axiomsClean computation_equal
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.Pair
