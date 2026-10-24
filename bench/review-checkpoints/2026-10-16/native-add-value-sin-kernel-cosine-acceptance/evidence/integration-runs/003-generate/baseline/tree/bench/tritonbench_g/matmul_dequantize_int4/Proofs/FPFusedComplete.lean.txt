import bench.tritonbench_g.matmul_dequantize_int4.Proofs.FPFusedExecution
namespace VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.FPFused
open VeriTile.Triton FP.Structural FPTyped
open Int4Matmul.FPSoftware (Word)
open FPCompiled (DotAcc)
open KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false

/-- Task-local instruction loop, executing every original iteration. The loop
count used by `exec` below is read from the actual source prelude's register. -/
def loop (M : Algebra Word) (D : DotAcc Word) (p : Parameters)
    (k : Nat) : Nat → State Word → Option (State Word)
  | 0, s => some s
  | n+1, s => do
    let next ← runBody M D p k s
    loop M D p (k+1) n next

theorem loopExecution (M : Algebra Word) (D : DotAcc Word)
    (p : Parameters) (initial current : State Word) (k n : Nat)
    (h : Invariant M D p initial k current) :
    ∃ final, loop M D p k n current=some final ∧
      Invariant M D p initial (k+n) final := by
  induction n generalizing k current with
  | zero => exact ⟨current,rfl,by simpa using h⟩
  | succ n ih =>
    let next := bodyExecution M D p initial current k h
    obtain ⟨final,he,hi⟩ := ih next.val (k+1) (bodyInvariant M D p initial current k h)
    refine ⟨final,?_,?_⟩
    · simp only [loop,next.property,Option.bind_some]
      exact he
    · simpa only [Nat.add_assoc,Nat.add_comm 1 n] using hi

/-- Compiled fused graph: actual shared-DSL prelude and epilogue, with one
explicit scalar-FMA instruction in every iteration. This is not claimed to be
the unfused source evaluator, nor the shared DSL's unsupported FMA node. -/
def exec (M : Algebra Word) (D : DotAcc Word) (p : Parameters)
    (initial : State Word) : Option (State Word) := do
  let start ← run (FPCompiled.backend M D) (FPCompiled.prelude p) initial
  let count ← start.regs .nat [] "num_pid_k"
  let mid ← loop M D p 0 (count PUnit.unit) start
  run (FPCompiled.backend M D) (FPCompiled.epilogue p) mid

def epilogueExecution (M : Algebra Word) (D : DotAcc Word)
    (p : Parameters) (initial current : State Word)
    (h : Invariant M D p initial (steps p) current) :
    {t : State Word // run (FPCompiled.backend M D) (FPCompiled.epilogue p) current = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [FPCompiled.epilogue, FPCompiled.source, FPCompiled.typed, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric,
    FP.Structural.bop, natLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
    FloatDType.numericDType, FloatDType.toTileDType,
    Region.cast, Option.bind, integralFloorDiv, integralMod,
    h.pidm, h.pidn, h.acc, h.memory, FPCompiled.backend]
  rfl

def written (M : Algebra Word) (D : DotAcc Word)
    (p : Parameters) (initial : State Word) : State Word :=
  (TileShape.allIndices [p.blockRows,p.blockColumns]).foldl
    (fun t i => if active p initial i then
      t.write p.output (outputOffset p initial i) (.mk .fp16 (outputValue M D p initial i)) else t) initial

theorem epilogueMemory (M : Algebra Word) (D : DotAcc Word)
    (p : Parameters) (initial current : State Word)
    (h : Invariant M D p initial (steps p) current) :
    (epilogueExecution M D p initial current h).val.mem = (written M D p initial).mem := by
  simp +contextual [epilogueExecution, written, masked_scatter_memory, h.memory,
    active, outputOffset, outputValue, FPCompiled.backend,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    and_assoc, and_comm, and_left_comm]

theorem correctness_memory (M : Algebra Word) (D : DotAcc Word)
    (p : Parameters) (initial : State Word) :
    ∃ final, exec M D p initial = some final ∧
      final.mem = (written M D p initial).mem := by
  let start := preludeExecution M D p initial
  obtain ⟨mid,hm,hi⟩ := loopExecution M D p initial start.val 0 (steps p)
    (preludeInvariant M D p initial)
  have hi' : Invariant M D p initial (steps p) mid := by simpa using hi
  let finish := epilogueExecution M D p initial mid hi'
  refine ⟨finish.val,?_,epilogueMemory M D p initial mid hi'⟩
  unfold exec
  rw [start.property]
  simp only [bind,Option.bind]
  rw [(preludeInvariant M D p initial).numk]
  simp only [bind,Option.bind]
  rw [hm]
  simp only [bind,Option.bind]
  exact finish.property

theorem correctness (M : Algebra Word) (D : DotAcc Word)
    (p : Parameters) (initial : State Word)
    (hw : ∀ i j, active p initial i → active p initial j →
      outputOffset p initial i = outputOffset p initial j → i = j) :
    ∃ final, exec M D p initial = some final ∧
      FPCompiled.Post p initial final (outputValue M D p initial) := by
  obtain ⟨final,he,hm⟩ := correctness_memory M D p initial
  refine ⟨final,he,?_,?_⟩
  · intro i hi
    rw [hm]
    exact masked_readback_active _ _ _ _ _ i hi hw
  · intro r o hframe
    rw [hm]
    exact State.masked_scatter_frame _ _ _ _ _ _ _ (hframe.imp id (fun h i _ => h i)) _

#axiomsClean loopExecution
#axiomsClean epilogueExecution
#axiomsClean epilogueMemory
#axiomsClean correctness_memory
#axiomsClean correctness
end
end VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.FPFused
