import bench.tritonbench_g.int4_matmul.Proofs.Kernel.SourceProof

namespace VeriTile.Bench.Repairs.Int4Matmul.KernelExecution
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false

theorem loopExecution [Inhabited α] (o : Ops α) (rest : Algebra α)
    (p : Parameters) (initial current : RegisterState α) (k : Nat) (hk : k≤steps p)
    (h : Invariant o p initial k current) :
    ∃ final,range o rest "k" k (steps p) 1 (body p) current=some final ∧
      Invariant o p initial (steps p) final := by
  have aux : ∀ n k (current : RegisterState α),steps p-k=n → k≤steps p →
      Invariant o p initial k current →
      ∃ final,range o rest "k" k (steps p) 1 (body p) current=some final ∧
        Invariant o p initial (steps p) final := by
    intro n
    induction n using Nat.strong_induction_on with
    | h n ih =>
      intro k current hn hk h
      by_cases hlt : k<steps p
      · let next := bodyExecution o rest p initial current k h
        have hi := bodyInvariant o rest p initial current k h
        obtain ⟨final,he,hf⟩ := ih (steps p-(k+1)) (by omega)
          (k+1) next.val rfl (by omega) hi
        refine ⟨final,?_,hf⟩
        rw [range,if_neg (by decide),if_pos hlt,next.property]
        exact he
      · have heq : k=steps p := by omega
        subst k
        refine ⟨current,?_,h⟩
        rw [range,if_neg (by decide),if_neg (by omega)]
  exact aux _ _ _ rfl hk h

def epilogueExecution [Inhabited α] (o : Ops α) (rest : Algebra α)
    (p : Parameters) (initial current : RegisterState α)
    (h : Invariant o p initial (steps p) current) :
    {t : RegisterState α // run o rest (epilogue p) current=some t} := by
  by_cases hs : p.splits=1
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [epilogue,kernel,ComputeKernel.surfaceBody,
      run,step,localStep,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,FP.Structural.store,
      FP.Structural.atomicAdd,atomicAddition,numeric,
      FP.Structural.bop,natLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
      FloatDType.numericDType,FloatDType.toTileDType,
      Region.cast,Option.bind,integralFloorDiv,integralMod,
      h.pidm,h.pidn,h.acc,h.memory,algebra,hs]
    rfl

/-- One masked source lane. Split-K atomics read the current destination at
the actual commit point, after preceding lanes/programs. -/
def writeLane [Inhabited α] (o : Ops α) (p : Parameters)
    (initial : RegisterState α) (values : CIndex p → α) (current : RegisterState α) (i : CIndex p) :=
  if active p initial i then current.write p.output (outputOffset p initial i)
    (.mk .real (if p.splits=1 then values i else
      o.atomicAdd ((current.mem p.output (outputOffset p initial i)).read .real) (values i)))
  else current

def written [Inhabited α] (o : Ops α) (p : Parameters) (initial : RegisterState α) :=
  (TileShape.allIndices [p.blockRows,p.blockColumns]).foldl
    (writeLane o p initial (accumulation o p initial (steps p))) initial

theorem writeLane_memory [Inhabited α] (o : Ops α) (p : Parameters)
    (initial : RegisterState α) (values : CIndex p → α) (a b : RegisterState α)
    (hm : a.mem=b.mem) (i : CIndex p) :
    (writeLane o p initial values a i).mem=(writeLane o p initial values b i).mem := by
  simp only [writeLane]
  split_ifs <;> simp only [State.write,hm]

theorem fold_memory [Inhabited α] (o : Ops α) (p : Parameters)
    (initial : RegisterState α) (values : CIndex p → α) (a b : RegisterState α)
    (hm : a.mem=b.mem) (lanes : List (CIndex p)) :
    (lanes.foldl (writeLane o p initial values) a).mem=
      (lanes.foldl (writeLane o p initial values) b).mem := by
  induction lanes generalizing a b with
  | nil => exact hm
  | cons i lanes ih => exact ih _ _ (writeLane_memory o p initial values a b hm i)

private theorem atomic_scatter_memory [Inhabited α] (xs : List ι)
    (live : ι → Prop) [DecidablePred live] (r : RegionName) (off : ι → Nat)
    (value : ι → α) (combine : α → α → α) (s : RegisterState α) :
    (xs.foldl (fun t i => if live i then t.write r (off i)
      (.mk .real (combine ((t.mem r (off i)).read .real) (value i))) else t) s).mem=
    xs.foldl (fun mem i => if live i then fun q a =>
      if q=r ∧ a=off i then .mk .real (combine ((mem r (off i)).read .real) (value i))
      else mem q a else mem) s.mem := by
  induction xs generalizing s with
  | nil => rfl
  | cons i xs ih =>
    simp only [List.foldl_cons,ih]
    by_cases hi : live i <;> simp only [hi,↓reduceIte] <;> rfl

theorem epilogueMemory [Inhabited α] (o : Ops α) (rest : Algebra α)
    (p : Parameters) (initial current : RegisterState α)
    (h : Invariant o p initial (steps p) current) :
    (epilogueExecution o rest p initial current h).val.mem=(written o p initial).mem := by
  unfold written writeLane
  by_cases hs : p.splits=1
  all_goals
    simp +contextual [epilogueExecution,written,writeLane,
      active,outputOffset,algebra,hs,
      FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
      and_assoc,and_comm,and_left_comm,
      VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy.masked_scatter_memory,
      atomic_scatter_memory,h.memory]

/-- Complete quoted-source execution, including every K tail and both output
branches. No injectivity premise is needed for this whole-memory result. -/
theorem correctness_memory [Inhabited α] (o : Ops α) (rest : Algebra α)
    (p : Parameters) (initial : RegisterState α) :
    ∃ final,exec o rest p initial=some final ∧ final.mem=(written o p initial).mem := by
  let start := preludeExecution o rest p initial
  obtain ⟨mid,hm,hi⟩ := loopExecution o rest p initial start.val 0 (by omega)
    (preludeInvariant o rest p initial)
  let finish := epilogueExecution o rest p initial mid hi
  refine ⟨finish.val,?_,epilogueMemory o rest p initial mid hi⟩
  change run o rest (kernel p).surfaceBody initial=some finish.val
  rw [source_decomposition,run_append,run_append,start.property]
  have hsingle (s : RegisterState α) :
      run o rest [.forRange "k" 0 (steps p) 1 (body p)] s=
        range o rest "k" 0 (steps p) 1 (body p) s := by simp [run,step]
  simp only [Option.bind_some,hsingle,hm]
  exact finish.property

#axiomsClean loopExecution
#axiomsClean epilogueExecution
#axiomsClean epilogueMemory
#axiomsClean correctness_memory
end
end VeriTile.Bench.Repairs.Int4Matmul.KernelExecution
