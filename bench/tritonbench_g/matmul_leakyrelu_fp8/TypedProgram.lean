import bench.tritonbench_g.matmul_leakyrelu_fp8.ExecutionBody

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.Execution
open VeriTile.Triton FP.Structural
open MatmulLeakyreluFp8
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

def accumulation {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Problem) (b : Buffers) (s : State α) : CIndex → α :=
  foldRange (update M D p b s) 0 (trips p) 1 (fun _ => M.literal (some .fp32) .real 0)

theorem loopExecution {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Problem) (b : Buffers) (initial current : State α)
    (ctx : Context p initial current)
    (ha : HasAcc (fun _ => M.literal (some .fp32) .real 0) current) :
    ∃ t, FP.Structural.range (backend M D) "k" 0 (trips p) 1 (body p b) current = some t ∧
      Context p initial t ∧ HasAcc (accumulation M D p b initial) t := by
  let I := fun (_ : Nat) (a : CIndex → α) (s : State α) => Context p initial s ∧ HasAcc a s
  exact range_values (backend M D) "k" (body p b) (update M D p b initial)
    I (fun a s => Context p initial s ∧ HasAcc a s) 0 (trips p) 1 _ current
    (by
      intro k a s h hk
      let t := bodyExecution M D p b initial s k a h.1 h.2
      exact ⟨t.val,t.property,bodyContext M D p b initial s k a h.1 h.2⟩)
    (by intro n a s h; exact h) ⟨ctx,ha⟩

def activation {α : Type} (M : Algebra α) (D : DotAcc α) (enabled : Bool) (x : α) : α :=
  if enabled then
    if D.le (some .fp32) .real (M.literal (some .fp32) .real 0.0) x then x
    else M.binary (some .fp32) .real .mul (M.literal (some .fp32) .real 0.01) x
  else x

def outputValue {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Problem) (b : Buffers) (s : State α) (i : CIndex) : α :=
  M.cast none .real .fp16 (activation M D p.enabled (accumulation M D p b s i))

def active {α : Type} (p : Problem) (s : State α) (i : CIndex) : Prop :=
  row p s i.1 < (p.M : Int) ∧ col p s i.2.1 < (p.N : Int)
instance {α : Type} (p : Problem) (s : State α) : DecidablePred (active p s) :=
  fun _ => inferInstanceAs (Decidable (_ ∧ _))

def cOffset {α : Type} (p : Problem) (b : Buffers) (s : State α) (i : CIndex) : Nat :=
  ((b.C.rowStride : Int)*(if row p s i.1 < (p.M : Int) then row p s i.1 else 0) +
    (b.C.columnStride : Int)*(if col p s i.2.1 < (p.N : Int) then col p s i.2.1 else 0)).toNat

def epilogueExecution {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Problem) (b : Buffers) (initial current : State α)
    (ctx : Context p initial current) (ha : HasAcc (accumulation M D p b initial) current) :
    {t : State α // run (backend M D) (epilogue p b) current = some t} := by
  simp only [HasAcc] at ha
  cases he : p.enabled <;>
    (refine ⟨?_, ?_⟩
     swap
     simp [epilogue,source,TypedSource.kernel,ComputeKernel.surfaceBody,run,step,evalExpr,
       evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericLt,numericEq,numericLe,
       ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,
       FloatDType.toTileDType,Region.cast,Option.bind,integralFloorDiv,integralMod,
       ctx.memory,ctx.pm,ctx.pn,ctx.lm,ctx.ln,ha,backend,he]
     rfl)

/-- Keep the fixed tile's lane list abstract during readback elaboration. -/
@[irreducible] def outputIndices : List CIndex := TileShape.allIndices [128,256]

theorem masked_readback_list {α ι : Type} (r : RegionName) (off : ι → Nat) (val : ι → Cell α)
    (active : ι → Prop) [DecidablePred active] (l : List ι) (s : State α) (i : ι)
    (hnd : l.Nodup) (himem : i ∈ l) (hi : active i)
    (hinj : ∀ a b, active a → active b → off a = off b → a = b) :
    (l.foldl (fun t k => if active k then t.write r (off k) (val k) else t) s).mem r (off i) = val i := by
  rw [State.masked_scatter_eq_filter]
  have hm : i ∈ l.filter (fun j => decide (active j)) := by simp [himem,hi]
  obtain ⟨before,after,hl⟩ := List.append_of_mem hm
  have hd := hnd.filter (fun j => decide (active j))
  rw [hl,List.nodup_append,List.nodup_cons] at hd
  obtain ⟨_,⟨hnot,_⟩,_⟩ := hd
  rw [hl,List.foldl_append,List.foldl_cons,State.scatter_frame]
  · exact State.write_same _ _ _ _
  · right
    intro j hj heq
    have hjmem : j ∈ l.filter (fun k => decide (active k)) := by
      rw [hl]
      simp only [List.mem_append,List.mem_cons]
      exact Or.inr (Or.inr hj)
    have hjactive : active j := by simpa using (List.mem_filter.mp hjmem).2
    exact hnot ((hinj i j hi hjactive heq) ▸ hj)

def written {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Problem) (b : Buffers) (initial : State α) : State α :=
  outputIndices.foldl (fun t i => if active p initial i then
    t.write "C" (cOffset p b initial i) (.mk .fp16 (outputValue M D p b initial i)) else t) initial

theorem epilogueMemory {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Problem) (b : Buffers) (initial current : State α)
    (ctx : Context p initial current) (ha : HasAcc (accumulation M D p b initial) current) :
    (epilogueExecution M D p b initial current ctx ha).val.mem = (written M D p b initial).mem := by
  cases p with
  | mk pm pn pk enabled profile =>
    cases enabled <;> simp [epilogueExecution,written,outputIndices,masked_scatter_memory,ctx.memory,
      active,cOffset,row,col,outputValue,activation,backend,FP.Structural.bop,
      Broadcast.leftIndex,Broadcast.rightIndex]
    all_goals rfl

def Post {α : Type} (p : Problem) (b : Buffers) (initial final : State α) (v : CIndex → α) : Prop :=
  (∀ i, active p initial i → final.mem "C" (cOffset p b initial i) = Cell.mk .fp16 (v i)) ∧
  (∀ (r : RegionName) o, (r ≠ "C" ∨ ∀ i, active p initial i → o ≠ cOffset p b initial i) →
    final.mem r o = initial.mem r o)

/-- Direct complete interpretation of the readable typed quotation. Geometry
injection is an internal lemma premise, to be discharged for the legal grid. -/
theorem programExecution {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Problem) (b : Buffers) (initial : State α)
    (hw : ∀ i j, active p initial i → active p initial j → cOffset p b initial i = cOffset p b initial j → i = j) :
    ∃ final, FP.Structural.exec (backend M D) (source p b) initial = some final ∧
      final.mem = (written M D p b initial).mem ∧ Post p b initial final (outputValue M D p b initial) := by
  let start := preludeExecution M D p b initial
  have context := preludeContext M D p b initial
  obtain ⟨mid,hm,hc,ha⟩ := loopExecution M D p b initial start.val context.1 context.2
  let finish := epilogueExecution M D p b initial mid hc ha
  refine ⟨finish.val,?_,epilogueMemory M D p b initial mid hc ha,?_,?_⟩
  · change run (backend M D) (source p b).surfaceBody initial = some finish.val
    rw [source_decomposition,run_append,run_append,start.property]
    simp only [Option.bind_some]
    have kt : start.val.regs .nat [] "k_tiles" = some (fun _ => trips p) := context.1.kt
    have hr : run (backend M D)
        [.forRangeDyn "k" (.constNat 0) (.ref .nat [] "k_tiles") (.constNat 1) (body p b)] start.val = some mid := by
      simpa [run,step,evalOp_unfold,kt] using hm
    rw [hr]
    exact finish.property
  · intro i hi
    rw [epilogueMemory]
    unfold written
    exact masked_readback_list "C" (cOffset p b initial)
      (fun j => Cell.mk .fp16 (outputValue M D p b initial j)) (active p initial) outputIndices initial i
      (by rw [outputIndices]; exact TileShape.allIndices_nodup _)
      (by rw [outputIndices]; exact TileShape.mem_allIndices _ _) hi hw
  · intro r o hh
    rw [epilogueMemory]
    unfold written
    apply State.masked_scatter_frame "C" (cOffset p b initial)
      (fun j => Cell.mk .fp16 (outputValue M D p b initial j)) (active p initial)
      outputIndices r o _ initial
    rcases hh with hr | ho
    · exact Or.inl hr
    · exact Or.inr (fun i _ hi => ho i hi)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.Execution
