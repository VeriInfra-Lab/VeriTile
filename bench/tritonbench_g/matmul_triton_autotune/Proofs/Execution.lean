import bench.tritonbench_g.matmul_triton_autotune.Proofs.Source
import bench.tritonbench_g.matmul_triton_autotune.Proofs.Addresses
import bench.tritonbench_g.matmul_triton_autotune.Proofs.SourcePair

/-! Reuse the repaired source's checked execution. The candidate's loop and
writeback are literally the same, and the complete prefix is connected by the
integer commutation proof in SourcePair. -/
namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Source
open VeriTile.Triton FP.Structural FPValue
open ParallelAttention.FPStages (foldRange)
open KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

abbrev Context {α : Type} (p : Parameters) (initial current : State α) :=
  FPValue.Context p initial 0 current

def reindexContext {α : Type} {p : Parameters} {initial current : State α} {k l : Nat}
    (ctx : FPValue.Context p initial k current) : FPValue.Context p initial l current :=
  ⟨ctx.memory,ctx.pm,ctx.pn,ctx.offk,ctx.am,ctx.bn⟩

def preludeExecution {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (s : State α) :
    {t : State α // run (backend M D) (prelude p) s=some t} :=
  ⟨(FPValue.preludeExecution M D p .single s).val,
    (SourcePair.prelude_execution_equal (backend M D) p s).symm.trans
      (FPValue.preludeExecution M D p .single s).property⟩

theorem preludeContext {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (s : State α) (geo : FPValue.Geometry p s) :
    Context p s (preludeExecution M D p s).val ∧
      HasAcc p (fun _=>M.literal (some .fp32) .real 0) (preludeExecution M D p s).val :=
  FPValue.preludeContext M D p .single s geo

def bodyExecution {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (initial current : State α) (k : Nat) (a : CIndex p→α)
    (ctx : Context p initial current) (ha : HasAcc p a current) :
    {t : State α // run (backend M D) (body p) (current.setReg "k" .int [] (fun _=>(k : Int)))=some t} :=
  FPValue.bodyExecution M D p .single initial current k a (reindexContext ctx) ha

theorem bodyContext {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (initial current : State α) (k : Nat) (a : CIndex p→α)
    (ctx : Context p initial current) (ha : HasAcc p a current) :
    let t:=(bodyExecution M D p initial current k a ctx ha).val
    Context p initial t ∧ HasAcc p (update M D p .single initial k a) t := by
  have h:=FPValue.bodyContext M D p .single initial current k a (reindexContext ctx) ha
  exact ⟨reindexContext h.1,h.2⟩

theorem loopExecution {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (initial current : State α) (ctx : Context p initial current)
    (ha : HasAcc p (fun _=>M.literal (some .fp32) .real 0) current) :
    ∃ t,rangeInt (backend M D) "k" 0 (p.numKBlocks : Int) 1 (body p) current=some t ∧
      Context p initial t ∧ HasAcc p (accumulation M D p .single initial) t := by
  obtain ⟨t,ht,k,hc,hacc⟩:=FPValue.loopExecution M D p .single initial current ctx ha
  exact ⟨t,ht,reindexContext hc,hacc⟩

def epilogueExecution {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (initial current : State α) (ctx : Context p initial current)
    (ha : HasAcc p (accumulation M D p .single initial) current) :
    {t : State α // run (backend M D) (epilogue p) current=some t} :=
  FPValue.epilogueExecution M D p .single initial current 0 ctx ha

def cSafeOffset {α : Type} (p : Parameters) (initial : State α) (i : CIndex p) :=
  p.stride_cm*(if row p initial i.1<p.M then row p initial i.1 else 0) +
  p.stride_cn*(if col p initial i.2.1<p.N then col p initial i.2.1 else 0)

def writtenSafe {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (initial : State α) : State α :=
  (TileShape.allIndices [BM p,BN p]).foldl (fun t i => if active p initial i then
    t.write p.C (cSafeOffset p initial i) (stored .single (outputValue M D p .single initial i)) else t) initial

theorem writtenSafe_eq {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (initial : State α) : writtenSafe M D p initial=written M D p .single initial := by
  have stepEq (t : State α) (i : CIndex p) :
      (if active p initial i then t.write p.C (cSafeOffset p initial i)
        (stored .single (outputValue M D p .single initial i)) else t) =
      (if active p initial i then t.write p.C (cOffset p initial i)
        (stored .single (outputValue M D p .single initial i)) else t) := by
    by_cases h : active p initial i
    · simp [h,cSafeOffset,cOffset,h.1,h.2]
    · simp [h]
  unfold writtenSafe written
  simp_rw [stepEq]


theorem epilogueMemory {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (initial current : State α) (ctx : Context p initial current)
    (ha : HasAcc p (accumulation M D p .single initial) current) :
    (epilogueExecution M D p initial current ctx ha).val.mem=(writtenSafe M D p initial).mem := by
  rw [writtenSafe_eq]
  exact FPValue.epilogueMemory M D p .single initial current 0 ctx ha

/-- On the existing launch geometry, the complete candidate writes the
independent live-carry formula and preserves every other memory cell. -/
theorem execution_memory {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (initial : State α) (geo : FPValue.Geometry p initial) :
    ∃ final,FP.Structural.exec (backend M D) (kernel p) initial=some final ∧
      final.mem=(written M D p .single initial).mem := by
  let start:=FPValue.preludeExecution M D p .single initial
  obtain ⟨mid,hm,k,hc,ha⟩:=FPValue.loopExecution M D p .single initial start.val
    (FPValue.preludeContext M D p .single initial geo).1
    (FPValue.preludeContext M D p .single initial geo).2
  let finish:=FPValue.epilogueExecution M D p .single initial mid k hc ha
  refine ⟨finish.val,?_,FPValue.epilogueMemory M D p .single initial mid k hc ha⟩
  rw [←SourcePair.execution_equal]
  change run (backend M D) (FPValue.source p .single).surfaceBody initial=some finish.val
  rw [FPValue.source_decomposition,run_append,run_append,start.property]
  simp only [Option.bind_some]
  have hr : run (backend M D) [FPValue.loopStatement p .single] start.val=some mid := by
    rw [FPValue.loop_run,FPValue.loopStop_value]
    exact hm
  rw [hr]
  exact finish.property

#axiomsClean execution_memory
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Source
