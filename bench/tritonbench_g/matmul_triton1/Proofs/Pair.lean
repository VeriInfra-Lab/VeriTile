import bench.tritonbench_g.matmul_triton1.Proofs.MaskSafety
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.MatmulTriton1.Pair
open VeriTile.Triton FP.Structural Revision001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- The unchanged source has the same concrete recurrence and scatter as the candidate. -/
theorem original_memory (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (initial : State α) (positive : 0<p.blockInner) :
    ∃ final, exec (Core.backend rest P) (Kernel.source p) initial=some final ∧
      final.mem=(Core.written P p initial).mem := by
  let start := Core.preludeExecution rest P p initial
  obtain ⟨middle,hm,hi⟩ := Core.loopExecution rest P p initial start.val 0 (by omega)
    (Core.preludeInvariant rest P p initial)
  let finish := Core.epilogueExecution rest P p initial middle hi
  refine ⟨finish.val,?_,Core.epilogueMemory rest P p initial middle hi⟩
  change run _ (Core.source p).surfaceBody initial=some finish.val
  rw [Core.source_decomposition,run_append,run_append,start.property]
  have hs (s : State α) : run (Core.backend rest P) [Core.loopStmt p] s =
      FP.Structural.range (Core.backend rest P) "k" 0 (Core.steps p) 1 (Core.body p) s := by
    simp [run,Core.dynamicLoop rest P p s positive]
  simp only [Option.bind_some,hs,hm]
  exact finish.property

theorem accumulation_congr (P : Core.Primitives α) (p : Core.Parameters)
    (s t : State α) (memory : t.mem=s.mem) (pids : t.pids=s.pids) (n : Nat) :
    Core.accumulation P p t n=Core.accumulation P p s n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [Core.accumulation,ih]
    congr 1 <;> funext i <;>
      simp [Core.aValues,Core.bValues,Core.aOffset,Core.bOffset,Core.row,Core.column,
        Core.rowBlock,Core.columnBlock,memory,pids]

theorem written_memory_congr (P : Core.Primitives α) (p : Core.Parameters)
    (s t : State α) (memory : t.mem=s.mem) (pids : t.pids=s.pids) :
    (Core.written P p t).mem=(Core.written P p s).mem := by
  simp [Core.written,masked_scatter_memory,accumulation_congr P p s t memory pids,
    Core.active,Core.cOffset,Core.row,Core.column,Core.rowBlock,Core.columnBlock,memory,pids]

theorem launch_transfer (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (positive : 0<p.blockInner) (pid : Nat) (s t original : State α) (memory : t.mem=s.mem)
    (ran : (MatmulIO.launch rest P p pid).run s=some original) :
    ∃ candidate, (launch rest P p pid).run t=some candidate ∧ candidate.mem=original.mem := by
  obtain ⟨old,ho,mo⟩ := original_memory rest P p ((MatmulIO.launch rest P p pid).initial s) positive
  have equal : old=original := Option.some.inj (ho.symm.trans ran)
  subst old
  obtain ⟨candidate,hc,mc⟩ := MaskHoist.exec_memory rest P p ((launch rest P p pid).initial t) positive
  refine ⟨candidate,hc,mc.trans (?_ : _=original.mem)⟩
  exact (written_memory_congr P p ((MatmulIO.launch rest P p pid).initial s)
    ((launch rest P p pid).initial t) memory rfl).trans mo.symm

theorem schedule_transfer (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (positive : 0<p.blockInner) (ids : List Nat) (s t original : State α)
    (memory : t.mem=s.mem)
    (ran : runSchedule (ids.map (MatmulIO.launch rest P p)) s=some original) :
    ∃ candidate, runSchedule (ids.map (launch rest P p)) t=some candidate ∧ candidate.mem=original.mem := by
  induction ids generalizing s t with
  | nil =>
    have eq : s=original := Option.some.inj ran
    exact ⟨t,rfl,memory.trans (congrArg State.mem eq)⟩
  | cons pid tail ih =>
    simp only [List.map_cons,runSchedule] at ran ⊢
    cases hr : (MatmulIO.launch rest P p pid).run s with
    | none => simp [hr] at ran
    | some middle =>
      obtain ⟨next,hn,mn⟩ := launch_transfer rest P p positive pid s t middle memory hr
      rw [hn]
      exact ih middle next mn (by simpa only [hr,Option.bind_some] using ran)

theorem launch_safe (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (xs : ChannelValues α (MatmulIO.inputs p)) (bounds : RegionName → Nat)
    (hb : (MatmulIO.io rest P p).InBounds bounds xs) (hBK : 0<p.blockInner)
    (pid : Nat) (s : State α) : (launch rest P p pid).Safe bounds s := by
  apply MaskHoist.source_safe rest P p bounds _ hBK
  · intro q i hr hk
    have hk' : q*p.blockInner+i.2.1.val<p.innerSize := by
      change i.2.1.val<p.innerSize-q*p.blockInner at hk
      omega
    have bound := hb.2.1 (0 : Fin 2)
      (⟨Core.row p ((launch rest P p pid).initial s) i.1,hr⟩,⟨_,hk'⟩,()) trivial
    change Core.row p ((launch rest P p pid).initial s) i.1*p.inputRow+
      (q*p.blockInner+i.2.1.val)*p.inputInner<bounds p.input at bound
    convert bound using 1
    unfold Core.aOffset
    ring
  · intro q i hk hc
    have hk' : q*p.blockInner+i.1.val<p.innerSize := by
      change i.1.val<p.innerSize-q*p.blockInner at hk
      omega
    have bound := hb.2.1 (1 : Fin 2)
      (⟨_,hk'⟩,⟨Core.column p ((launch rest P p pid).initial s) i.2.1,hc⟩,()) trivial
    change (q*p.blockInner+i.1.val)*p.weightInner+
      Core.column p ((launch rest P p pid).initial s) i.2.1*p.weightColumn<bounds p.weight at bound
    convert bound using 1
    unfold Core.bOffset
    ring
  · intro i h
    exact hb.2.2.1 (0 : Fin 1)
      (⟨Core.row p ((launch rest P p pid).initial s) i.1,h.1⟩,
        ⟨Core.column p ((launch rest P p pid).initial s) i.2.1,h.2⟩,()) trivial

theorem schedule_safe (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (xs : ChannelValues α (MatmulIO.inputs p)) (bounds : RegionName → Nat)
    (hb : (MatmulIO.io rest P p).InBounds bounds xs) (hBK : 0<p.blockInner)
    (order : List Nat) (s : State α) :
    ScheduleSafe bounds (order.map (launch rest P p)) s := by
  induction order generalizing s with
  | nil => trivial
  | cons pid order ih =>
    exact ⟨launch_safe rest P p xs bounds hb hBK pid s,fun final _ => ih final⟩


theorem implements_transfer (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (positive : 0<p.blockInner)
    (f : (MatmulIO.io rest P p).Inputs → (MatmulIO.io rest P p).Outputs)
    (original : (MatmulIO.io rest P p).Implements f) : (candidateIO rest P p).Implements f := by
  intro vals hv bounds hb initial hin
  obtain ⟨safe,final,ran,outputs,frame⟩ := original vals hv bounds hb initial hin
  obtain ⟨candidate,hc,mc⟩ := schedule_transfer rest P p positive _ initial initial final rfl ran
  refine ⟨schedule_safe rest P p vals bounds hb positive _ initial,candidate,hc,?_,?_⟩
  · simpa only [ProgramIO.OutputBinding,candidateIO,mc] using outputs
  · simpa only [ProgramIO.Frame,candidateIO,mc] using frame

/-- Same complete original domain, both successful safe actual grids,
all typed output cells and both exact exterior raw memory frames. -/
theorem equivalent (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (canonical : FiniteSource.Canonical p) (ready : FiniteIndex.Ready p)
    (storage : FiniteIndex.StorageBounds p) (geometry : Geometry.Valid (Core.geometry p))
    (ha : p.input≠p.output) (hb : p.weight≠p.output) :
    (MatmulIO.io rest P p).Equivalent (variant rest P p) := by
  have original := IOCorrect.precision rest P p canonical ready storage geometry ha hb
  exact ProgramIO.EquivalentOn.of_implementsOn original
    (implements_transfer rest P p (by have := ready.blockInner.1; omega) _ original)

#axiomsClean original_memory
#axiomsClean launch_transfer
#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.MatmulTriton1.Pair
