import bench.repairs.tritonbench_g.matmul_triton2.KernelOnly.PairIO
import bench.repairs.tritonbench_g.matmul_triton2.KernelOnly.Proofs.MaskSafety
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.Pair
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
    ∃ final, exec (Core.backend rest P) (Core.source p) initial=some final ∧
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
        Core.rowBlock,Core.columnBlock,Core.groupSize,Core.firstRow,Core.group,Core.width,memory,pids]

theorem written_memory_congr (P : Core.Primitives α) (p : Core.Parameters)
    (s t : State α) (memory : t.mem=s.mem) (pids : t.pids=s.pids) :
    (Core.written P p t).mem=(Core.written P p s).mem := by
  simp [Core.written,masked_scatter_memory,accumulation_congr P p s t memory pids,
    Core.active,Core.cOffset,Core.row,Core.column,Core.rowBlock,Core.columnBlock,Core.groupSize,Core.firstRow,Core.group,Core.width,memory,pids]

theorem launch_transfer (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (positive : 0<p.blockInner) (pid : Nat) (s t original : State α) (memory : t.mem=s.mem)
    (ran : (IO.launch rest P p pid).run s=some original) :
    ∃ candidate, (launch rest P p pid).run t=some candidate ∧ candidate.mem=original.mem := by
  obtain ⟨old,ho,mo⟩ := original_memory rest P p ((IO.launch rest P p pid).initial s) positive
  have equal : old=original := Option.some.inj (ho.symm.trans ran)
  subst old
  obtain ⟨candidate,hc,mc⟩ := MaskHoist.exec_memory rest P p ((launch rest P p pid).initial t) positive
  refine ⟨candidate,hc,mc.trans (?_ : _=original.mem)⟩
  exact (written_memory_congr P p ((IO.launch rest P p pid).initial s)
    ((launch rest P p pid).initial t) memory rfl).trans mo.symm

theorem schedule_transfer (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (positive : 0<p.blockInner) (ids : List Nat) (s t original : State α)
    (memory : t.mem=s.mem)
    (ran : runSchedule (ids.map (IO.launch rest P p)) s=some original) :
    ∃ candidate, runSchedule (ids.map (launch rest P p)) t=some candidate ∧ candidate.mem=original.mem := by
  induction ids generalizing s t with
  | nil =>
    have eq : s=original := Option.some.inj ran
    exact ⟨t,rfl,memory.trans (congrArg State.mem eq)⟩
  | cons pid tail ih =>
    simp only [List.map_cons,runSchedule] at ran ⊢
    cases hr : (IO.launch rest P p pid).run s with
    | none => simp [hr] at ran
    | some middle =>
      obtain ⟨next,hn,mn⟩ := launch_transfer rest P p positive pid s t middle memory hr
      rw [hn]
      exact ih middle next mn (by simpa only [hr,Option.bind_some] using ran)

theorem launch_safe (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (xs : ChannelValues α (IO.inputs p)) (bounds : RegionName → Nat)
    (hb : (IO.matmul rest P p).InBounds bounds xs) (hBK : 0<p.blockInner)
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
    (xs : ChannelValues α (IO.inputs p)) (bounds : RegionName → Nat)
    (hb : (IO.matmul rest P p).InBounds bounds xs) (hBK : 0<p.blockInner)
    (order : List Nat) (s : State α) :
    ScheduleSafe bounds (order.map (launch rest P p)) s := by
  induction order generalizing s with
  | nil => trivial
  | cons pid order ih =>
    exact ⟨launch_safe rest P p xs bounds hb hBK pid s,fun final _ => ih final⟩


theorem implements_transfer (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (positive : 0<p.blockInner)
    (f : (IO.matmul rest P p).Inputs → (IO.matmul rest P p).Outputs)
    (original : (IO.matmul rest P p).Implements f) : (candidateIO rest P p).Implements f := by
  intro vals hv bounds hb initial hin
  obtain ⟨safe,final,ran,outputs,frame⟩ := original vals hv bounds hb initial hin
  obtain ⟨candidate,hc,mc⟩ := schedule_transfer rest P p positive _ initial initial final rfl ran
  refine ⟨schedule_safe rest P p vals bounds hb positive _ initial,candidate,hc,?_,?_⟩
  · simpa only [ProgramIO.OutputBinding,candidateIO,mc] using outputs
  · simpa only [ProgramIO.Frame,candidateIO,mc] using frame

/-- Original grouped grid, all logical input words and complete output scope;
actual safety, success and both raw frames are conclusions. -/
theorem equivalent (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (geometry : Geometry.Valid (Core.geometry p)) (inner : 0<p.blockInner)
    (ha : p.input≠p.output) (hb : p.weight≠p.output) :
    (IO.matmul rest P p).Equivalent (variant rest P p) := by
  intro vals _ bounds bounded initial _
  have safe : ScheduleSafe bounds ((IO.matmul rest P p).schedule vals) initial := by
    apply PublicIO.schedule_safe rest P p inner bounds
    · intro r k hr hk
      exact bounded.2.1 (0 : Fin 2) (⟨r,hr⟩,⟨k,hk⟩,PUnit.unit) trivial
    · intro k c hk hc
      exact bounded.2.1 (1 : Fin 2) (⟨k,hk⟩,⟨c,hc⟩,PUnit.unit) trivial
    · intro r c hr hc
      exact bounded.2.2.1 (0 : Fin 1) (⟨r,hr⟩,⟨c,hc⟩,PUnit.unit) trivial
  obtain ⟨final,ran,_,frame⟩ := Core.grid_correct rest P p geometry inner ha hb initial
  have ioRun : runSchedule ((IO.matmul rest P p).schedule vals) initial=some final :=
    (PublicIO.schedule_run rest P p _ initial).trans ran
  have ioFrame : (IO.matmul rest P p).Frame vals initial final := by
    intro region offset outside
    apply frame region offset
    by_cases other : region≠p.output
    · exact Or.inl other
    · right
      intro x _
      have miss := outside.1 (0 : Fin 1) (x.1,x.2,PUnit.unit) trivial
      exact miss.resolve_left other
  obtain ⟨candidate,candidateRun,memory⟩ :=
    schedule_transfer rest P p inner _ initial initial final rfl ioRun
  have candidateFrame : (IO.matmul rest P p).Frame vals initial candidate := by
    simpa only [ProgramIO.Frame,memory] using ioFrame
  refine ⟨safe,schedule_safe rest P p vals bounds bounded inner _ initial,
    final,candidate,ioRun,candidateRun,?_,ioFrame,candidateFrame⟩
  intro o j _
  change mapCell _ (final.mem _ _)=mapCell _ (candidate.mem _ _)
  rw [memory]
  rfl

#axiomsClean original_memory
#axiomsClean launch_transfer
#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.Pair
