import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.NativeFinal

namespace VeriTile.Bench.Repairs.BmmChunkFwd.NativeGraph
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open FPSuccess
noncomputable section
set_option linter.unusedSimpArgs false

theorem complete_materialized (rest : Algebra Expr) (h : Host.Call) (c : Host.Config)
    (s : State Expr) (hm : Memory s) (ha : h.aStorage=.fp32) (hb : h.bStorage=.fp32) :
    ∃ t, FP.Structural.exec (engine (backend rest c).algebra) (Kernels.source h c) s=some t ∧ Memory t := by
  obtain ⟨initial,hpre,hbase,hmem,hactive,hcontrols⟩ := prologue (backend rest c) h c s
  have hm0 : Memory initial := by simpa only [Memory,hmem] using hm
  cases act : active h c s
  · refine ⟨initial,?_,hm0⟩
    change run _ (Kernels.source h c).surfaceBody s=some initial
    rw [source_split,run_append,hpre]
    simp only [Option.bind_some,run,step,hactive,act,Bool.false_eq_true,↓reduceIte]
    rfl
  · obtain ⟨prepared,hsetup,hready,hm1,hf1,hacc⟩ := setup_materialized rest h c initial hbase ha hb
    have hmp : Memory prepared := by simpa only [Memory,hm1] using hm0
    obtain ⟨mid,hloop,hmready,hm2,hf2,hmacc⟩ := loop_materialized rest h c prepared hready hmp hacc ha hb
    have hmm : Memory mid := by simpa only [Memory,hm2] using hmp
    obtain ⟨final,hpost,hfinal⟩ :=
      epilogue_materialized rest h c s mid hmready ((hcontrols.fixed hf1).fixed hf2) hmm hmacc ha hb
    refine ⟨final,?_,hfinal⟩
    change run _ (Kernels.source h c).surfaceBody s=some final
    rw [source_split,run_append,hpre]
    simp only [Option.bind_some,run,step,hactive,act,↓reduceIte]
    rw [branch_split,run_append,run_append,hsetup]
    simp only [Option.bind_some,run,hloop]
    simp [Option.bind,hpost]

theorem schedule_materialized (rest : Algebra Expr) (h : Host.Call) (c : Host.Config)
    (jobs : List (GridIndex (h.grid c))) (s : State Expr) (hm : Memory s)
    (ha : h.aStorage=.fp32) (hb : h.bStorage=.fp32) :
    ∃ t, FPGrid.schedule (backend rest c) h c jobs s=some t ∧ Memory t := by
  induction jobs generalizing s with
  | nil => exact ⟨s,rfl,hm⟩
  | cons idx jobs ih =>
    obtain ⟨mid,hmid,hm'⟩ := complete_materialized rest h c (FPGrid.atIndex s idx) hm ha hb
    obtain ⟨t,ht,htm⟩ := ih mid hm'
    refine ⟨t,?_,htm⟩
    simp only [FPGrid.schedule,hmid,Option.bind_some]
    exact ht

theorem host_materialized (rest : Algebra Expr) (h : Host.Call) (c : Host.Config)
    (s : State Expr) (hm : Memory s) (ha : h.aStorage=.fp32) (hb : h.bStorage=.fp32) :
    ∃ t, FPGrid.runHost (backend rest c) h c s=some t ∧ Memory t := by
  by_cases empty : h.shape.outputSize=0
  · exact ⟨s,by simp [FPGrid.runHost,empty],hm⟩
  · obtain ⟨t,ht,htm⟩ := schedule_materialized rest h c (FPGrid.allPrograms h c) s hm ha hb
    exact ⟨t,by simpa [FPGrid.runHost,empty] using ht,htm⟩

/-- No standalone-dot fallback is observable. Changing only that primitive
cannot change interpretation of a materialized graph. -/
theorem evaluate_without_pending {α : Type} (p q : Primitives α)
    (leaf : p.leaf=q.leaf) (literal : p.literal=q.literal) (cast : p.cast=q.cast)
    (binary : p.binary=q.binary) (dotAcc : p.dotAcc=q.dotAcc)
    (x : Expr) (hx : Materialized x) : evaluate p x=evaluate q x := by
  induction x with
  | leaf d w => exact congrFun (congrFun leaf d) w
  | literal precision dtype value => simp only [evaluate,literal]
  | cast precision source target x ih => simp only [evaluate,cast,ih hx]
  | binary precision dtype op x y ihx ihy => simp only [evaluate,binary,ihx hx.1,ihy hx.2]
  | pendingDot => exact False.elim hx
  | dotAcc c precision l r out opts batch m k n a b acc i iha ihb ihacc =>
      simp only [evaluate,dotAcc]
      congr 1
      · funext j; exact iha j (hx.1 j)
      · funext j; exact ihb j (hx.2.1 j)
      · exact ihacc hx.2.2

#axiomsClean host_materialized
#axiomsClean evaluate_without_pending
end
end VeriTile.Bench.Repairs.BmmChunkFwd.NativeGraph

#guard_msgs (drop info) in
#auditModuleAxioms
