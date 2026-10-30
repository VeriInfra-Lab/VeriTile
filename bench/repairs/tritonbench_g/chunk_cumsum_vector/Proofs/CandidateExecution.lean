import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.CandidateStep
import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.FPExecution

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.CandidateExecution
open VeriTile.Triton FP.Structural IndexProgram CandidateProgram FPValues
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Reference recurrence over immutable input cells and padding; it does not
inspect registers, interpreter results, or the final output. -/
def carry (M : Algebra α) (h : Call) (c : Config) (feature row : Nat) (initial : State α) : Nat → Values α .real [BS]
  | 0 => fun _ => M.literal (some .fp32) .real 0
  | q+1 => CandidateValues.updated M c (loaded M h c feature row initial q) (carry M h c feature row initial q)
def expected (M : Algebra α) (dot : DotBackend α) (h : Call) (c : Config) (positive : 0<c.BT)
    (p : Geometry.Program h) (initial : State α) (time : Nat) (j : Fin BS) : α :=
  output M dot c (loaded M h c p.feature p.row initial (time/c.BT))
    (carry M h c p.feature p.row initial (time/c.BT))
    (⟨time%c.BT,Nat.mod_lt _ positive⟩,j,PUnit.unit)

structure Holds (M : Algebra α) (dot : DotBackend α) (initial : State α) (h : Call) (c : Config)
    (p : Geometry.Program h) (q : Nat) (u : State α) : Prop where
  count : q≤Geometry.chunks h c
  batch : u.regs .nat [] "i_b"=some (fun _ => p.row/h.H)
  head : u.regs .nat [] "i_h"=some (fun _ => p.row%h.H)
  mask : u.regs .bool [BS] "active_s"=some (axisMask h.S BS (p.feature*BS))
  columns : u.regs .nat [BS] "columns"=some (coordinate h.S BS (p.feature*BS))
  triangular : u.regs .real [c.BT,c.BT] "m_s"=some (triangular M c)
  carryReg : u.regs .real [BS] "b_z"=some (carry M h c p.feature p.row initial q)
  reads : ∀ off, u.mem h.input off=initial.mem h.input off
  undef : u.undef=initial.undef
  output : ∀ k, k<q → ∀ (i : Fin c.BT) (j : Fin BS), Geometry.active h c p.feature k i.val j.val →
    u.mem h.output (h.outputStrides.address h.H p.row (k*c.BT+i.val) (p.feature*BS+j.val))=
      FPMemory.stored M h (FPValues.output M dot c (loaded M h c p.feature p.row initial k)
        (carry M h c p.feature p.row initial k) (i,j,PUnit.unit))
  frame : FPMemory.Frame h p initial u

theorem iteration (M : Algebra α) (dot : DotBackend α) (initial : State α)
    (h : Call) (layout : Geometry.OutputLayout h) (c : Config) (positive : 0<c.BT)
    (p : Geometry.Program h) (q : Nat) (u : State α) (hq : q<Geometry.chunks h c)
    (hr : u.regs .nat [] "i_t"=some (fun _ => q)) (inv : Holds M dot initial h c p q u) :
    ∃ v, run (machine M dot) (loopBody h c) u=some v ∧ Holds M dot initial h c p (q+1) v := by
  let acc := carry M h c p.feature p.row initial q
  let values := FPValues.output M dot c (loaded M h c p.feature p.row initial q) acc
  let pre := CandidateStep.beforeStore M dot h c p.feature p.row q initial acc u
  let v := CandidateStep.result M dot h c p.feature p.row q initial acc u
  have he := CandidateStep.executes M dot h c p.feature p.row q initial acc u hr inv.batch inv.head inv.mask inv.columns inv.triangular inv.carryReg inv.reads inv.undef
  refine ⟨v,he,Nat.succ_le_of_lt hq,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩
  · simp [v,CandidateStep.result,FPMemory.scatter,CandidateStep.beforeStore,State.masked_scatter_eq_filter,inv.batch]
  · simp [v,CandidateStep.result,FPMemory.scatter,CandidateStep.beforeStore,State.masked_scatter_eq_filter,inv.head]
  · simp [v,CandidateStep.result,FPMemory.scatter,CandidateStep.beforeStore,State.masked_scatter_eq_filter,inv.mask]
  · simp [v,CandidateStep.result,FPMemory.scatter,CandidateStep.beforeStore,State.masked_scatter_eq_filter,inv.columns]
  · simp [v,CandidateStep.result,FPMemory.scatter,CandidateStep.beforeStore,State.masked_scatter_eq_filter,inv.triangular]
  · simp [v,CandidateStep.result,carry,acc]
  · intro off
    change (FPMemory.scatter M h c p.feature p.row q values pre).mem h.input off=initial.mem h.input off
    rw [FPMemory.other_region M h c p.feature p.row q values pre h.input layout.distinct]
    simpa [pre,CandidateStep.beforeStore] using inv.reads off
  · exact (FPMemory.scatter_undef M h c p.feature p.row q values pre).trans inv.undef
  · intro k hk i j ha
    change (FPMemory.scatter M h c p.feature p.row q values pre).mem h.output _=_
    by_cases same : k=q
    · subst k
      exact FPMemory.cell M h layout c positive p q hq values pre i j ha
    · have earlier : k*c.BT+i.val<q*c.BT := by
        have kl : k+1≤q := by omega
        have mult := Nat.mul_le_mul_right c.BT kl
        have hi := i.isLt
        rw [Nat.add_mul,Nat.one_mul] at mult
        omega
      have ki : k<Geometry.chunks h c := by omega
      have bounds := (Geometry.active_iff h c p k i.val j.val (Geometry.chunk_start h c positive k ki)).mp ha
      rw [FPMemory.preserves_earlier M h layout c positive p q hq values pre _ j bounds.1 bounds.2 earlier]
      exact inv.output k (by omega) i j ha
  · have frame := FPMemory.frame M h c positive p q hq values pre
    intro r off hn
    exact (inv.frame r off hn).trans (frame r off hn)

abbrev initial := @FPExecution.initial
abbrev init_executes := @FPExecution.init_executes

theorem invariant (M : Algebra α) (dot : DotBackend α) (h : Call) (c : Config)
    (p : Geometry.Program h) (u : State α) : Holds M dot u h c p 0 (initial M h c p u) := by
  refine ⟨Nat.zero_le _,?_,?_,?_,?_,?_,?_,by intros; rfl,rfl,?_,fun _ _ _ => rfl⟩
  all_goals first | simp [initial,FPExecution.initial,carry] | (intro k hk; omega)

theorem bind_invariant (M : Algebra α) (dot : DotBackend α) (initial : State α)
    (h : Call) (c : Config) (p : Geometry.Program h) (q : Nat) (u : State α)
    (inv : Holds M dot initial h c p q u) : Holds M dot initial h c p q (u.setReg "i_t" .nat [] (fun _ => q)) := by
  refine ⟨inv.count,?_,?_,?_,?_,?_,?_,inv.reads,inv.undef,inv.output,inv.frame⟩
  all_goals first | simpa using inv.batch | simpa using inv.head | simpa using inv.mask | simpa using inv.columns | simpa using inv.triangular | simpa using inv.carryReg

theorem loop_executes (M : Algebra α) (dot : DotBackend α) (initial : State α)
    (h : Call) (layout : Geometry.OutputLayout h) (c : Config) (positive : 0<c.BT)
    (p : Geometry.Program h) (q : Nat) (u : State α) (inv : Holds M dot initial h c p q u) :
    ∃ v, FP.Structural.range (machine M dot) "i_t" q (Geometry.chunks h c) 1 (loopBody h c) u=some v ∧
      Holds M dot initial h c p (Geometry.chunks h c) v := by
  by_cases hq : q<Geometry.chunks h c
  · obtain ⟨v,he,hv⟩ := iteration M dot initial h layout c positive p q
      (u.setReg "i_t" .nat [] (fun _ => q)) hq (by simp) (bind_invariant M dot initial h c p q u inv)
    obtain ⟨w,hw,hf⟩ := loop_executes M dot initial h layout c positive p (q+1) v hv
    refine ⟨w,?_,hf⟩
    rw [FP.Structural.range]
    simp only [Nat.one_ne_zero,if_false,hq,if_true,he,bind,Option.bind_some]
    exact hw
  · have eq : q=Geometry.chunks h c := Nat.le_antisymm inv.count (Nat.le_of_not_gt hq)
    refine ⟨u,?_,by simpa only [eq] using inv⟩
    rw [FP.Structural.range]
    simp only [Nat.one_ne_zero,if_false,hq]
termination_by Geometry.chunks h c-q

theorem statement_eval (M : Algebra α) (h : Call) (c : Config) (u : State α) :
    step M (.forRangeDyn "i_t" (.constNat 0) (countOp h c) (.constNat 1) (loopBody h c)) u=
      FP.Structural.range M "i_t" 0 (Geometry.chunks h c) 1 (loopBody h c) u := by
  simp [step,countOp,FP.Structural.evalOp,numeric,bop,Broadcast.leftIndex,Broadcast.rightIndex,Geometry.chunks]

theorem source_correct (M : Algebra α) (dot : DotBackend α) (h : Call) (layout : Geometry.OutputLayout h)
    (c : Config) (positive : 0<c.BT) (p : Geometry.Program h) (u : State α)
    (hf : u.pids 0=p.feature) (hr : u.pids 1=p.row) :
    ∃ v, FP.Structural.exec (machine M dot) (Kernels.sumLeft h c) u=some v ∧
      (∀ (time : Nat) (j : Fin BS), time<h.T → p.feature*BS+j.val<h.S →
        v.mem h.output (h.outputStrides.address h.H p.row time (p.feature*BS+j.val))=
          FPMemory.stored M h (expected M dot h c positive p u time j)) ∧
      FPMemory.Frame h p u v := by
  obtain ⟨v,he,inv⟩ := loop_executes M dot u h layout c positive p 0 (initial M h c p u) (invariant M dot h c p u)
  refine ⟨v,?_,?_,inv.frame⟩
  · change run (machine M dot) (Kernels.sumLeft h c).surfaceBody u=some v
    rw [source_body,body,run_append,init_executes M dot h c p u hf hr]
    simp only [Option.bind_some,run,statement_eval,he]
    rfl
  · intro time j ht hj
    have cover := Geometry.coordinate_cover h.T c.BT time positive ht
    let i : Fin c.BT := ⟨time%c.BT,cover.2.1⟩
    have eq : time/c.BT*c.BT+i.val=time := cover.2.2
    have active := (Geometry.active_iff h c p (time/c.BT) i.val j.val
      (Geometry.chunk_start h c positive _ cover.1)).mpr (by simpa [eq] using And.intro ht hj)
    have out := inv.output (time/c.BT) cover.1 i j active
    simpa only [eq] using out

#axiomsClean source_correct
end
end VeriTile.Bench.Repairs.ChunkCumsumVector.CandidateExecution
