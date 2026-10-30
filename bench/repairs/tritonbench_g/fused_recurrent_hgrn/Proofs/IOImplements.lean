import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.GridSafety
import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.FormulaCongruence

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn.IOImplements
open VeriTile.Triton FP.Structural Execution
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false
set_option linter.unusedSectionVars false
variable {α : Type} [Inhabited α]

theorem invocation_initial (p : Ops α) (source : ComputeKernel) (id : Nat × Nat) (s : State α) :
    (IO.invocation p source id).initial s=enter s id := by
  unfold Invocation.initial IO.invocation bindRegisters enter
  simp only [List.foldl_nil,Bool.false_eq_true,if_false,Option.getD_none]
  congr 1
  funext ax
  by_cases h0 : ax=0 <;> by_cases h1 : ax=1 <;> simp [h0,h1]

theorem invocation_run (p : Ops α) (source : ComputeKernel) (id : Nat × Nat) (s : State α) :
    (IO.invocation p source id).run s=run (algebra p) source.surfaceBody (enter s id) := by
  unfold Invocation.run
  rw [invocation_initial]
  rfl

theorem run_schedule (p : Ops α) (source : ComputeKernel) (ids : List (Nat × Nat))
    (s : State α) : runSchedule (ids.map (IO.invocation p source)) s=
      gridRun p source.surfaceBody ids s := by
  induction ids generalizing s with
  | nil => rfl
  | cons id ids ih =>
    simp only [List.map_cons,runSchedule,invocation_run,gridRun]
    cases run (algebra p) source.surfaceBody (enter s id) <;> simp [ih]

theorem windows (p : Ops α) (obs : α → α) (BH T D BDF BDB : Nat) (U S V : Bool)
    (vals : ChannelValues α (IO.inputs BH T D)) (bounds : RegionName → Nat)
    (h : (IO.recurrentIO p obs BH T D BDF BDB U S V).InBounds bounds vals) :
    SourceSafety.Windows bounds BH T D U S V := by
  constructor
  · intro r hr bh hb t ht d hd
    rcases hr with rfl|rfl|rfl|rfl|rfl|rfl
    · exact h.2.1 (0:Fin 5) (⟨bh,hb⟩,⟨t,ht⟩,⟨d,hd⟩,()) trivial
    · exact h.2.1 (1:Fin 5) (⟨bh,hb⟩,⟨t,ht⟩,⟨d,hd⟩,()) trivial
    · exact h.2.1 (2:Fin 5) (⟨bh,hb⟩,⟨t,ht⟩,⟨d,hd⟩,()) trivial
    · exact h.2.2.1 (0:Fin 4) (⟨bh,hb⟩,⟨t,ht⟩,⟨d,hd⟩,()) (Or.inl (by simp))
    · exact h.2.2.1 (1:Fin 4) (⟨bh,hb⟩,⟨t,ht⟩,⟨d,hd⟩,()) (Or.inl (by simp))
    · exact h.2.2.1 (2:Fin 4) (⟨bh,hb⟩,⟨t,ht⟩,⟨d,hd⟩,()) (Or.inl (by simp))
  · intro hu bh hb d hd
    exact h.2.1 (3:Fin 5) (⟨bh,hb⟩,⟨d,hd⟩,()) hu
  · intro hs bh hb d hd
    exact h.2.2.1 (3:Fin 4) (⟨bh,hb⟩,⟨d,hd⟩,()) (Or.inr hs)
  · intro hv bh hb d hd
    exact h.2.1 (4:Fin 5) (⟨bh,hb⟩,⟨d,hd⟩,()) hv

private theorem mapped_safe (p : Ops α) (source : ComputeKernel) (bounds : RegionName → Nat)
    (BH : Nat) (safe : ∀ s : State α, s.pids 1<BH → TraceSafe (algebra p) bounds source s)
    (ids : List (Nat × Nat)) (valid : ∀ id∈ids, id.2<BH) (s : State α) :
    ScheduleSafe bounds (ids.map (IO.invocation p source)) s := by
  induction ids generalizing s with
  | nil => trivial
  | cons id rest ih =>
    constructor
    · unfold Invocation.Safe
      rw [invocation_initial]
      exact safe _ (valid id List.mem_cons_self)
    · intro t _
      exact ih (fun j hj => valid j (List.mem_cons_of_mem _ hj)) t

theorem schedule_safe (p : Ops α) (obs : α → α) (BH T D BDF BDB : Nat)
    (U S V : Bool) (vals : ChannelValues α (IO.inputs BH T D))
    (bounds : RegionName → Nat)
    (h : (IO.recurrentIO p obs BH T D BDF BDB U S V).InBounds bounds vals) (s : State α) :
    ScheduleSafe bounds ((IO.recurrentIO p obs BH T D BDF BDB U S V).schedule vals) s := by
  have W := windows p obs BH T D BDF BDB U S V vals bounds h
  apply (scheduleSafe_append _ _ _ _).mpr
  constructor
  · apply mapped_safe p _ bounds BH
      (fun s hb => SourceSafety.forward_safe p s BH T D BDF U S V bounds W hb)
    intro id hid
    exact ((programs_mem BH D BDF id).mp hid).2
  · intro t _
    apply mapped_safe p _ bounds BH
      (fun s hb => SourceSafety.backward_safe p s BH T D BDB U S V bounds W hb)
    intro id hid
    exact ((programs_mem BH D BDB id).mp hid).2

/-- The old execution theorem uses total readers. This internal snapshot
supplies those readers without imposing values outside the finite IO windows. -/
def snapshot (s : State α) (T D : Nat) : Input α := fun r bh t d =>
  if r="h0" ∨ r="dht" then read s r (stateAddr bh D d) else read s r (rowAddr bh T D t d)

theorem snapshot_inputs (s : State α) (T D : Nat) : Inputs s (snapshot s T D) T D := by
  constructor
  · intro r hr bh t d
    rcases hr with rfl|rfl|rfl <;> simp [snapshot]
  · intro r hr bh d
    rcases hr with rfl|rfl <;> simp [snapshot]

theorem input_agreement (p : Ops α) (obs : α → α) (BH T D BDF BDB : Nat)
    (U S V : Bool) (vals : ChannelValues α (IO.inputs BH T D)) (s : State α)
    (h : (IO.recurrentIO p obs BH T D BDF BDB U S V).InputBinding vals s)
    (bh d : Nat) (hb : bh<BH) (hd : d<D) :
    FormulaCongruence.Agree (snapshot s T D) (IO.asInput p BH T D vals) T bh d U V := by
  constructor
  · intro r hr t ht
    rcases hr with rfl|rfl|rfl
    · simpa [snapshot,IO.asInput,hb,hd,ht,Execution.read,IO.recurrentIO,mapValue] using
        h (0:Fin 5) (⟨bh,hb⟩,⟨t,ht⟩,⟨d,hd⟩,()) trivial
    · simpa [snapshot,IO.asInput,hb,hd,ht,Execution.read,IO.recurrentIO,mapValue] using
        h (1:Fin 5) (⟨bh,hb⟩,⟨t,ht⟩,⟨d,hd⟩,()) trivial
    · simpa [snapshot,IO.asInput,hb,hd,ht,Execution.read,IO.recurrentIO,mapValue] using
        h (2:Fin 5) (⟨bh,hb⟩,⟨t,ht⟩,⟨d,hd⟩,()) trivial
  · intro hu
    simpa [snapshot,IO.asInput,hb,hd,Execution.read,IO.recurrentIO,mapValue] using
      h (3:Fin 5) (⟨bh,hb⟩,⟨d,hd⟩,()) hu
  · intro hv
    simpa [snapshot,IO.asInput,hb,hd,Execution.read,IO.recurrentIO,mapValue] using
      h (4:Fin 5) (⟨bh,hb⟩,⟨d,hd⟩,()) hv

private theorem row_index (BH T D a : Nat) (h : a<BH*T*D) :
    ∃ j : TileIndex [BH,T,D], rowAddr j.1.val T D j.2.1.val j.2.2.1.val=a := by
  have dp : 0<D := by
    by_contra hn
    have hz : D=0 := by omega
    simp [hz] at h
  have tp : 0<T := by
    by_contra hn
    have hz : T=0 := by omega
    simp [hz] at h
  have q : a/D<BH*T := (Nat.div_lt_iff_lt_mul dp).mpr h
  have bh : a/D/T<BH := (Nat.div_lt_iff_lt_mul tp).mpr q
  refine ⟨(⟨a/D/T,bh⟩,⟨a/D%T,Nat.mod_lt _ tp⟩,⟨a%D,Nat.mod_lt _ dp⟩,()),?_⟩
  dsimp [rowAddr]
  rw [Nat.mul_comm (a/D/T) T,Nat.div_add_mod,Nat.mul_comm (a/D) D,Nat.div_add_mod]

private theorem slot_index (BH D a : Nat) (h : a<BH*D) :
    ∃ j : TileIndex [BH,D], stateAddr j.1.val D j.2.1.val=a := by
  have dp : 0<D := by
    by_contra hn
    have hz : D=0 := by omega
    simp [hz] at h
  refine ⟨(⟨a/D,(Nat.div_lt_iff_lt_mul dp).mpr h⟩,⟨a%D,Nat.mod_lt _ dp⟩,()),?_⟩
  simpa only [stateAddr,Nat.mul_comm] using Nat.div_add_mod a D

theorem outside_phases (p : Ops α) (obs : α → α) (BH T D BDF BDB : Nat)
    (U S V : Bool) (vals : ChannelValues α (IO.inputs BH T D)) (r : RegionName) (a : Nat)
    (h : (IO.recurrentIO p obs BH T D BDF BDB U S V).Outside vals r a) :
    ¬ForwardRegion BH T D S r a ∧ ¬BackwardRegion BH T D r a := by
  constructor
  · rintro (⟨rfl,ha⟩|⟨hs,rfl,ha⟩)
    · obtain ⟨j,hj⟩ := row_index BH T D a ha
      exact (h.1 (0:Fin 4) j (Or.inl (by simp))).elim (fun hn => hn rfl) (fun hn => hn hj.symm)
    · obtain ⟨j,hj⟩ := slot_index BH D a ha
      exact (h.1 (3:Fin 4) j (Or.inr hs)).elim (fun hn => hn rfl) (fun hn => hn hj.symm)
  · rintro ⟨hr,ha⟩
    obtain ⟨j,hj⟩ := row_index BH T D a ha
    rcases hr with rfl|rfl
    · exact (h.1 (1:Fin 4) j (Or.inl (by simp))).elim (fun hn => hn rfl) (fun hn => hn hj.symm)
    · exact (h.1 (2:Fin 4) j (Or.inl (by simp))).elim (fun hn => hn rfl) (fun hn => hn hj.symm)

def target (p : Ops α) (obs : α → α) (BH T D : Nat) (U V : Bool)
    (vals : ChannelValues α (IO.inputs BH T D)) : ChannelValues α (IO.outputs BH T D) :=
  let input := IO.asInput p BH T D vals
  IO.result BH T D (fun bh d n => obs (state p U input bh d n))
    (fun bh d t => obs (dx p V input T bh d t))
    (fun bh d t => obs (dg p U V input T bh d t))

/-- Both complete source phases, every output, active accesses and exact final
frame. The reused RawKernel result also retains the separate phase frames. -/
theorem implements (p : Ops α) (obs : α → α) (BH T D BDF BDB : Nat)
    (hF : 0<BDF) (hB : 0<BDB) (U S V : Bool) :
    (IO.recurrentIO p obs BH T D BDF BDB U S V).Implements (target p obs BH T D U V) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds hb initial _
    exact schedule_safe p obs BH T D BDF BDB U S V vals bounds hb initial
  · intro vals initial hin
    let input := snapshot initial T D
    obtain ⟨f,b,h⟩ := RawKernel.complete p input BH T D BDF BDB hF hB U S V initial (snapshot_inputs initial T D)
    have agree := input_agreement p obs BH T D BDF BDB U S V vals initial hin
    refine ⟨b,?_,?_,?_⟩
    · change runSchedule (_++_) initial=some b
      rw [runSchedule_append,run_schedule,show IO.programs BH D BDF=programs BH D BDF from rfl,h.forwardExecuted]
      simpa only [Option.bind_some,run_schedule] using h.backwardExecuted
    · intro o j hm
      fin_cases o
      · have result := h.output j.1.val j.1.isLt j.2.1.val j.2.1.isLt j.2.2.1.val j.2.2.1.isLt
        have eq := FormulaCongruence.state_eq p input (IO.asInput p BH T D vals)
          T j.1.val j.2.2.1.val U V (agree _ _ j.1.isLt j.2.2.1.isLt) (j.2.1.val+1) (by omega)
        change mapCell obs (b.mem "o" (rowAddr j.1.val T D j.2.1.val j.2.2.1.val))=
          Cell.mk .real (obs (state p U (IO.asInput p BH T D vals) j.1.val j.2.2.1.val (j.2.1.val+1)))
        rw [result]
        change Cell.mk .real (obs (state p U input j.1.val j.2.2.1.val (j.2.1.val+1)))=_
        rw [eq]
      · have result := (h.gradients j.1.val j.1.isLt j.2.1.val j.2.1.isLt j.2.2.1.val j.2.2.1.isLt).1
        have eq := FormulaCongruence.dx_eq p input (IO.asInput p BH T D vals)
          T j.1.val j.2.2.1.val U V (agree _ _ j.1.isLt j.2.2.1.isLt) j.2.1.val j.2.1.isLt
        change mapCell obs (b.mem "dx" (rowAddr j.1.val T D j.2.1.val j.2.2.1.val))=
          Cell.mk .real (obs (dx p V (IO.asInput p BH T D vals) T j.1.val j.2.2.1.val j.2.1.val))
        rw [result]
        change Cell.mk .real (obs (dx p V input T j.1.val j.2.2.1.val j.2.1.val))=_
        rw [eq]
      · have result := (h.gradients j.1.val j.1.isLt j.2.1.val j.2.1.isLt j.2.2.1.val j.2.2.1.isLt).2
        have eq := FormulaCongruence.dg_eq p input (IO.asInput p BH T D vals)
          T j.1.val j.2.2.1.val U V (agree _ _ j.1.isLt j.2.2.1.isLt) j.2.1.val j.2.1.isLt
        change mapCell obs (b.mem "dg" (rowAddr j.1.val T D j.2.1.val j.2.2.1.val))=
          Cell.mk .real (obs (dg p U V (IO.asInput p BH T D vals) T j.1.val j.2.2.1.val j.2.1.val))
        rw [result]
        change Cell.mk .real (obs (dg p U V input T j.1.val j.2.2.1.val j.2.1.val))=_
        rw [eq]
      · have hs : S=Bool.true := hm.resolve_left (by simp)
        have result := h.finalState hs j.1.val j.1.isLt j.2.1.val j.2.1.isLt
        have eq := FormulaCongruence.state_eq p input (IO.asInput p BH T D vals)
          T j.1.val j.2.1.val U V (agree _ _ j.1.isLt j.2.1.isLt) T (by omega)
        change mapCell obs (b.mem "ht" (stateAddr j.1.val D j.2.1.val))=
          Cell.mk .real (obs (state p U (IO.asInput p BH T D vals) j.1.val j.2.1.val T))
        rw [result]
        change Cell.mk .real (obs (state p U input j.1.val j.2.1.val T))=_
        rw [eq]
    · intro r a outside
      obtain ⟨hf,hb⟩ := outside_phases p obs BH T D BDF BDB U S V vals r a outside
      rw [h.backwardFrame r a hb,h.forwardFrame r a hf]

#axiomsClean implements
end
end VeriTile.Bench.Repairs.FusedRecurrentHgrn.IOImplements
