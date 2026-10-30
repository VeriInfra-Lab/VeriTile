import bench.tritonbench_g.attn_fwd_triton.Proofs.NativeBindings
import bench.tritonbench_g.attn_fwd_triton.Proofs.RecurrenceLocality

/-! A narrow serialization of the existing independent attention grid.
Earlier stores preserve every original input read, and every output has a
unique program owner. No general concurrency or GPU runtime is modeled. -/
namespace VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.NativeInterface
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open AttnFwdTriton.TypedExecution Graph Model ReadLocality
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

abbrev width (p : Config) := AttnFwdTriton.HostShape.blockD p.headDim

theorem invocation_initial (p : Config) (idx : ProgramIndex p) (s : State Expr) :
    (invocation p idx).initial s = atProgram p s idx := rfl

theorem schedule_safe (p : Config) (stage : p.stage=3) (work : List (ProgramIndex p))
    (bounds : RegionName → Nat) (windows : SourceSafety.Windows p bounds) (s : State Expr) :
    ScheduleSafe bounds (work.map (invocation p)) s := by
  induction work generalizing s with
  | nil => trivial
  | cons idx rest ih =>
    refine ⟨SourceSafety.source_safe p (width p) idx stage s bounds windows,?_⟩
    exact fun t _ => ih t

theorem schedule_run (p : Config) (stage : p.stage=3) (nonoverlap : NonOverlapping p)
    (separate : Separated p) (work : List (ProgramIndex p)) (unique : work.Nodup)
    (original : ReadLocality.Memory) (s : State Expr) (same : Agree p s.mem original) :
    ∃ t, runSchedule (work.map (invocation p)) s=some t ∧
      (∀ idx ∈ work, ∀ i : TileIndex [128,width p], outputActive p (width p) idx.1.val i=Bool.true →
        t.mem p.Out (outputAddress p (width p) idx.1.val idx.2.val i)=
          Cell.mk .bf16 (Model.output (Model.result p (width p) original idx.1.val idx.2.val) i)) ∧
      (∀ r off, (∀ idx ∈ work, ¬Written p (width p) idx.1.val idx.2.val r off) →
        t.mem r off=s.mem r off) := by
  induction work generalizing s with
  | nil => exact ⟨s,rfl,by simp,by intro r off _; rfl⟩
  | cons idx rest ih =>
    obtain ⟨notin,nodup⟩ := List.nodup_cons.mp unique
    obtain ⟨mid,ran,hframe,outputs⟩ := Model.program p (width p) stage nonoverlap s idx
    have next : Agree p mid.mem original := by
      intro key input
      exact (frame_agree p (width p) idx separate s mid hframe key input).trans (same key input)
    obtain ⟨t,tail,tailout,tailframe⟩ := ih nodup mid next
    refine ⟨t,?_,?_,?_⟩
    · change ((invocation p idx).run s).bind (runSchedule (rest.map (invocation p)))=some t
      change (FP.Structural.exec (engine backend) (source p (width p) Bool.false)
        (atProgram p s idx)).bind (runSchedule (rest.map (invocation p)))=some t
      rw [ran]
      exact tail
    · intro j member i active
      rcases List.mem_cons.mp member with equal | member
      · subst j
        rw [tailframe p.Out (outputAddress p (width p) idx.1.val idx.2.val i) (by
          intro j member written
          have equal := disjoint p (width p) nonoverlap idx j p.Out _
            ⟨rfl,i,active,rfl⟩ written
          exact notin (equal ▸ member))]
        rw [outputs i active,ReadLocality.result_eq p (width p) idx s.mem original same]
      · exact tailout j member i active
    · intro r off outside
      rw [tailframe r off (fun j member => outside j (List.mem_cons_of_mem idx member))]
      exact hframe r off (outside idx List.mem_cons_self)

theorem implements (p : Config) (stage : p.stage=3) (nonoverlap : NonOverlapping p)
    (separate : Separated p) : (programIO p).Implements (formula p) := by
  apply ProgramIO.Implements.intro
  · intro values bounds backed s _
    exact schedule_safe p stage (fullGrid p) bounds (windows p values bounds backed) s
  · intro values s bound
    obtain ⟨t,ran,outputs,outside⟩ := schedule_run p stage nonoverlap separate (fullGrid p)
      (by exact Finset.nodup_toList _) (logicalMemory p values) s (logical_agree p values s bound)
    refine ⟨t,ran,?_,frame p values s t ?_⟩
    · intro o j _
      fin_cases o
      let c := coordinate p j
      have h := outputs (owner p c) (by simp [fullGrid])
        (lane p (width p) (AttnFwdTriton.HostShape.blockD_covers _) c)
        (lane_active p (width p) (AttnFwdTriton.HostShape.blockD_covers _) c)
      rw [lane_address] at h
      change mapCell id (t.mem p.Out (logicalAddress p c)) =
        Cell.mk .bf16 (Model.output (Model.result p (width p) (logicalMemory p values)
          (owner p c).1.val (owner p c).2.val)
          (lane p (width p) (AttnFwdTriton.HostShape.blockD_covers _) c))
      rw [h]
      rfl
    · intro r off notOutput
      exact outside r off (fun idx _ written =>
        notOutput (written_within p (width p) idx r off written))

/-- The original ordinary grid has exactly the same memory as the shared
schedule. Registers remain local program state and are not a grid observable. -/
theorem grid_memory_eq (p : Config) (stage : p.stage=3) (nonoverlap : NonOverlapping p)
    (separate : Separated p) (s ordinary : State Expr)
    (grid : GridRun backend p (width p) Bool.false s ordinary) :
    ∃ serial, runSchedule ((fullGrid p).map (invocation p)) s=some serial ∧
      serial.mem=ordinary.mem := by
  obtain ⟨serial,ran,outputs,frame⟩ := schedule_run p stage nonoverlap separate (fullGrid p)
    (by exact Finset.nodup_toList _) s.mem s (fun _ _=>rfl)
  refine ⟨serial,ran,?_⟩
  funext r off
  by_cases inside : Window p r off
  · obtain ⟨rfl,c,rfl⟩ := inside
    let idx := owner p c
    let i := lane p (width p) (AttnFwdTriton.HostShape.blockD_covers _) c
    have active := lane_active p (width p) (AttnFwdTriton.HostShape.blockD_covers _) c
    have address := lane_address p (width p) (AttnFwdTriton.HostShape.blockD_covers _) c
    obtain ⟨one,executed,_,value⟩ := Model.program p (width p) stage nonoverlap s idx
    have same : one=grid.programs idx := Option.some.inj (executed.symm.trans (grid.executed idx))
    have hs := outputs idx (by simp [idx,fullGrid]) i active
    have ho := value i active
    rw [address] at hs ho
    rw [hs,grid.readback idx p.Out _ (covered p (width p)
      (AttnFwdTriton.HostShape.blockD_covers _) c),←same,ho]
  · rw [frame r off (fun idx _ written => inside (written_within p (width p) idx r off written))]
    exact (grid.frame r off inside).symm

#axiomsClean implements
#axiomsClean grid_memory_eq
end
end VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.NativeInterface
