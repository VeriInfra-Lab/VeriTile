import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeOutputReads

/-! Exact relation between actual typed-cell source stores and the established
native payload recurrence. The observer is used in a proof only; execution
retains every predecessor cell and never exports, reimports or retags memory. -/
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.NativeProgram
open VeriTile Triton FP.Structural
open FPStorage (Memory state payload)
open FPStages (Stage diagonalLayout mergeLayout)
open NativeLaunch (start)
open NativeReads
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
variable {α : Type} [Inhabited α]
variable (scope : RegionName → Nat → Prop)
set_option maxRecDepth 30000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def Relation (scope : RegionName → Nat → Prop) (observe : RegionName → Cell α → α) (s t : State α) : Prop :=
  ∀ r off,scope r off → observe r (s.mem r off)=payload (t.mem r off)

theorem imported (observe : RegionName → Cell α → α) (stage : Stage)
    (idx : GridIndex stage.grid) (s : State α) (m : Memory α)
    (hm : ∀ r off,scope r off → observe r (s.mem r off)=m r off) :
    Relation scope observe (start stage idx s) (state m idx.toPids stage.grid.toNumPids) := by
  intro r off inside
  exact (hm r off inside).trans (FPStorage.payload_cell _ _).symm

theorem write_relation (observe : RegionName → Cell α → α) (s t : State α)
    (h : Relation scope observe s t) (r : RegionName) (off : Nat) (cell : Cell α)
    (hv : observe r cell=payload cell) : Relation scope observe (s.write r off cell) (t.write r off cell) := by
  intro region address inside
  dsimp only [State.write]
  by_cases active : region=r ∧ address=off
  · rw [if_pos active,if_pos active]
    simpa only [active.1] using hv
  · rw [if_neg active,if_neg active]
    exact h region address inside

theorem scatter_relation {ι : Type} (observe : RegionName → Cell α → α)
    (r : RegionName) (active : ι → Prop) [DecidablePred active]
    (address : ι → Nat) (values : ι → Cell α) (order : List ι)
    (hv : ∀ i,observe r (values i)=payload (values i))
    (s t : State α) (h : Relation scope observe s t) :
    Relation scope observe
      (order.foldl (fun a i=>if active i then a.write r (address i) (values i) else a) s)
      (order.foldl (fun a i=>if active i then a.write r (address i) (values i) else a) t) := by
  induction order generalizing s t with
  | nil => exact h
  | cons i rest ih =>
    simp only [List.foldl_cons]
    by_cases hi : active i
    · rw [if_pos hi,if_pos hi]
      exact ih _ _ (write_relation scope observe s t h r (address i) (values i) (hv i))
    · rw [if_neg hi,if_neg hi]
      exact ih s t h

theorem range_relation (observe : RegionName → Cell α → α)
    (f g : Nat → State α → State α) (first last : Nat) (s t : State α)
    (h : Relation scope observe s t)
    (step : ∀ j,first≤j → j<last → ∀ a b,Relation scope observe a b → Relation scope observe (f j a) (g j b)) :
    Relation scope observe (foldRange f first last 1 s) (foldRange g first last 1 t) := by
  by_cases inside : first<last
  · conv_lhs => rw [foldRange]
    conv_rhs => rw [foldRange]
    simp only [Nat.one_ne_zero,↓reduceDIte,inside]
    exact range_relation observe f g (first+1) last _ _ (step first (Nat.le_refl _) inside s t h)
      (fun j hj hj' a b=>step j (by omega) hj' a b)
  · conv_lhs => rw [foldRange]
    conv_rhs => rw [foldRange]
    simpa only [Nat.one_ne_zero,↓reduceDIte,inside] using h
termination_by last-first
decreasing_by omega

def WriteCompatible (observe : RegionName → Cell α → α) : Stage → Prop
  | .intra _ _ _ => ∀ v,observe "attention" (.mk .real v)=v
  | .split _ _ => ∀ v,observe "scratch" (.mk .real v)=v
  | .inter _ _ | .merge _ => ∀ v,observe "attention" (.mk .fp32 v)=v
  | .output _ _ => ∀ v,observe "o" (.mk .fp32 v)=v

theorem inter_relation (observe : RegionName → Cell α → α)
    (M : Algebra α) (D : FPOutput.DotPrimitive α) (c : ProducerGeometry.Shape) (scale : ℝ)
    (idx : GridIndex (ProducerGeometry.grid c)) (s : State α) (m : Memory α)
    (hm : ∀ r off,scope r off → observe r (s.mem r off)=m r off)
    (reads : Reads (.inter c scale) s m) (hw : ∀ v,observe "attention" (.mk .fp32 v)=v) :
    Relation scope observe (FPStages.program M D (.inter c scale) (start (.inter c scale) idx s))
      (FPStages.program M D (.inter c scale) (state m idx.toPids (ProducerGeometry.grid c).toNumPids)) := by
  have base := imported scope observe (.inter c scale) idx s m hm
  dsimp only [FPStages.program]
  by_cases enabled : FPInter.enabled c (start (.inter c scale) idx s)
  · have enabled' : FPInter.enabled c (state m idx.toPids (ProducerGeometry.grid c).toNumPids) := enabled
    rw [if_pos enabled,if_pos enabled']
    have values := inter_value M D c scale idx s m reads enabled
    unfold FPInter.written
    rw [values]
    exact scatter_relation scope observe "attention" _ _ _ _ (fun i=>hw _) _ _ base
  · have disabled : ¬FPInter.enabled c (state m idx.toPids (ProducerGeometry.grid c).toNumPids) := enabled
    rw [if_neg enabled,if_neg disabled]
    exact base

theorem merge_relation (observe : RegionName → Cell α → α)
    (M : Algebra α) (D : FPOutput.DotPrimitive α) (c : ProducerGeometry.Shape)
    (idx : GridIndex (DiagonalGeometry.grid c)) (s : State α) (m : Memory α)
    (hm : ∀ r off,scope r off → observe r (s.mem r off)=m r off)
    (reads : Reads (.merge c) s m) (hw : ∀ v,observe "attention" (.mk .fp32 v)=v) :
    Relation scope observe (FPStages.program M D (.merge c) (start (.merge c) idx s))
      (FPStages.program M D (.merge c) (state m idx.toPids (DiagonalGeometry.grid c).toNumPids)) := by
  have base := imported scope observe (.merge c) idx s m hm
  dsimp only [FPStages.program]
  by_cases enabled : FPMerge.enabled (mergeLayout c) (start (.merge c) idx s)
  · have enabled' : FPMerge.enabled (mergeLayout c) (state m idx.toPids (DiagonalGeometry.grid c).toNumPids) := enabled
    rw [if_pos enabled,if_pos enabled']
    have values := merge_value M c idx s m reads
    unfold FPMerge.written
    rw [values]
    exact scatter_relation scope observe "attention" _ _ _ _ (fun i=>hw _) _ _ base
  · have disabled : ¬FPMerge.enabled (mergeLayout c) (state m idx.toPids (DiagonalGeometry.grid c).toNumPids) := enabled
    rw [if_neg enabled,if_neg disabled]
    exact base

theorem output_relation (observe : RegionName → Cell α → α)
    (M : Algebra α) (D : FPOutput.DotPrimitive α) (c : OutputGrid.Shape) (scale : ℝ)
    (idx : GridIndex (OutputGrid.grid c)) (s : State α) (m : Memory α)
    (hm : ∀ r off,scope r off → observe r (s.mem r off)=m r off)
    (reads : Reads (.output c scale) s m) (hw : ∀ v,observe "o" (.mk .fp32 v)=v) :
    Relation scope observe (FPStages.program M D (.output c scale) (start (.output c scale) idx s))
      (FPStages.program M D (.output c scale) (state m idx.toPids (OutputGrid.grid c).toNumPids)) := by
  have base := imported scope observe (.output c scale) idx s m hm
  dsimp only [FPStages.program]
  have values := output_value M D c scale idx s m reads
  unfold FPOutput.written
  rw [values]
  exact scatter_relation scope observe "o" _ _ _ _ (fun i=>hw _) _ _ base

theorem diagonal_relation (observe : RegionName → Cell α → α)
    (M : Algebra α) (split : Bool) (c : FPDiagonal.Layout) (scale : ℝ) (heads : Nat)
    (s : State α) (m : Memory α) (hh : s.pids 2<heads)
    (hm : ∀ r off,scope r off → observe r (s.mem r off)=m r off)
    (reads : DiagonalReads c heads s m) (hw : ∀ v,observe (FPDiagonal.out split) (.mk .real v)=v) :
    Relation scope observe
      (if FPDiagonal.enabled split c s then FPDiagonal.written M split c scale s else s)
      (if FPDiagonal.enabled split c (state m s.pids s.numPids) then
        FPDiagonal.written M split c scale (state m s.pids s.numPids) else state m s.pids s.numPids) := by
  have base : Relation scope observe s (state m s.pids s.numPids) :=
    fun r off inside=>(hm r off inside).trans (FPStorage.payload_cell _ _).symm
  by_cases enabled : FPDiagonal.enabled split c s
  · have enabled' : FPDiagonal.enabled split c (state m s.pids s.numPids) := enabled
    rw [if_pos enabled,if_pos enabled']
    unfold FPDiagonal.written
    apply range_relation scope observe _ _ _ _ _ _ base
    intro j _ hj a b related
    have values := diagonal_value M split c scale heads s m hh reads j hj
    unfold FPDiagonal.columnWritten
    rw [values]
    exact scatter_relation scope observe (FPDiagonal.out split) _ _ _ _ (fun i=>hw _) a b related
  · have disabled : ¬FPDiagonal.enabled split c (state m s.pids s.numPids) := enabled
    rw [if_neg enabled,if_neg disabled]
    exact base

theorem program_relation (observe : RegionName → Cell α → α)
    (M : Algebra α) (D : FPOutput.DotPrimitive α) (stage : Stage)
    (idx : GridIndex stage.grid) (s : State α) (m : Memory α)
    (hm : ∀ r off,scope r off → observe r (s.mem r off)=m r off)
    (reads : Reads stage s m) (hw : WriteCompatible observe stage) :
    Relation scope observe (FPStages.program M D stage (start stage idx s))
      (FPStages.program M D stage (state m idx.toPids stage.grid.toNumPids)) := by
  cases stage with
  | inter c scale => exact inter_relation scope observe M D c scale idx s m hm reads hw
  | merge c => exact merge_relation scope observe M D c idx s m hm reads hw
  | output c scale => exact output_relation scope observe M D c scale idx s m hm reads hw
  | intra c BK scale =>
    apply diagonal_relation scope observe M Bool.false (diagonalLayout c BK) scale c.heads
      (start (.intra c BK scale) idx s) m _ hm _ hw
    · simpa only [start,Stage.grid,DiagonalGeometry.grid,Grid.dim] using
        FPStages.pid_bound (DiagonalGeometry.grid c) idx 2 (by simp [DiagonalGeometry.grid,Grid.rank])
    · intro r hr off ho
      exact real_read reads r (c.heads*c.T*c.K) (by
        rcases hr with rfl|rfl|rfl <;> simp [NativeLaunch.inputExtents])
        (by rcases hr with rfl|rfl|rfl <;> decide) off ho
  | split c scale =>
    apply diagonal_relation scope observe M Bool.true (diagonalLayout c c.BK) scale c.heads
      (start (.split c scale) idx s) m _ hm _ hw
    · simpa only [start,Stage.grid,SplitGeometry.grid,Grid.dim] using
        FPStages.pid_bound (SplitGeometry.grid c) idx 2 (by simp [SplitGeometry.grid,Grid.rank])
    · intro r hr off ho
      exact real_read reads r (c.heads*c.T*c.K) (by
        rcases hr with rfl|rfl|rfl <;> simp [NativeLaunch.inputExtents])
        (by rcases hr with rfl|rfl|rfl <;> decide) off ho

theorem source_reference (observe : RegionName → Cell α → α)
    (M : Algebra α) (D : FPOutput.DotPrimitive α) (stage : Stage)
    (idx : GridIndex stage.grid) (s : State α) (m : Memory α)
    (hm : ∀ r off,scope r off → observe r (s.mem r off)=m r off)
    (reads : Reads stage s m) (hw : WriteCompatible observe stage) :
    ∃ t,(NativeLaunch.job M D stage idx).run s=some t ∧
      ∀ r off,scope r off → observe r (t.mem r off)=FPStages.referenceStep M D stage idx m r off := by
  obtain ⟨t,ran,mem⟩ := NativeLaunch.program_run M D stage idx s
  refine ⟨t,ran,?_⟩
  intro r off inside
  rw [mem]
  exact program_relation scope observe M D stage idx s m hm reads hw r off inside

#axiomsClean source_reference

end
end VeriTile.Bench.Repairs.ChunkGlaFwd.NativeProgram
