import bench.tritonbench_g.destindex_copy_kv1.IO
import bench.tritonbench_g.destindex_copy_kv1.Proofs.LogicalGrid
import bench.tritonbench_g.destindex_copy_kv1.Proofs.SourceSafety

namespace VeriTile.Bench.Repairs.DestindexCopyKv1.CopyIO
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 5000000
variable {α : Type} [Inhabited α]

/-- Finite logical IO bindings are exactly the reads needed by the source. -/
theorem reads (rest : Algebra α) (c : Config) (ps : List Nat) (values : Inputs α c)
    (admitted : Admissible c values) (s : State α)
    (input : (copyIO rest c ps).InputBinding values s) :
    LogicalGrid.Bound c s (source c values) (destination c values) := by
  constructor
  · intro row hr head dim hh hd
    simpa [copyIO,memory,inputChannels,source,hr,hh,hd,mapValue] using
      input ⟨0,by simp [copyIO,memory,inputChannels]⟩
        (⟨row,hr⟩,⟨head,hh⟩,⟨dim,hd⟩,PUnit.unit) trivial
  · intro row hr
    have h := input ⟨1,by simp [copyIO,memory,inputChannels]⟩ (⟨row,hr⟩,PUnit.unit) trivial
    change (s.mem metadataRegion.cast row).read .int=destinationWord c values ⟨row,hr⟩ at h
    rw [h]
    simp only [destination,dif_pos hr]
    exact (Int.toNat_of_nonneg (admitted.1 ⟨row,hr⟩)).symm

theorem destination_unique (c : Config) (values : Inputs α c) (admitted : Admissible c values) :
    ∀ p q,p<c.count → q<c.count → destination c values p=destination c values q → p=q := by
  intro p q hp hq same
  have hp0 := admitted.1 ⟨p,hp⟩
  have hq0 := admitted.1 ⟨q,hq⟩
  have eq : destinationWord c values ⟨p,hp⟩=destinationWord c values ⟨q,hq⟩ := by
    simp only [destination,dif_pos hp,dif_pos hq] at same
    omega
  exact congrArg Fin.val (admitted.2 eq)

theorem invocation_run (rest : Algebra α) (c : Config) (pid : Nat) (s : State α) :
    (invocation rest c pid).run s=some (LogicalGrid.step rest c s pid) := by
  simpa only [invocation,Invocation.run,Invocation.initial,bindRegisters,List.foldl_nil,
    Option.getD_none,LogicalGrid.launch,LogicalGrid.step] using
    (Tile.execution rest (layout c) (LogicalGrid.launch pid s)).property

theorem schedule_run (rest : Algebra α) (c : Config) (ps : List Nat) (s : State α) :
    runSchedule (ps.map (invocation rest c)) s=some (LogicalGrid.run rest c ps s) := by
  induction ps generalizing s with
  | nil => rfl
  | cons p ps ih =>
    simp only [List.map_cons,runSchedule,invocation_run,Option.bind]
    exact ih _

theorem invocation_safe (rest : Algebra α) (c : Config) (ps : List Nat)
    (values : Inputs α c) (bounds : RegionName → Nat)
    (bounded : (copyIO rest c ps).InBounds bounds values) (s : State α)
    (read : LogicalGrid.Bound c s (source c values) (destination c values))
    (pid : Nat) (hp : pid<c.count) : (invocation rest c pid).Safe bounds s := by
  apply Tile.source_trace_safe
  · exact bounded.2.1 ⟨1,by simp [copyIO,memory,inputChannels]⟩ (⟨pid,hp⟩,PUnit.unit) trivial
  · intro i active
    have h := bounded.2.1 ⟨0,by simp [copyIO,memory,inputChannels]⟩
      (⟨pid,hp⟩,⟨i.1.val,active.1⟩,⟨i.2.1.val,active.2⟩,PUnit.unit) trivial
    simpa [copyIO,memory,inputChannels,invocation,Invocation.initial,bindRegisters,
      Tile.inputOffset,layout,flat,Nat.mul_comm] using h
  · intro i active
    have h := bounded.2.2.1 ⟨0,by simp [copyIO,memory,outputChannels]⟩
      (⟨pid,hp⟩,⟨i.1.val,active.1⟩,⟨i.2.1.val,active.2⟩,PUnit.unit) trivial
    change Tile.outputOffset (layout c) (LogicalGrid.launch pid s) i<bounds outputRegion
    rw [LogicalGrid.offset c s pid hp (source c values) (destination c values) read i]
    simpa [copyIO,memory,outputChannels,destination,hp] using h

theorem schedule_safe (rest : Algebra α) (c : Config) (ps : List Nat)
    (values : Inputs α c) (bounds : RegionName → Nat)
    (bounded : (copyIO rest c ps).InBounds bounds values) (s : State α)
    (read : LogicalGrid.Bound c s (source c values) (destination c values))
    (members : ∀ p∈ps,p<c.count) : ScheduleSafe bounds (ps.map (invocation rest c)) s := by
  induction ps generalizing s with
  | nil => trivial
  | cons p ps ih =>
    constructor
    · exact invocation_safe rest c (p::ps) values bounds bounded s read p (members p (by simp))
    · intro next ran
      rw [invocation_run] at ran
      cases Option.some.inj ran
      apply ih
      · exact bounded
      · exact LogicalGrid.step_bound rest c s p (source c values) (destination c values) read
      · intro q hq;exact members q (by simp [hq])

/-- Complete shared IO judgment; the proof reuses the accepted source tile
execution and finite-grid composition. Only logical metadata conditions remain. -/
theorem implements (rest : Algebra α) (c : Config) (ps : List Nat)
    (order : ∀ p,p∈ps ↔ p<c.count) :
    (copyIO rest c ps).ImplementsOn (Admissible c) (target c) := by
  intro values admitted bounds bounded s input
  have read := reads rest c ps values admitted s input
  have unique := destination_unique c values admitted
  have members : ∀ p∈ps,p<c.count := fun p hp => (order p).mp hp
  refine ⟨schedule_safe rest c ps values bounds bounded s read members,
    LogicalGrid.run rest c ps s,schedule_run rest c ps s,?_,?_⟩
  · intro o j _
    fin_cases o
    rcases j with ⟨row,head,dim,⟨⟩⟩
    have h := LogicalGrid.run_read rest c ps s (source c values) (destination c values)
      read unique members row.val head.val dim.val ((order row.val).mpr row.isLt)
      head.isLt dim.isLt
    have h' : (LogicalGrid.run rest c ps s).mem outputRegion
        (flat c (destinationWord c values row).toNat head.val dim.val)=
        .mk .real (values ⟨0,by simp [copyIO,memory,inputChannels]⟩ (row,head,dim,PUnit.unit)) := by
      simpa only [destination,dif_pos row.isLt,source,dif_pos head.isLt,dif_pos dim.isLt] using h
    change mapCell (id : α → α) ((LogicalGrid.run rest c ps s).mem outputRegion
      (flat c (destinationWord c values row).toNat head.val dim.val))=_
    rw [h']
    rfl
  · intro r offset outside
    by_cases hr : r=outputRegion
    · subst r
      apply LogicalGrid.run_frame rest c ps s (source c values) (destination c values) read members
      intro pid hp head dim hh hd
      have row := (order pid).mp hp
      have h := outside.1 ⟨0,by simp [copyIO,memory,outputChannels]⟩
        (⟨pid,row⟩,⟨head,hh⟩,⟨dim,hd⟩,PUnit.unit) trivial
      simpa [copyIO,memory,outputChannels,destination,row] using h
    · exact LogicalGrid.other_frame rest c ps s r offset hr

#axiomsClean implements
end
end VeriTile.Bench.Repairs.DestindexCopyKv1.CopyIO
