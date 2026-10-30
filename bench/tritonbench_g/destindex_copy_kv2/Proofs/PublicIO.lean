import bench.tritonbench_g.destindex_copy_kv2.IO
import bench.tritonbench_g.destindex_copy_kv2.Proofs.LogicalGrid
import bench.tritonbench_g.destindex_copy_kv1.Proofs.PublicIO

namespace VeriTile.Bench.Repairs.DestindexCopyKv2.CopyIO
open VeriTile.Triton FP.Structural
open DestindexCopyKv1.CopyIO (inputChannels outputChannels memory source destination destinationWord)
noncomputable section
set_option maxHeartbeats 5000000
variable {α : Type} [Inhabited α]

theorem invocation_run (rest : Algebra α) (c : Config) (pid : Nat) (s : State α) :
    (invocation rest c pid).run s = some (LogicalGrid.step rest c s pid) := by
  simpa only [invocation,Invocation.run,Invocation.initial,bindRegisters,List.foldl_nil,
    Option.getD_none,LogicalGrid.launch,DestindexCopyKv1.LogicalGrid.launch,LogicalGrid.step] using
    (Tile.execution rest (layout c) (LogicalGrid.launch pid s)).property

theorem schedule_run (rest : Algebra α) (c : Config) (ps : List Nat) (s : State α) :
    runSchedule (ps.map (invocation rest c)) s = some (LogicalGrid.run rest c ps s) := by
  induction ps generalizing s with
  | nil => rfl
  | cons p ps ih =>
    simp only [List.map_cons,runSchedule,invocation_run,Option.bind]
    exact ih _

theorem invocation_safe (rest : Algebra α) (c : Config) (width : block c.dims = c.dims)
    (ps : List Nat) (values : Inputs α c) (bounds : RegionName → Nat)
    (bounded : (copyIO rest c ps).InBounds bounds values) (s : State α)
    (read : LogicalGrid.Bound c s (source c values) (destination c values))
    (pid : Nat) (hp : pid < c.count) : (invocation rest c pid).Safe bounds s := by
  apply Tile.source_trace_safe
  · exact bounded.2.1 ⟨1,by simp [copyIO,memory,inputChannels]⟩ (⟨pid,hp⟩,PUnit.unit) trivial
  · intro i active
    have hd : i.2.1.val < c.dims := by
      have := i.2.1.isLt
      change i.2.1.val < block c.dims at this
      omega
    have h := bounded.2.1 ⟨0,by simp [copyIO,memory,inputChannels]⟩
      (⟨pid,hp⟩,⟨i.1.val,active⟩,⟨i.2.1.val,hd⟩,PUnit.unit) trivial
    simpa [copyIO,memory,inputChannels,invocation,Invocation.initial,bindRegisters,
      Tile.inputOffset,layout,DestindexCopyKv1.layout,flat,DestindexCopyKv1.flat,Nat.mul_comm] using h
  · intro i active
    have hd : i.2.1.val < c.dims := by
      have := i.2.1.isLt
      change i.2.1.val < block c.dims at this
      omega
    have h := bounded.2.2.1 ⟨0,by simp [copyIO,memory,outputChannels]⟩
      (⟨pid,hp⟩,⟨i.1.val,active⟩,⟨i.2.1.val,hd⟩,PUnit.unit) trivial
    change Tile.outputOffset (layout c) (LogicalGrid.launch pid s) i < bounds DestindexCopyKv1.outputRegion
    rw [LogicalGrid.offset c s pid hp (source c values) (destination c values) read i]
    simpa [copyIO,memory,outputChannels,destination,hp] using h

theorem schedule_safe (rest : Algebra α) (c : Config) (width : block c.dims = c.dims)
    (ps : List Nat) (values : Inputs α c) (bounds : RegionName → Nat)
    (bounded : (copyIO rest c ps).InBounds bounds values) (s : State α)
    (read : LogicalGrid.Bound c s (source c values) (destination c values))
    (members : ∀ p∈ps,p<c.count) : ScheduleSafe bounds (ps.map (invocation rest c)) s := by
  induction ps generalizing s with
  | nil => trivial
  | cons p ps ih =>
    constructor
    · exact invocation_safe rest c width (p::ps) values bounds bounded s read p (members p (by simp))
    · intro next ran
      rw [invocation_run] at ran
      cases Option.some.inj ran
      apply ih
      · exact bounded
      · exact LogicalGrid.step_bound rest c width s p (source c values) (destination c values) read
      · intro q hq
        exact members q (by simp [hq])

/-- Complete actual kv2 source grid and logical row-copy result. Source safety
uses the explicit width condition; finite value/frame induction is reused from
kv1 only after the actual source effects are proved equal. -/
theorem implements (rest : Algebra α) (c : Config) (width : block c.dims = c.dims)
    (ps : List Nat) (order : ∀ p,p∈ps ↔ p<c.count) :
    (copyIO rest c ps).ImplementsOn (Admissible c) (target c) := by
  intro values admitted bounds bounded s input
  have read := DestindexCopyKv1.CopyIO.reads rest c ps values admitted s input
  have unique := DestindexCopyKv1.CopyIO.destination_unique c values admitted
  have members : ∀ p∈ps,p<c.count := fun p hp => (order p).mp hp
  refine ⟨schedule_safe rest c width ps values bounds bounded s read members,
    LogicalGrid.run rest c ps s, schedule_run rest c ps s, ?_, ?_⟩
  · intro o j _
    fin_cases o
    rcases j with ⟨row,head,dim,⟨⟩⟩
    have h := DestindexCopyKv1.LogicalGrid.run_read rest c ps s (source c values) (destination c values)
      read unique members row.val head.val dim.val ((order row.val).mpr row.isLt) head.isLt dim.isLt
    have h' : (LogicalGrid.run rest c ps s).mem DestindexCopyKv1.outputRegion
        (flat c (destinationWord c values row).toNat head.val dim.val) =
        .mk .real (values ⟨0,by simp [copyIO,memory,inputChannels]⟩ (row,head,dim,PUnit.unit)) := by
      rw [LogicalGrid.run_memory rest c width ps s s rfl]
      simpa only [destination,dif_pos row.isLt,source,dif_pos head.isLt,dif_pos dim.isLt] using h
    change mapCell (id : α → α) ((LogicalGrid.run rest c ps s).mem DestindexCopyKv1.outputRegion
      (flat c (destinationWord c values row).toNat head.val dim.val)) = _
    rw [h']
    rfl
  · intro r offset outside
    rw [LogicalGrid.run_memory rest c width ps s s rfl]
    by_cases hr : r = DestindexCopyKv1.outputRegion
    · subst r
      apply DestindexCopyKv1.LogicalGrid.run_frame rest c ps s (source c values) (destination c values) read members
      intro pid hp head dim hh hd
      have row := (order pid).mp hp
      have h := outside.1 ⟨0,by simp [copyIO,memory,outputChannels]⟩
        (⟨pid,row⟩,⟨head,hh⟩,⟨dim,hd⟩,PUnit.unit) trivial
      simpa [copyIO,memory,outputChannels,destination,row] using h
    · exact DestindexCopyKv1.LogicalGrid.other_frame rest c ps s r offset hr

#axiomsClean implements
end
end VeriTile.Bench.Repairs.DestindexCopyKv2.CopyIO
