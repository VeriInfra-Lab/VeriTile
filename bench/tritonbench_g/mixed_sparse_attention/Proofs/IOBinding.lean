import bench.tritonbench_g.mixed_sparse_attention.IO

namespace VeriTile.Bench.Repairs.MixedSparseAttention.KernelIO
open VeriTile Triton FP.Structural
open Whole002 Source Whole Checked
noncomputable section
variable {α : Type} [Inhabited α]

/-- Simultaneous decoded input binding retains all read-only aliases and the
original default semantics for cells whose raw tags differ from a role. -/
structure Reads (c : Config) (xs : Values α c) (s : State α) : Prop where
  query : ∀ off,(s.mem c.q off).read .real=query c xs off
  key : ∀ off,(s.mem c.k off).read .real=key c xs off
  value : ∀ off,(s.mem c.v off).read .real=value c xs off
  length : ∀ off,(s.mem (Region.cast c.seq) off).read .nat=length c xs off
  blocks : ∀ off,(s.mem (Region.cast c.blocks) off).read .nat=blocks c xs off
  offsets : ∀ off,(s.mem (Region.cast c.offsets) off).read .nat=offsets c xs off
  counts : ∀ off,(s.mem (Region.cast c.counts) off).read .nat=counts c xs off
  columns : ∀ off,(s.mem (Region.cast c.columns) off).read .nat=columns c xs off

theorem input_reads (M : Algebra α) (D : Dot α) (c : Config) (work : List Program)
    (xs : Values α c) (s : State α) (h : (io M D c work).InputBinding xs s) : Reads c xs s := by
  exact ⟨h (0 : Fin 8),h (1 : Fin 8),h (2 : Fin 8),h (3 : Fin 8),h (4 : Fin 8),h (5 : Fin 8),h (6 : Fin 8),h (7 : Fin 8)⟩

theorem Reads.inputBinding (M : Algebra α) (D : Dot α) (c : Config) (work : List Program)
    (xs : Values α c) (s : State α) (h : Reads c xs s) :
    (io M D c work).InputBinding xs s := by
  intro i off
  fin_cases i
  all_goals first | exact h.query off | exact h.key off | exact h.value off |
    exact h.length off | exact h.blocks off | exact h.offsets off | exact h.counts off | exact h.columns off

theorem Reads.program (c : Config) (xs : Values α c) (s : State α) (h : Reads c xs s)
    (g : Whole.Grid) (p : Program) : Reads c xs (start g p s) :=
  ⟨h.query,h.key,h.value,h.length,h.blocks,h.offsets,h.counts,h.columns⟩

theorem Reads.stable (c : Config) (xs : Values α c) (s t : State α) (h : Reads c xs s)
    (sep : Separated c) (frame : ∀ r off,r≠c.out → t.mem r off=s.mem r off) : Reads c xs t := by
  constructor
  all_goals intro off
  · rw [frame c.q off (sep.inputs _ (by simp))];exact h.query off
  · rw [frame c.k off (sep.inputs _ (by simp))];exact h.key off
  · rw [frame c.v off (sep.inputs _ (by simp))];exact h.value off
  · rw [frame (Region.cast c.seq) off (sep.inputs _ (by simp))];exact h.length off
  · rw [frame (Region.cast c.blocks) off (sep.inputs _ (by simp))];exact h.blocks off
  · rw [frame (Region.cast c.offsets) off (sep.inputs _ (by simp))];exact h.offsets off
  · rw [frame (Region.cast c.counts) off (sep.inputs _ (by simp))];exact h.counts off
  · rw [frame (Region.cast c.columns) off (sep.inputs _ (by simp))];exact h.columns off

theorem Reads.lengths (c : Config) (xs : Values α c) (s : State α) (h : Reads c xs s)
    (hd : domain c xs) : LengthsBound c s := by
  intro batch hb
  rw [h.length]
  exact hd batch hb

theorem invocation_initial (M : Algebra α) (D : Dot α) (c : Config) (p : Program) (s : State α) :
    (invocation M D c p).initial s=start (grid c) p s := rfl

theorem schedule_run (M : Algebra α) (D : Dot α) (c : Config) (work : List Program) (s : State α) :
    runSchedule (work.map (invocation M D c)) s=
      runPrograms (fun _=>M) (fun _=>D) c (grid c) work s := by
  induction work generalizing s with
  | nil=>rfl
  | cons p ps ih=>
    simp only [List.map_cons,runSchedule,runPrograms,Invocation.run,invocation_initial]
    exact congrArg (Option.bind (FP.Structural.exec (backend M D) (kernel c) (start (grid c) p s)))
      (funext fun t=>ih t)

theorem Reads.selected (c : Config) (xs : Values α c) (s : State α) (h : Reads c xs s)
    (p : Program) (ph : Bool) (n : Nat) (j : Fin c.BN) :
    Source.columnValues c (start (grid c) p s) ph n j=selected c xs p ph n j := by
  simp [Source.columnValues,Source.blockCount,Source.columnCount,Source.metadataIndex,
    start,ids,h.blocks,h.counts,h.offsets,h.columns,KernelIO.selected,KernelIO.blockCount,KernelIO.columnCount,KernelIO.slot]

theorem Reads.keyActive (c : Config) (xs : Values α c) (s : State α) (h : Reads c xs s)
    (p : Program) (ph : Bool) (n : Nat) (j : Fin c.BN) :
    Source.keyActive c (start (grid c) p s) ph n j↔keyActive c xs p ph n j := by
  simp only [Source.keyActive,KernelIO.keyActive,Reads.selected c xs s h p ph n j]
  cases ph <;>
    simp [Source.columnValues,Source.sequenceLength,Source.blockCount,Source.columnCount,
      Source.metadataIndex,start,ids,h.blocks,h.counts,h.length,h.offsets,KernelIO.selected,KernelIO.sequence,
      KernelIO.blockCount,KernelIO.columnCount,KernelIO.slot]

#axiomsClean schedule_run
end
end VeriTile.Bench.Repairs.MixedSparseAttention.KernelIO
