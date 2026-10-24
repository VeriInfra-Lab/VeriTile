import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.HostGrid

namespace VeriTile.Bench.Repairs.FastCeLoss.HostStages
open Source Grid
noncomputable section
set_option maxHeartbeats 3000000

def window (t : Tensor) (rows : Nat) (p : Address) : Prop :=
  p.1=t.storage ∧ t.offset≤p.2 ∧ p.2<t.offset+rows
instance (t : Tensor) (rows : Nat) : DecidablePred (window t rows) := by
  intro p
  unfold window
  infer_instance

/-- Ordinary tensor pointwise store, including every active output coordinate.
Values are scalar source expressions; this is not a whole-result contract. -/
def writeRows {α : Type} (t : Tensor) (rows : Nat) (values : Nat → α) (m : Memory α) : Memory α :=
  fun p => if window t rows p then values (p.2-t.offset) else m p

theorem write_here {α : Type} (t : Tensor) (rows row : Nat) (values : Nat → α)
    (m : Memory α) (active : row<rows) : writeRows t rows values m (addr t row)=values row := by
  simp [writeRows,window,addr,active]

theorem write_other {α : Type} (t other : Tensor) (rows row : Nat) (values : Nat → α)
    (m : Memory α) (different : other.storage≠t.storage) :
    writeRows t rows values m (addr other row)=m (addr other row) := by
  simp [writeRows,window,addr,different]

def lseStage {α : Type} (a : Arithmetic α) (s : Shape) (b : Buffers) (m : Memory α) : Memory α :=
  writeRows b.lse s.rows (globalLSE a s b m) m

def addStage {α : Type} (a : Arithmetic α) (s : Shape) (b : Buffers) (m : Memory α) : Memory α :=
  writeRows b.losses s.rows (fun row => a.hostAdd (a.load (m (addr b.losses row)))
    (a.load (m (addr b.lse row)))) m

def maskStage {α : Type} (a : Arithmetic α) (s : Shape) (b : Buffers) (labels : Nat → Int)
    (m : Memory α) : Memory α :=
  writeRows b.losses s.rows (fun row => if labels row = -100 then a.literal 0 else m (addr b.losses row)) m

/-- Actual host order: complete logsumexp, complete add_, then masked_fill_.
The add expression exists on ignored rows as well, before mask overwrites it. -/
def finish {α : Type} (a : Arithmetic α) (s : Shape) (b : Buffers) (labels : Nat → Int)
    (m : Memory α) : Memory α :=
  if s.chunks=1 then m else maskStage a s b labels (addStage a s b (lseStage a s b m))

theorem lse_output {α : Type} (a : Arithmetic α) (s : Shape) (b : Buffers) (labels : Nat → Int)
    (m : Memory α) (sep : HostSeparated b) (chunked : s.chunks≠1) (row : Nat) (hr : row<s.rows) :
    finish a s b labels m (addr b.lse row)=globalLSE a s b m row := by
  simp only [finish,if_neg chunked,maskStage,addStage,lseStage]
  rw [write_other _ _ _ _ _ _ sep.lseLosses,write_other _ _ _ _ _ _ sep.lseLosses,
    write_here _ _ _ _ _ hr]

theorem loss_output {α : Type} (a : Arithmetic α) (s : Shape) (b : Buffers) (labels : Nat → Int)
    (m : Memory α) (sep : HostSeparated b) (chunked : s.chunks≠1) (row : Nat) (hr : row<s.rows) :
    finish a s b labels m (addr b.losses row)=
      if labels row = -100 then a.literal 0
      else a.hostAdd (a.load (m (addr b.losses row))) (a.load (globalLSE a s b m row)) := by
  simp only [finish,if_neg chunked,maskStage,addStage,lseStage]
  rw [write_here _ _ _ _ _ hr]
  split_ifs
  · rfl
  · rw [write_here _ _ _ _ _ hr,write_other _ _ _ _ _ _ sep.lseLosses.symm,
      write_here _ _ _ _ _ hr]

theorem frame {α : Type} (a : Arithmetic α) (s : Shape) (b : Buffers) (labels : Nat → Int)
    (m : Memory α) (p : Address) (hl : ¬window b.lse s.rows p) (hy : ¬window b.losses s.rows p) :
    finish a s b labels m p=m p := by
  unfold finish
  split_ifs
  · rfl
  · simp only [maskStage,addStage,lseStage,writeRows,if_neg hl,if_neg hy]

theorem window_coordinate (t : Tensor) (rows : Nat) (p : Address) (h : window t rows p) :
    ∃ row<rows, p=addr t row := by
  refine ⟨p.2-t.offset,?_,?_⟩
  · dsimp [window] at h
    omega
  · apply Prod.ext h.1
    dsimp [addr,window] at *
    omega

/-- The row-composed model is a proved final-state abstraction of the actual
three host tensor operations. It is not claimed to be their event trace. -/
theorem state_equivalence {α : Type} (a : Arithmetic α) (s : Shape) (b : Buffers)
    (labels : Nat → Int) (m : Memory α) (sep : HostSeparated b) :
    finish a s b labels m=finishGrid a s b labels m := by
  by_cases single : s.chunks=1
  · simp only [finish,finishGrid,if_pos single]
  · funext p
    by_cases hl : window b.lse s.rows p
    · rcases window_coordinate _ _ _ hl with ⟨row,hr,rfl⟩
      rw [lse_output a s b labels m sep single row hr,
        Grid.finish_output a s b labels m sep single row hr _ (Or.inl rfl)]
      simp [finishValues]
    · by_cases hy : window b.losses s.rows p
      · rcases window_coordinate _ _ _ hy with ⟨row,hr,rfl⟩
        rw [loss_output a s b labels m sep single row hr,
          Grid.finish_output a s b labels m sep single row hr _ (Or.inr rfl)]
        have different : addr b.losses row≠addr b.lse row := by
          intro h
          exact sep.lseLosses (congrArg Prod.fst h.symm)
        simp [finishValues,different,single]
      · rw [frame a s b labels m p hl hy,Grid.finish_frame a s b labels m sep]
        intro row hr writes
        rcases writes with h|h
        · subst p
          apply hl
          simp [window,addr,hr]
        · subst p
          apply hy
          simp [window,addr,hr]

#axiomsClean state_equivalence
#axiomsClean lse_output
#axiomsClean loss_output
#axiomsClean frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.HostStages
