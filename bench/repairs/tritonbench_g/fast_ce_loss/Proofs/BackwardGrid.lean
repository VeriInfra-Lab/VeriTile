import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.ForwardGrid
import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.Geometry

namespace VeriTile.Bench.Repairs.FastCeLoss.Grid
open Source CrossEntropy1
noncomputable section
set_option maxHeartbeats 4000000

structure BackwardSeparated (b : Buffers) : Prop where
  logitsLSE : b.logits.storage≠b.lse.storage
  logitsUpstream : b.logits.storage≠b.upstream.storage

theorem backward_unique (s : Shape) (b : Buffers) (positive : 0<s.vocab)
    (q other : Task) (point : Address)
    (hq : backwardWrites s b q point) (ho : backwardWrites s b other point) : q=other := by
  dsimp only [backwardWrites] at hq ho
  have qrow : (point.2-b.logits.offset)/s.vocab=q.1 := by
    apply Geometry.owner_unique positive
    · omega
    · have top := Nat.min_le_right ((q.2+1)*4096) s.vocab
      rw [Nat.add_mul,Nat.one_mul]
      omega
  have orow : (point.2-b.logits.offset)/s.vocab=other.1 := by
    apply Geometry.owner_unique positive
    · omega
    · have top := Nat.min_le_right ((other.2+1)*4096) s.vocab
      rw [Nat.add_mul,Nat.one_mul]
      omega
  have rows : q.1=other.1 := qrow.symm.trans orow
  have qblock : (point.2-(b.logits.offset+q.1*s.vocab))/4096=q.2 := by
    apply Geometry.owner_unique (by decide)
    · omega
    · have top := Nat.min_le_left ((q.2+1)*4096) s.vocab
      omega
  have oblock : (point.2-(b.logits.offset+other.1*s.vocab))/4096=other.2 := by
    apply Geometry.owner_unique (by decide)
    · omega
    · have top := Nat.min_le_left ((other.2+1)*4096) s.vocab
      omega
  have blocks : q.2=other.2 := by rw [rows] at qblock; exact qblock.symm.trans oblock
  exact Prod.ext rows blocks

def backwardReads (s : Shape) (b : Buffers) (q : Task) (point : Address) : Prop :=
  backwardWrites s b q point ∨ point.1=b.lse.storage ∨ point.1=b.upstream.storage

theorem backward_frozen {α : Type} (a : Arithmetic α) (o : Options) (s : Shape)
    (b : Buffers) (labels : Nat → Int) (entry : Memory α)
    (sep : BackwardSeparated b) (positive : 0<s.vocab) :
    backwardGrid a o s b labels entry=
      Schedule.run (Schedule.put (backwardWrites s b) (backwardValues a o s b labels entry))
        (tasks s.rows s.blocks) entry := by
  apply Schedule.freeze_reads (reads:=backwardReads s b) (writes:=backwardWrites s b) (entry:=entry)
  · intro q current same
    funext point
    change (if backwardWrites s b q point then backwardValues a o s b labels current q point else current point) =
      (if backwardWrites s b q point then backwardValues a o s b labels entry q point else current point)
    split_ifs with active
    · have hx := same point (Or.inl active)
      have hl := same (addr b.lse q.1) (Or.inr (Or.inl rfl))
      have hu := same (addr b.upstream (q.1*b.upstreamStride)) (Or.inr (Or.inr rfl))
      simp only [backwardValues,hx,hl,hu]
    · rfl
  · intro q current point outside
    exact if_neg outside
  · exact CrossEntropy1.Grid.task_nodup _ _
  · intro q _ other _ different point read write
    rcases read with h|h|h
    · exact different (backward_unique s b positive q other point write h)
    · exact sep.logitsLSE (write.1.symm.trans h)
    · exact sep.logitsUpstream (write.1.symm.trans h)
  · intro _ _ _ _
    rfl

theorem backward_active (s : Shape) (b : Buffers) (row col : Nat) (hc : col<s.vocab) :
    backwardWrites s b (row,col/4096) (addr b.logits (row*s.vocab+col)) := by
  have owned := Geometry.backward_owner s col hc
  refine ⟨rfl,?_,?_⟩
  · dsimp [addr]
    omega
  · dsimp [addr]
    have upper : col<min ((col/4096+1)*4096) s.vocab := lt_min owned.2.2 hc
    omega

theorem backward_output {α : Type} (a : Arithmetic α) (o : Options) (s : Shape)
    (b : Buffers) (labels : Nat → Int) (entry : Memory α)
    (sep : BackwardSeparated b) (positive : 0<s.vocab)
    (row col : Nat) (hr : row<s.rows) (hc : col<s.vocab) :
    backwardGrid a o s b labels entry (addr b.logits (row*s.vocab+col))=
      backwardScalar a o (labels row) col (entry (addr b.logits (row*s.vocab+col)))
        (entry (addr b.lse row)) (entry (addr b.upstream (row*b.upstreamStride))) := by
  rw [backward_frozen a o s b labels entry sep positive]
  have present : (row,col/4096)∈tasks s.rows s.blocks :=
    CrossEntropy1.Grid.task_mem _ _ _ |>.mpr ⟨hr,(Geometry.backward_owner s col hc).1⟩
  have unique : ∀ q∈tasks s.rows s.blocks,
      backwardWrites s b q (addr b.logits (row*s.vocab+col)) → q=(row,col/4096) := by
    intro q _ hw
    exact backward_unique s b positive q _ _ hw (backward_active s b row col hc)
  rw [Schedule.fixed_output (backwardWrites s b) (backwardValues a o s b labels entry)
    (tasks s.rows s.blocks) entry (row,col/4096) _ (backward_active s b row col hc) unique,if_pos present]
  have offset : b.logits.offset+(row*s.vocab+col)-(b.logits.offset+row*s.vocab)=col := by omega
  simpa only [backwardValues,addr,offset]

theorem backward_frame {α : Type} (a : Arithmetic α) (o : Options) (s : Shape)
    (b : Buffers) (labels : Nat → Int) (entry : Memory α)
    (sep : BackwardSeparated b) (positive : 0<s.vocab) (point : Address)
    (outside : point.1≠b.logits.storage ∨ point.2<b.logits.offset ∨
      b.logits.offset+s.rows*s.vocab≤point.2) :
    backwardGrid a o s b labels entry point=entry point := by
  rw [backward_frozen a o s b labels entry sep positive]
  apply Schedule.fixed_frame
  intro q hq hw
  have rows := (CrossEntropy1.Grid.task_mem _ _ _).mp hq |>.1
  have top := Nat.min_le_right ((q.2+1)*4096) s.vocab
  have bound := Nat.mul_le_mul_right s.vocab (Nat.succ_le_of_lt rows)
  dsimp [backwardWrites] at hw
  rcases outside with outside|outside|outside
  · exact outside hw.1
  · omega
  · nlinarith

#axiomsClean backward_frozen
#axiomsClean backward_output
#axiomsClean backward_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.Grid
