import bench.tritonbench_g.triton_attention.Proofs.MemoryAgreement

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.MemoryAgreement
open VeriTile Triton FP.Structural Source
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 4000000
set_option maxRecDepth 32000

def InnerAgree (limit : Nat) {p : BackwardParameters} (a b : BackwardInner α p) : Prop :=
  Agree limit a.mem b.mem ∧ a.dk=b.dk ∧ a.dv=b.dv

theorem backward_inner_advance (P : Primitives α) (c : Shape) (scale : ℝ) (pid : Grid.Program)
    (legal : Grid.Legal (Grid.backwardGrid c) pid) (initial : Memory α)
    (key value : Values α .fp32 [128,c.dim]) (block query : Nat) (live : query<c.blocks)
    (a b : BackwardInner α (backwardParameters c scale)) (h : InnerAgree c.elements a b) :
    InnerAgree c.elements
      (backwardInnerAdvance P (backwardParameters c scale) (Grid.entry (Grid.backwardGrid c) pid initial) key value block query a)
      (backwardInnerAdvance P (backwardParameters c scale) (Grid.entry (Grid.backwardGrid c) pid initial) key value block query b) := by
  let p := backwardParameters c scale
  let s := Grid.entry (Grid.backwardGrid c) pid initial
  have reads (r : RegionName) : backwardRead p s a.mem r query=backwardRead p s b.mem r query :=
    backward_read c scale pid legal initial a.mem b.mem h.1 r query live
  have rows (r : RegionName) (i : BRow p) :
      a.mem r (backwardRowOffset p s query i)=b.mem r (backwardRowOffset p s query i) :=
    h.1 r _ (lt_of_lt_of_le (SourceAccess.backward_row_bound c scale s legal.1 query live i) (row_capacity c))
  have dq (i : BLane p) : a.mem p.dq (backwardOffset p s query i)=b.mem p.dq (backwardOffset p s query i) :=
    h.1 p.dq _ (SourceAccess.backward_element_bound c scale s legal.1 query live i)
  have weights : backwardWeights P p s a.mem key block query=backwardWeights P p s b.mem key block query := by
    funext i
    simp only [backwardWeights,backwardMaskedQK,backwardQK,reads,rows]
  have ds : backwardDS P p s a.mem key value block query=backwardDS P p s b.mem key value block query := by
    funext i
    simp only [backwardDS,backwardDP,backwardDPInitial,backwardDPDot,weights,reads,rows]
  have keyDot : backwardKeyDot P p s a.mem key value block query=backwardKeyDot P p s b.mem key value block query := by
    funext i
    simp only [backwardKeyDot,ds,reads]
  have valueDot : backwardValueDot P p s a.mem key block query=backwardValueDot P p s b.mem key block query := by
    funext i
    simp only [backwardValueDot,weights,reads]
  have cells : backwardQueryCell P p s a.mem key value block query=backwardQueryCell P p s b.mem key value block query := by
    funext i
    simp only [backwardQueryCell,backwardQueryValue,backwardQueryDot,ds,dq]
  constructor
  · change Agree c.elements
      ((TileShape.allIndices [p.blockM,p.head]).foldl (fun m i=>writeMemory m p.dq
        (backwardOffset p s query i) (backwardQueryCell P p s a.mem key value block query i)) a.mem)
      ((TileShape.allIndices [p.blockM,p.head]).foldl (fun m i=>writeMemory m p.dq
        (backwardOffset p s query i) (backwardQueryCell P p s b.mem key value block query i)) b.mem)
    rw [cells]
    exact scatter c.elements a.mem b.mem h.1 _ _ _ _
  constructor
  · change (fun i=>P.binary (some .fp32) .real .add (a.dk i)
      (P.cast (some .fp32) .fp32 .real (backwardKeyDot P p s a.mem key value block query i)))=_
    rw [h.2.1,keyDot]
    rfl
  · change (fun i=>P.binary (some .fp32) .real .add (a.dv i)
      (P.cast (some .fp32) .fp32 .real (backwardValueDot P p s a.mem key block query i)))=_
    rw [h.2.2,valueDot]
    rfl

theorem backward_inner_rounds (P : Primitives α) (c : Shape) (scale : ℝ) (pid : Grid.Program)
    (legal : Grid.Legal (Grid.backwardGrid c) pid) (initial : Memory α)
    (a b : Memory α) (h : Agree c.elements a b) (block : Nat) (live : block<c.blocks)
    (n : Nat) (hn : block+n≤c.blocks) :
    InnerAgree c.elements
      (backwardInnerRounds P (backwardParameters c scale) (Grid.entry (Grid.backwardGrid c) pid initial) a block n)
      (backwardInnerRounds P (backwardParameters c scale) (Grid.entry (Grid.backwardGrid c) pid initial) b block n) := by
  induction n with
  | zero => exact ⟨h,rfl,rfl⟩
  | succ n ih =>
    have key := backward_read c scale pid legal initial a b h (backwardParameters c scale).k block live
    have value := backward_read c scale pid legal initial a b h (backwardParameters c scale).v block live
    simp only [backwardInnerRounds]
    rw [key,value]
    exact backward_inner_advance P c scale pid legal initial _ _ block (block+n) (by omega)
      _ _ (ih (by omega))

theorem backward_outer (P : Primitives α) (c : Shape) (scale : ℝ) (pid : Grid.Program)
    (legal : Grid.Legal (Grid.backwardGrid c) pid) (initial : Memory α)
    (a b : Memory α) (h : Agree c.elements a b) (block : Nat) (live : block<c.blocks) :
    Agree c.elements
      (backwardOuterAdvance P (backwardParameters c scale) (Grid.entry (Grid.backwardGrid c) pid initial) block a)
      (backwardOuterAdvance P (backwardParameters c scale) (Grid.entry (Grid.backwardGrid c) pid initial) block b) := by
  have inner := backward_inner_rounds P c scale pid legal initial a b h block live (c.blocks-block) (by omega)
  unfold backwardOuterAdvance
  dsimp only []
  simp only [show (backwardParameters c scale).blocks=c.blocks from rfl]
  rw [inner.2.1,inner.2.2]
  apply maskedScatter
  apply maskedScatter
  exact inner.1

theorem backward_step (P : Primitives α) (c : Shape) (scale : ℝ) (pid : Grid.Program)
    (legal : Grid.Legal (Grid.backwardGrid c) pid) (a b : Memory α) (h : Agree c.elements a b) :
    Agree c.elements (Grid.backwardStep P c scale pid a) (Grid.backwardStep P c scale pid b) := by
  have innerRebase (mem : Memory α) (block n : Nat) :
      backwardInnerRounds P (backwardParameters c scale) (Grid.entry (Grid.backwardGrid c) pid b) mem block n=
        backwardInnerRounds P (backwardParameters c scale) (Grid.entry (Grid.backwardGrid c) pid a) mem block n := by
    induction n with
    | zero=>rfl
    | succ n ih=>
      rw [backwardInnerRounds,backwardInnerRounds,ih]
      rfl
  have rebase (mem : Memory α) (block : Nat) :
      backwardOuterAdvance P (backwardParameters c scale) (Grid.entry (Grid.backwardGrid c) pid b) block mem=
        backwardOuterAdvance P (backwardParameters c scale) (Grid.entry (Grid.backwardGrid c) pid a) block mem := by
    unfold backwardOuterAdvance
    rw [innerRebase]
    rfl
  have rounds (n : Nat) (hn : n≤c.blocks) :
      Agree c.elements
        (backwardRounds P (backwardParameters c scale) (Grid.entry (Grid.backwardGrid c) pid a) n)
        (backwardRounds P (backwardParameters c scale) (Grid.entry (Grid.backwardGrid c) pid b) n) := by
    induction n with
    | zero => exact h
    | succ n ih =>
      rw [backwardRounds,backwardRounds,rebase]
      exact backward_outer P c scale pid legal a _ _ (ih (by omega)) n (by omega)
  exact rounds c.blocks (Nat.le_refl _)

#axiomsClean backward_step
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.MemoryAgreement
