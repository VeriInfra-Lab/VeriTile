import bench.tritonbench_g.triton_attention.Proofs.Grid
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Grid
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
variable {α : Type} [Inhabited α]

def Written (c : Shape) (r : RegionName) (o : Nat) : Prop :=
  (r ∈ (["Out","NewDO","DQ","DK","DV"] : List RegionName) ∧ o<c.elements) ∨
  (r ∈ (["L","M","Delta"] : List RegionName) ∧ o<c.groups*c.rows)
def Legal (grid pid : Program) : Prop := pid.1<grid.1 ∧ pid.2<grid.2

theorem row_bound (c : Shape) (g b lane : Nat) (hg : g<c.groups) (hb : b<c.blocks) (hl : lane<128) :
    g*c.rows+b*128+lane<c.groups*c.rows := by
  have hr : b*128+lane<c.rows := by change _<c.blocks*128; omega
  have hh:=Nat.mul_le_mul_right c.rows (Nat.succ_le_of_lt hg)
  nlinarith

theorem element_bound (c : Shape) (row d : Nat) (hr : row<c.groups*c.rows) (hd : d<c.dim) :
    row*c.dim+d<c.elements := by
  have hh:=Nat.mul_le_mul_right c.dim (Nat.succ_le_of_lt hr)
  change _<c.groups*c.rows*c.dim
  nlinarith

theorem forwardHead_eq (c : Shape) (scale : ℝ) (pid : Program) (mem : Memory α) :
    forwardHeadRow (forwardParameters c scale) (entry (forwardGrid c) pid mem)=pid.2*c.rows := by
  simp [forwardHeadRow,forwardParameters,entry,ids,c.dim_pos]

theorem forwardRow_bound (c : Shape) (scale : ℝ) (pid : Program) (mem : Memory α)
    (hp : Legal (forwardGrid c) pid) (i : FRow (forwardParameters c scale)) :
    forwardRowOffset (forwardParameters c scale) (entry (forwardGrid c) pid mem) i<c.groups*c.rows := by
  simpa [forwardRowOffset,forwardQueryRow,forwardParameters,entry,ids,Nat.add_assoc] using row_bound c _ _ _ hp.2 hp.1 i.1.isLt

theorem forwardOffset_bound (c : Shape) (scale : ℝ) (pid : Program) (mem : Memory α)
    (hp : Legal (forwardGrid c) pid) (i : FQuery (forwardParameters c scale)) :
    forwardOffset (forwardParameters c scale) (entry (forwardGrid c) pid mem) i<c.elements := by
  simp only [forwardOffset,forwardHead_eq]
  apply element_bound
  · simpa [forwardQueryRow,forwardParameters,entry,ids,Nat.add_assoc] using row_bound c _ _ _ hp.2 hp.1 i.1.isLt
  · exact i.2.1.isLt

theorem backwardHead_eq (c : Shape) (scale : ℝ) (pid : Program) (mem : Memory α) :
    backwardHeadRow (backwardParameters c scale) (entry (backwardGrid c) pid mem)=pid.1*c.rows := by
  change (pid.1/c.heads)*((c.heads*c.rows*c.dim)/c.dim)+(pid.1%c.heads)*((c.rows*c.dim)/c.dim)=_
  simp only [Nat.mul_div_cancel _ c.dim_pos]
  calc
    _=(pid.1/c.heads*c.heads+pid.1%c.heads)*c.rows := by ring
    _=pid.1*c.rows := by rw [Nat.div_add_mod']

theorem backwardOffset_bound (c : Shape) (scale : ℝ) (pid : Program) (mem : Memory α)
    (hp : Legal (backwardGrid c) pid) (b : Nat) (hb : b<c.blocks)
    (i : BLane (backwardParameters c scale)) :
    backwardOffset (backwardParameters c scale) (entry (backwardGrid c) pid mem) b i<c.elements := by
  simp only [backwardOffset,backwardHead_eq]
  apply element_bound
  · exact row_bound c _ _ _ hp.1 hb i.1.isLt
  · exact i.2.1.isLt

theorem forwardStep_frame (P : Primitives α) (c : Shape) (scale : ℝ) (pid : Program) (mem : Memory α)
    (hp : Legal (forwardGrid c) pid) (r : RegionName) (o : Nat) (h : ¬Written c r o) :
    forwardStep P c scale pid mem r o=mem r o := by
  unfold forwardStep forwardWrite
  rw [maskedScatter_frame]
  · rw [State.scatter_frame]
    · apply State.scatter_frame
      by_cases hr : r=(forwardParameters c scale).denominator
      · right
        intro i _ he
        exact h (Or.inr ⟨by simp [hr,forwardParameters],by simpa only [he] using forwardRow_bound c scale pid mem hp i⟩)
      · exact Or.inl hr
    · by_cases hr : r=(forwardParameters c scale).maximum
      · right
        intro i _ he
        exact h (Or.inr ⟨by simp [hr,forwardParameters],by simpa only [he] using forwardRow_bound c scale pid mem hp i⟩)
      · exact Or.inl hr
  · by_cases hr : r=(forwardParameters c scale).output
    · right
      intro i _ _ he
      exact h (Or.inl ⟨by simp [hr,forwardParameters],by simpa only [he] using forwardOffset_bound c scale pid mem hp i⟩)
    · exact Or.inl hr

theorem backwardStep_frame (P : Primitives α) (c : Shape) (scale : ℝ) (pid : Program) (mem : Memory α)
    (hp : Legal (backwardGrid c) pid) (r : RegionName) (o : Nat) (h : ¬Written c r o) :
    backwardStep P c scale pid mem r o=mem r o := by
  apply backwardFrame
  intro hw
  apply h
  left
  rcases hw with ⟨hr,b,hb,i,ho⟩|⟨hr,b,hb,i,_,ho⟩
  · exact ⟨by simp [hr,backwardParameters],by simpa only [ho] using backwardOffset_bound c scale pid mem hp b hb i⟩
  · rcases hr with hr|hr <;>
      exact ⟨by simp [hr,backwardParameters],by simpa only [ho] using backwardOffset_bound c scale pid mem hp b hb i⟩


theorem preRow_bound (c : Shape) (pid : Program) (mem : Memory α)
    (hp : Legal (preprocessGrid c) pid) (i : PreRow (preParameters c)) :
    preRow (preParameters c) (entry (preprocessGrid c) pid mem) i<c.groups*c.rows := by
  have hi:=i.1.isLt
  have hh:=hp.1
  change pid.1<c.blocks*c.groups at hh
  change i.1.val<128 at hi
  change pid.1*128+i.1.val<c.groups*(c.blocks*128)
  have hm:=Nat.mul_le_mul_right 128 (Nat.succ_le_of_lt hh)
  nlinarith

theorem preOffset_bound (c : Shape) (pid : Program) (mem : Memory α)
    (hp : Legal (preprocessGrid c) pid) (i : PreLane (preParameters c)) :
    preOffset (preParameters c) (entry (preprocessGrid c) pid mem) i<c.elements := by
  exact element_bound c _ _ (preRow_bound c pid mem hp (i.1,())) i.2.1.isLt

theorem preprocessStep_frame (P : Primitives α) (c : Shape) (pid : Program) (mem : Memory α)
    (hp : Legal (preprocessGrid c) pid) (r : RegionName) (o : Nat) (h : ¬Written c r o) :
    preprocessStep P c pid mem r o=mem r o := by
  unfold preprocessStep preWrite
  rw [State.scatter_frame]
  · apply State.scatter_frame
    by_cases hr : r=(preParameters c).newGradient
    · right
      intro i _ he
      exact h (Or.inl ⟨by simp [hr,preParameters],by simpa only [he] using preOffset_bound c pid mem hp i⟩)
    · exact Or.inl hr
  · by_cases hr : r=(preParameters c).delta
    · right
      intro i _ he
      exact h (Or.inr ⟨by simp [hr,preParameters],by simpa only [he] using preRow_bound c pid mem hp i⟩)
    · exact Or.inl hr

theorem fold_frame (step : Program → Memory α → Memory α) (work : List Program) (mem : Memory α)
    (r : RegionName) (o : Nat) (h : ∀ pid∈work,∀ m,step pid m r o=m r o) :
    (work.foldl (fun m p=>step p m) mem) r o=mem r o := by
  induction work generalizing mem with
  | nil=>rfl
  | cons p ps ih=>
    rw [List.foldl_cons,ih _ (by intro pid hp m;exact h pid (by simp [hp]) m)]
    exact h p (by simp) mem

theorem wholeReference_frame (F P B : Primitives α) (c : Shape) (scale : ℝ)
    (fw pw bw : List Program) (mem : Memory α)
    (hf : ∀ pid∈fw,Legal (forwardGrid c) pid)
    (hp : ∀ pid∈pw,Legal (preprocessGrid c) pid)
    (hb : ∀ pid∈bw,Legal (backwardGrid c) pid)
    (r : RegionName) (o : Nat) (h : ¬Written c r o) :
    wholeReference F P B c scale fw pw bw mem r o=mem r o := by
  unfold wholeReference
  rw [fold_frame _ bw _ r o (by intro pid hpid m;exact backwardStep_frame B c scale pid m (hb pid hpid) r o h)]
  rw [fold_frame _ pw _ r o (by intro pid hpid m;exact preprocessStep_frame P c pid m (hp pid hpid) r o h)]
  exact fold_frame _ fw mem r o (by intro pid hpid m;exact forwardStep_frame F c scale pid m (hf pid hpid) r o h)

theorem whole_execution_frame (M : Algebra α) (F P B : Primitives α) (c : Shape) (scale : ℝ)
    (fw pw bw : List Program) (mem : Memory α)
    (hf : ∀ pid∈fw,Legal (forwardGrid c) pid)
    (hp : ∀ pid∈pw,Legal (preprocessGrid c) pid)
    (hb : ∀ pid∈bw,Legal (backwardGrid c) pid) :
    ∃ final,wholeRun M F P B c scale fw pw bw mem=some final ∧
      ∀ r o,¬Written c r o → final r o=mem r o := by
  exact ⟨_,whole_success M F P B c scale fw pw bw mem,wholeReference_frame F P B c scale fw pw bw mem hf hp hb⟩

#axiomsClean preprocessStep_frame
#axiomsClean whole_execution_frame

#axiomsClean forwardStep_frame
#axiomsClean backwardStep_frame
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Grid
