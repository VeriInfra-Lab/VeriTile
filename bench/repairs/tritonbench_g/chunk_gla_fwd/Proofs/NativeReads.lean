import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeLaunch

/-! Finite reached-read locality for the unchanged native source recurrence.
Only actual masked operands are compared. Arbitrary exterior cells and their
Structural tags are retained; no global export/import equality is assumed. -/
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.NativeReads
open VeriTile Triton FP.Structural
open FPStorage (Memory state)
open FPStages (Stage diagonalLayout mergeLayout)
open NativeLaunch (start)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 30000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def Reads (stage : Stage) (s : State α) (m : Memory α) : Prop :=
  ∀ port∈NativeLaunch.inputExtents stage,∀ off,off<port.2 →
    if port.1="h" then (s.mem port.1 off).read .fp32=m port.1 off
    else (s.mem port.1 off).read .real=m port.1 off

theorem real_read {stage : Stage} {s : State α} {m : Memory α} (h : Reads stage s m)
    (r : RegionName) (n : Nat) (port : (r,n)∈NativeLaunch.inputExtents stage)
    (hr : r≠"h") (off : Nat) (ho : off<n) : (s.mem r off).read .real=m r off := by
  simpa only [if_neg hr] using h (r,n) port off ho

theorem history_read {stage : Stage} {s : State α} {m : Memory α} (h : Reads stage s m)
    (n : Nat) (port : (("h" : RegionName),n)∈NativeLaunch.inputExtents stage)
    (off : Nat) (ho : off<n) : (s.mem "h" off).read .fp32=m "h" off := by
  simpa only [if_pos rfl] using h ("h",n) port off ho

theorem fold_congr {β : Type} (f g : Nat → β → β) (first last : Nat) (a : β)
    (h : ∀ j,first≤j → j<last → ∀ b,f j b=g j b) :
    foldRange f first last 1 a=foldRange g first last 1 a := by
  conv_lhs => rw [foldRange]
  conv_rhs => rw [foldRange]
  simp only [Nat.one_ne_zero,↓reduceDIte]
  split
  · rename_i inside
    rw [h first (Nat.le_refl _) inside]
    exact fold_congr f g (first+1) last _ (fun j hj hj' => h j (by omega) hj')
  · rfl
termination_by last-first
decreasing_by omega

theorem inter_query (M : Algebra α) (c : ProducerGeometry.Shape) (scale : ℝ)
    (idx : GridIndex (ProducerGeometry.grid c)) (s : State α) (m : Memory α)
    (reads : Reads (.inter c scale) s m) (r : RegionName) (hr : r="q" ∨ r="g")
    (f : Nat) :
    FPInter.queryLoad M c (start (.inter c scale) idx s) r f=
      FPInter.queryLoad M c (state m idx.toPids (ProducerGeometry.grid c).toNumPids) r f := by
  funext i
  have port : (r,c.heads*c.T*c.K)∈NativeLaunch.inputExtents (.inter c scale) := by
    rcases hr with rfl|rfl <;> simp [NativeLaunch.inputExtents]
  have rh : r≠"h" := by rcases hr with rfl|rfl <;> decide
  dsimp only [FPInter.queryLoad,FPInter.row,FPInter.rowStart,start,state,Stage.grid]
  by_cases active : FPInter.row c (start (.inter c scale) idx s) i.1.val<c.T ∧
      FPInter.feature c f i.2.1.val<c.K
  · have hb := FPReadBounds.inter_query c idx m f i active.1 active.2
    dsimp only [FPInter.row,FPInter.rowStart,start,state,Stage.grid] at active hb ⊢
    rw [if_pos active,if_pos active]
    apply congrArg M.fp32Load
    exact (real_read reads r _ port rh _ hb).trans (FPStorage.import_real _ _ _ rh).symm
  · dsimp only [FPInter.row,FPInter.rowStart,start,state,Stage.grid] at active ⊢
    rw [if_neg active,if_neg active]

theorem inter_key (M : Algebra α) (c : ProducerGeometry.Shape) (scale : ℝ)
    (idx : GridIndex (ProducerGeometry.grid c)) (s : State α) (m : Memory α)
    (reads : Reads (.inter c scale) s m) (r : RegionName) (hr : r="k" ∨ r="g")
    (f : Nat) :
    FPInter.keyLoad M c (start (.inter c scale) idx s) r f=
      FPInter.keyLoad M c (state m idx.toPids (ProducerGeometry.grid c).toNumPids) r f := by
  funext i
  have port : (r,c.heads*c.T*c.K)∈NativeLaunch.inputExtents (.inter c scale) := by
    rcases hr with rfl|rfl <;> simp [NativeLaunch.inputExtents]
  have rh : r≠"h" := by rcases hr with rfl|rfl <;> decide
  dsimp only [FPInter.keyLoad,FPInter.columnStart,start,state,Stage.grid]
  by_cases active : FPInter.feature c f i.1.val<c.K ∧
      FPInter.columnStart c (start (.inter c scale) idx s)+i.2.1.val<c.T
  · have hb := FPReadBounds.inter_key c idx m f i active.1 active.2
    dsimp only [FPInter.columnStart,start,state,Stage.grid] at active hb ⊢
    rw [if_pos active,if_pos active]
    apply congrArg M.fp32Load
    exact (real_read reads r _ port rh _ hb).trans (FPStorage.import_real _ _ _ rh).symm
  · dsimp only [FPInter.columnStart,start,state,Stage.grid] at active ⊢
    rw [if_neg active,if_neg active]

theorem inter_reference (M : Algebra α) (c : ProducerGeometry.Shape) (scale : ℝ)
    (idx : GridIndex (ProducerGeometry.grid c)) (s : State α) (m : Memory α)
    (reads : Reads (.inter c scale) s m)
    (enabled : FPInter.enabled c (start (.inter c scale) idx s)) (f e : Nat) :
    FPInter.reference M c (start (.inter c scale) idx s) f e=
      FPInter.reference M c (state m idx.toPids (ProducerGeometry.grid c).toNumPids) f e := by
  dsimp only [FPInter.reference,FPInter.rowStart,start,state,Stage.grid]
  by_cases active : FPInter.feature c f e<c.K
  · have hb := FPReadBounds.inter_reference c idx m f e enabled active
    dsimp only [FPInter.rowStart,start,state,Stage.grid] at hb ⊢
    rw [if_pos active,if_pos active]
    apply congrArg M.fp32Load
    exact (real_read reads "g" _ (by simp [NativeLaunch.inputExtents]) (by decide) _ hb).trans
      (FPStorage.import_real _ _ _ (by decide)).symm
  · rw [if_neg active,if_neg active]

theorem inter_value (M : Algebra α) (D : FPOutput.DotPrimitive α)
    (c : ProducerGeometry.Shape) (scale : ℝ) (idx : GridIndex (ProducerGeometry.grid c))
    (s : State α) (m : Memory α) (reads : Reads (.inter c scale) s m)
    (enabled : FPInter.enabled c (start (.inter c scale) idx s)) :
    FPInter.value M D c scale (start (.inter c scale) idx s)=
      FPInter.value M D c scale (state m idx.toPids (ProducerGeometry.grid c).toNumPids) := by
  have hq (f : Nat) : FPInter.queryGate M c scale (start (.inter c scale) idx s) f=
      FPInter.queryGate M c scale (state m idx.toPids (ProducerGeometry.grid c).toNumPids) f := by
    funext i
    simp only [FPInter.queryGate,inter_query M c scale idx s m reads "q" (Or.inl rfl),
      inter_query M c scale idx s m reads "g" (Or.inr rfl),inter_reference M c scale idx s m reads enabled]
  have hk (f : Nat) : FPInter.keyGate M c (start (.inter c scale) idx s) f=
      FPInter.keyGate M c (state m idx.toPids (ProducerGeometry.grid c).toNumPids) f := by
    funext i
    simp only [FPInter.keyGate,inter_key M c scale idx s m reads "k" (Or.inl rfl),
      inter_key M c scale idx s m reads "g" (Or.inr rfl),inter_reference M c scale idx s m reads enabled]
  have update : FPInter.update M D c scale (start (.inter c scale) idx s)=
      FPInter.update M D c scale (state m idx.toPids (ProducerGeometry.grid c).toNumPids) := by
    funext f a i
    simp only [FPInter.update,hq,hk]
  funext i
  simp only [FPInter.value,FPInter.accumulated,update]

end
end VeriTile.Bench.Repairs.ChunkGlaFwd.NativeReads
