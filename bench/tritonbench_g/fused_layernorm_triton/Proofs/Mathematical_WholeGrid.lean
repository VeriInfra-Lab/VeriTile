import bench.tritonbench_g.fused_layernorm_triton.Proofs.Mathematical_WholeRow

namespace VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole
open VeriTile.Triton FP.Structural FPStages
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def atRow {α : Type} (s : State α) (row : Nat) : State α :=
  { s with pids := fun dim => if dim = 0 then row else 0 }

/-- A generic dispatch fold. The theorem below allows any order and repetitions;
completion of a grid means its list contains every launched row. -/
def runRows {α : Type} [Inhabited α] (M : Algebra α) (kernel : ComputeKernel) :
    List Nat → State α → Option (State α)
  | [], s => some s
  | row :: rows, s => do
      let t ← FP.Structural.exec M kernel (atRow s row)
      runRows M kernel rows t

theorem lane_recurrence_congr {α : Type} [Inhabited α] (M : Algebra α)
    (s t : State α) (X : RegionName) (D B start count : Nat)
    (previous : Fin B → WelfordTriple α) (hm : s.mem X = t.mem X) (hp : s.pids 0 = t.pids 0) :
    laneRecurrence M s X D B start count previous = laneRecurrence M t X D B start count previous := by
  induction count generalizing start previous with
  | zero => rfl
  | succ n ih =>
    simp only [laneRecurrence]
    rw [ih]
    congr 1
    funext i
    simp only [advance, hm, hp]

theorem row_value_congr {α : Type} [Inhabited α] (M : Algebra α) (eq32 : α → α → Bool)
    (s t : State α) (X : RegionName) (D B row : Nat) (tree : ReductionTree B)
    (hm : s.mem X = t.mem X) :
    rowValue M eq32 (atRow s row) X D B tree = rowValue M eq32 (atRow t row) X D B tree := by
  unfold rowValue allLaneValues
  rw [lane_recurrence_congr M (atRow s row) (atRow t row) X D B 1 (blockCount D B - 1) _ hm rfl]
  have hf : firstLaneValues M (atRow s row) X D B = firstLaneValues M (atRow t row) X D B := by
    funext i
    simp [firstLaneValues, atRow, hm]
  rw [hf]

def rowOutput {α : Type} [Inhabited α] (M : Algebra α) (eq32 : α → α → Bool)
    (s : State α) (X W Bias : RegionName) (D B row col : Nat) (tree : ReductionTree B) : α :=
  normalizeValue M s X W Bias D row col (rowValue M eq32 (atRow s row) X D B tree).mean
    (rstdValue M D (rowValue M eq32 (atRow s row) X D B tree))

structure GridEffect {α : Type} [Inhabited α] (M : Algebra α) (eq32 : α → α → Bool)
    (s t : State α) (X W Bias Y Mean Rstd : RegionName) (D B : Nat)
    (tree : ReductionTree B) (rows : List Nat) : Prop where
  mean : ∀ row ∈ rows, t.mem Mean row = .mk .real (rowValue M eq32 (atRow s row) X D B tree).mean
  rstd : ∀ row ∈ rows, t.mem Rstd row = .mk .real (rstdValue M D (rowValue M eq32 (atRow s row) X D B tree))
  output : ∀ row ∈ rows, ∀ col, col < D → t.mem Y (col + D * row) =
    .mk .bf16 (rowOutput M eq32 s X W Bias D B row col tree)
  frame : ∀ q off, (∀ row ∈ rows, rowMiss Y Mean Rstd D row q off) → t.mem q off = s.mem q off

theorem whole_grid {α : Type} [Inhabited α] (M : Algebra α)
    (eq32 : α → α → Bool) (s : State α) (X W Bias Y Mean Rstd : RegionName)
    (S D B : Nat) (tree : ReductionTree B) (hB : 0 < B)
    (sep : Separation X W Bias Y Mean Rstd) (rows : List Nat) :
    ∃ final, runRows (withEquality M eq32) (rowKernel X W Bias Y Mean Rstd S D B tree) rows s = some final ∧
      GridEffect M eq32 s final X W Bias Y Mean Rstd D B tree rows := by
  induction rows generalizing s with
  | nil =>
    refine ⟨s, rfl, ?_, ?_, ?_, ?_⟩
    · simp
    · simp
    · simp
    · intro q off h; rfl
  | cons row rows ih =>
    obtain ⟨u, hu, he⟩ := whole_row M eq32 (atRow s row) X W Bias Y Mean Rstd S D B tree hB sep
    obtain ⟨t, ht, hf⟩ := ih u
    have hinput : ∀ q, q = X ∨ q = W ∨ q = Bias → u.mem q = s.mem q := by
      intro q hq
      funext off
      exact he.frame q off ⟨.inl (sep.inputs q hq).1, .inl (sep.inputs q hq).2.1, .inl (sep.inputs q hq).2.2⟩
    have hv : ∀ r, rowValue M eq32 (atRow u r) X D B tree = rowValue M eq32 (atRow s r) X D B tree :=
      fun r => row_value_congr M eq32 u s X D B r tree (hinput X (.inl rfl))
    have ho : ∀ r col, rowOutput M eq32 u X W Bias D B r col tree = rowOutput M eq32 s X W Bias D B r col tree := by
      intro r col
      simp only [rowOutput, hv, normalizeValue, hinput X (.inl rfl),
        hinput W (.inr (.inl rfl)), hinput Bias (.inr (.inr rfl))]
    refine ⟨t, by simp [runRows, hu, ht], ?_, ?_, ?_, ?_⟩
    · intro r hr
      by_cases hin : r ∈ rows
      · simpa only [hv] using hf.mean r hin
      · have hrw : r = row := by simpa [hin] using hr
        subst r
        rw [hf.frame Mean row ?_]
        · simpa [atRow] using he.mean
        · intro r hr
          exact ⟨.inr (fun hh => hin (hh ▸ hr)), .inl sep.mean_rstd, .inl sep.mean_y⟩
    · intro r hr
      by_cases hin : r ∈ rows
      · simpa only [hv] using hf.rstd r hin
      · have hrw : r = row := by simpa [hin] using hr
        subst r
        rw [hf.frame Rstd row ?_]
        · simpa [atRow] using he.rstd
        · intro r hr
          exact ⟨.inl (Ne.symm sep.mean_rstd), .inr (fun hh => hin (hh ▸ hr)), .inl sep.rstd_y⟩
    · intro r hr col hc
      by_cases hin : r ∈ rows
      · simpa only [ho] using hf.output r hin col hc
      · have hrw : r = row := by simpa [hin] using hr
        subst r
        rw [hf.frame Y (col + D * row) ?_]
        · simpa [rowOutput, atRow, normalizeValue] using he.output col hc
        · intro r hr
          refine ⟨.inl (Ne.symm sep.mean_y), .inl (Ne.symm sep.rstd_y), .inr ?_⟩
          intro j hj heq
          have hrow := (row_column_injective D row r col j hc hj heq).1
          exact hin (hrow ▸ hr)
    · intro q off hmiss
      rw [hf.frame q off (fun r hr => hmiss r (List.mem_cons_of_mem _ hr))]
      exact he.frame q off (by simpa [atRow] using hmiss row (by simp))

#axiomsClean whole_grid
end
end VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole
