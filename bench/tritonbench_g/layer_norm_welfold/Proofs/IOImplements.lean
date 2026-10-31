import bench.tritonbench_g.layer_norm_welfold.IO
import bench.tritonbench_g.layer_norm_welfold.Proofs.ReadableSafety
import bench.tritonbench_g.layer_norm_welfold.Proofs.Trials

namespace VeriTile.Bench.Repairs.LayerNormWelfold.IOProof
open VeriTile.Triton FP.Structural Reuse
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
variable {α : Type} [Inhabited α]

theorem schedule_eq (M : Algebra α) (Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : ReductionTree B) (rows : List Nat) (s : State α) :
    runSchedule (rows.map (IO.invocation M Y Mean Rstd S D B tree)) s =
      SourceCode.runRows (withEquality M (fun _ _ => Bool.false))
        (SourceCode.rowKernel "X" "W" "Bias" Y Mean Rstd S D B (SourceBinding.tree tree)) rows s := by
  induction rows generalizing s with
  | nil => rfl
  | cons row rows ih =>
    simp only [List.map_cons,runSchedule,SourceCode.runRows]
    have head : (IO.invocation M Y Mean Rstd S D B tree row).run s =
        exec (withEquality M (fun _ _ => Bool.false))
          (SourceCode.rowKernel "X" "W" "Bias" Y Mean Rstd S D B (SourceBinding.tree tree))
          {s with pids := fun ax => if ax=0 then row else 0} := rfl
    rw [head]
    congr 1
    funext t
    exact ih t

theorem schedule_safe (M : Algebra α) (observe : α → α)
    (Y Mean Rstd : RegionName) (S D B : Nat) (tree : ReductionTree B)
    (separated : ("X" : RegionName) ≠ Mean)
    (vals : ChannelValues α (IO.inputs S D)) (bounds : RegionName → Nat)
    (bounded : (IO.layerNormIO M observe Y Mean Rstd S D B tree).InBounds bounds vals)
    (rows : List Nat) (legal : ∀ row ∈ rows, row < S) (s : State α) :
    ScheduleSafe bounds (rows.map (IO.invocation M Y Mean Rstd S D B tree)) s := by
  induction rows generalizing s with
  | nil => trivial
  | cons row rows ih =>
    refine ⟨?_,fun t _ => ih (fun r hr => legal r (List.mem_cons_of_mem _ hr)) t⟩
    have hr := legal row List.mem_cons_self
    apply Readable.source_safe M (fun _ _ => Bool.false) Bool.false (atRow s row)
      "X" "W" "Bias" Y Mean Rstd S D B tree separated bounds
    · intro j hj
      exact bounded.2.1 0 (⟨row,hr⟩,⟨j,hj⟩,()) trivial
    · intro j hj
      exact bounded.2.1 1 (⟨j,hj⟩,()) trivial
    · intro j hj
      exact bounded.2.1 2 (⟨j,hj⟩,()) trivial
    · intro j hj
      exact bounded.2.2.1 2 (⟨row,hr⟩,⟨j,hj⟩,()) trivial
    · exact bounded.2.2.1 0 (⟨row,hr⟩,()) trivial
    · exact bounded.2.2.1 1 (⟨row,hr⟩,()) trivial

theorem arrayX_at (S D row col : Nat) (vals : ChannelValues α (IO.inputs S D))
    (hr : row < S) (hc : col < D) :
    IO.arrayX S D vals (col+D*row) = vals 0 (⟨row,hr⟩,⟨col,hc⟩,()) := by
  have hp : 0 < D := by omega
  have div : (col+D*row)/D = row := by
    rw [Nat.add_mul_div_left _ _ hp,Nat.div_eq_of_lt hc,Nat.zero_add]
  have mod : (col+D*row)%D = col := by
    simp [Nat.add_mod,Nat.mod_eq_of_lt hc]
  simp [IO.arrayX,div,mod,hr,hc]

/-- Compose the preserved complete source-grid result with the public
channels. Actual source safety and bounds follow independently above. -/
theorem implements (M : Algebra α) (observe : α → α)
    (Y Mean Rstd : RegionName) (S D B : Nat) (tree : ReductionTree B)
    (separated : ("X" : RegionName) ≠ Mean)
    (f : (Nat → α) → (Nat → α) → (Nat → α) → Nat → α × α × (Nat → α))
    (correct : ∀ (s : State α) (x w bias : Nat → α),
      (∀ row, row < S → ∀ col, col < D → (s.mem "X" (col+D*row)).read .real = x (col+D*row)) →
      (∀ col, col < D → (s.mem "W" col).read .real = w col) →
      (∀ col, col < D → (s.mem "Bias" col).read .real = bias col) →
      ∃ final, SourceCode.launch (withEquality M (fun _ _ => Bool.false)) Bool.false s Y Mean Rstd S D B
          (SourceBinding.tree tree) = some final ∧
        (∀ row, row < S → mapCell observe (final.mem Mean row) = .mk .real (f x w bias row).1) ∧
        (∀ row, row < S → mapCell observe (final.mem Rstd row) = .mk .real (f x w bias row).2.1) ∧
        (∀ row, row < S → ∀ col, col < D →
          mapCell observe (final.mem Y (col+D*row)) = .mk .bf16 ((f x w bias row).2.2 col)) ∧
        (∀ q off, gridMiss Y Mean Rstd S D q off → final.mem q off = s.mem q off)) :
    (IO.layerNormIO M observe Y Mean Rstd S D B tree).Implements (IO.result Y Mean Rstd S D f) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds bounded s _
    exact schedule_safe M observe Y Mean Rstd S D B tree separated vals bounds bounded
      (List.range S) (fun row hr => List.mem_range.mp hr) s
  · intro vals s binding
    have hx : ∀ row, row < S → ∀ col, col < D →
        (s.mem "X" (col+D*row)).read .real = IO.arrayX S D vals (col+D*row) := by
      intro row hr col hc
      rw [arrayX_at S D row col vals hr hc]
      exact binding 0 (⟨row,hr⟩,⟨col,hc⟩,()) trivial
    have hw : ∀ col, col < D → (s.mem "W" col).read .real = IO.arrayW S D vals col := by
      intro col hc
      simpa only [IO.arrayW,dif_pos hc] using binding 1 (⟨col,hc⟩,()) trivial
    have hb : ∀ col, col < D → (s.mem "Bias" col).read .real = IO.arrayB S D vals col := by
      intro col hc
      simpa only [IO.arrayB,dif_pos hc] using binding 2 (⟨col,hc⟩,()) trivial
    obtain ⟨final,ran,hm,hi,hy,frame⟩ := correct s _ _ _ hx hw hb
    refine ⟨final,(schedule_eq M Y Mean Rstd S D B tree (List.range S) s).trans ran,?_,?_⟩
    · intro o j _
      fin_cases o
      · exact hm j.1.val j.1.isLt
      · exact hi j.1.val j.1.isLt
      · exact hy j.1.val j.1.isLt j.2.1.val j.2.1.isLt
    · intro q off outside
      apply frame
      intro row hr
      refine ⟨outside.1 0 (⟨row,hr⟩,()) trivial,outside.1 1 (⟨row,hr⟩,()) trivial,?_⟩
      by_cases same : q ≠ Y
      · exact Or.inl same
      · right
        intro col hc
        exact (outside.1 2 (⟨row,hr⟩,⟨col,hc⟩,()) trivial).resolve_left same

#axiomsClean implements
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormWelfold.IOProof
