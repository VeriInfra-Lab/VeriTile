import bench.repairs.tritonbench_g.square_matrix.Proofs.Geometry
import bench.tritonbench_g.square_matrix.Proofs.Real
import bench.tritonbench_g.square_matrix.Proofs.Source

/-! Reuses the established square-matrix whole-row induction, specialized to
this checked contiguous wrapper. The unchanged per-row DSL proof is imported. -/
namespace VeriTile.Bench.Repairs.SquareMatrix.RealProof
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.SquareMatrix (square_kernel square_kernel_region_run)
noncomputable section
open Classical
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000

/-- The mathematical executor is exactly the explicit FP32 quotation after
erasing its precision annotations, reusing the existing checked projection. -/
theorem source_projection (n B : Nat) :
    (fp32Kernel n B).eraseDType.toAlgKernel.body =
      (sourceKernel "out" "x" n n n B).toAlgKernel.body := by
  change (VeriTile.Bench.TritonBenchG.SquareMatrix.typedKernel .fp32 Bool.false
    "out" "x" n n n B).eraseDType.toAlgKernel.body = _
  rw [VeriTile.Bench.TritonBenchG.SquareMatrix.typed_projection]
  rfl


def programState (s : BlockState) (rows row : Nat) : BlockState :=
  (s.withPids (fun ax => if ax=0 then row else 0)).withNumPids
    (fun ax => if ax=0 then rows else 1)

def kernel (X O : RegionName) (si so n : Nat) : Kernel :=
  (square_kernel O X si so n (block n)).toAlgKernel

def runRows (X O : RegionName) (si so n rows : Nat) : List Nat → BlockState → Option BlockState
  | [], s => some s
  | row::rest, s => do
    let t ← exec (kernel X O si so n) (programState s rows row)
    runRows X O si so n rows rest t

def writesAt (so n row o : Nat) : Prop := row*so≤o ∧ o<row*so+n

/-- Independent last writer, including overlapping output row windows.
It consults geometry/schedule only, not execution or resulting memory. -/
def lastWriter (so n : Nat) : List Nat → Nat → Option Nat
  | [], _ => none
  | row::rest, o => match lastWriter so n rest o with
    | some later => some later
    | none => if writesAt so n row o then some row else none

def expected (X : RegionName) (si so : Nat) (initial : BlockState) (row o : Nat) : ℝ :=
  let v := initial.readMem X (row*si+(o-row*so))
  v*v

def Values (X O : RegionName) (si so n : Nat) (schedule : List Nat)
    (s t : BlockState) : Prop :=
  ∀ row o, lastWriter so n schedule o=some row → t.readMem O o=expected X si so s row o

def Frame (O : RegionName) (so n : Nat) (schedule : List Nat) (s t : BlockState) : Prop :=
  ∀ r o, (r≠O ∨ lastWriter so n schedule o=none) → t.mem r o=s.mem r o

theorem read_eq {s t : BlockState} {r : RegionName} {o : Nat}
    (h : t.mem r o=s.mem r o) : t.readMem r o=s.readMem r o := by
  simp only [BlockState.readMem,h]

theorem row_run (X O : RegionName) (si so n rows row : Nat) (s : BlockState) :
    ∃ t, exec (kernel X O si so n) (programState s rows row)=some t ∧
      Values X O si so n [row] s t ∧ Frame O so n [row] s t := by
  obtain ⟨t,he,hv,hf⟩ := square_kernel_region_run O X si so n (block n)
    (programState s rows row) (fun j => s.readMem X (row*si+j.val)) (by intro j _; rfl)
  refine ⟨t,he,?_,?_⟩
  · intro owner o ho
    simp only [lastWriter] at ho
    split at ho
    next h =>
      have eq : row=owner := Option.some.inj ho
      subst owner
      have lo : row*so≤o := h.1
      have hi : o<row*so+n := h.2
      have col : o-row*so<n := by omega
      let j : Fin (block n) := ⟨o-row*so,lt_of_lt_of_le col (block_covers n)⟩
      have addr : row*so+j.val=o := by dsimp [j]; omega
      have value := hv j col
      simpa [programState,addr,expected,j] using value
    next h => cases ho
  · intro r o h
    apply hf r o
    rcases h with hr | ho
    · exact Or.inl hr
    · right
      intro j hj eq
      have eq' : o=row*so+j.val := eq
      have hit : writesAt so n row o := by unfold writesAt; omega
      simp [lastWriter,hit] at ho

theorem runRows_correct (X O : RegionName) (si so n rows : Nat)
    (schedule : List Nat) (s : BlockState) (fresh : X≠O) :
    ∃ t, runRows X O si so n rows schedule s=some t ∧
      Values X O si so n schedule s t ∧ Frame O so n schedule s t := by
  induction schedule generalizing s with
  | nil => exact ⟨s,rfl,by simp [Values,lastWriter],by intro r o _; rfl⟩
  | cons row rest ih =>
    obtain ⟨mid,hm,hvm,hfm⟩ := row_run X O si so n rows row s
    obtain ⟨t,ht,hvt,hft⟩ := ih mid
    refine ⟨t,by simp only [runRows,hm]; exact ht,?_,?_⟩
    · intro owner o ho
      cases hl : lastWriter so n rest o with
      | some later =>
        have eq : later=owner := by simpa [lastWriter,hl] using ho
        subst owner
        rw [hvt later o hl]
        unfold expected
        rw [read_eq (hfm X _ (Or.inl fresh))]
      | none =>
        have single : lastWriter so n [row] o=some owner := by simpa [lastWriter,hl] using ho
        exact (read_eq (hft O o (Or.inr hl))).trans (hvm owner o single)
    · intro r o h
      rcases h with hr | ho
      · exact (hft r o (Or.inl hr)).trans (hfm r o (Or.inl hr))
      · have tail : lastWriter so n rest o=none := by
          cases hl : lastWriter so n rest o <;> simp_all [lastWriter]
        have head : lastWriter so n [row] o=none := by simpa [lastWriter,tail] using ho
        exact (hft r o (Or.inr tail)).trans (hfm r o (Or.inr head))

theorem lastWriter_none (so n : Nat) (schedule : List Nat) (o : Nat) :
    lastWriter so n schedule o=none ↔ ∀ row∈schedule, ¬writesAt so n row o := by
  induction schedule with
  | nil => simp [lastWriter]
  | cons row rest ih =>
    cases hl : lastWriter so n rest o with
    | none =>
      constructor
      · intro h r hr
        rcases List.mem_cons.mp hr with eq | hm
        · subst r
          intro hit
          simp [lastWriter,hl,hit] at h
        · exact ih.mp hl r hm
      · intro h
        have miss := h row (by simp)
        simp [lastWriter,hl,miss]
    | some later =>
      have hex : ¬(∀ row∈rest, ¬writesAt so n row o) := by
        intro h; have := ih.mpr h; simp [hl] at this
      simp [lastWriter,hl,List.mem_cons,hex]

theorem full_grid_correct (X O : RegionName) (si so n rows : Nat)
    (s : BlockState) (fresh : X≠O) :
    ∃ t, runRows X O si so n rows (List.range rows) s=some t ∧
      Values X O si so n (List.range rows) s t ∧
      Frame O so n (List.range rows) s t :=
  runRows_correct X O si so n rows _ s fresh

theorem every_source_write_covered (so n rows row col : Nat) (hr : row<rows) (hc : col<n) :
    lastWriter so n (List.range rows) (row*so+col)≠none := by
  intro h
  have miss := (lastWriter_none so n _ _).mp h row (List.mem_range.mpr hr)
  exact miss (by unfold writesAt; omega)

theorem lastWriter_sound (so n : Nat) (schedule : List Nat) (o row : Nat)
    (h : lastWriter so n schedule o=some row) : row∈schedule ∧ writesAt so n row o := by
  induction schedule with
  | nil => simp [lastWriter] at h
  | cons first rest ih =>
    cases hl : lastWriter so n rest o with
    | some later =>
      have eq : later=row := by simpa [lastWriter,hl] using h
      subst later
      obtain ⟨hm,hw⟩ := ih hl
      exact ⟨List.mem_cons_of_mem _ hm,hw⟩
    | none =>
      simp only [lastWriter,hl] at h
      split at h
      next hit =>
        have eq := Option.some.inj h
        subst row
        exact ⟨by simp,hit⟩
      next miss => cases h


theorem logical_grid (n rows : Nat) (s : BlockState) :
    ∃ t, runRows "x" "out" n n n rows (List.range rows) s=some t ∧
      (∀ r c, r < rows → c < n →
        t.readMem "out" (r*n+c)=s.readMem "x" (r*n+c)*s.readMem "x" (r*n+c)) ∧
      (∀ (region : RegionName) i, region ≠ "out" ∨ rows*n ≤ i → t.mem region i=s.mem region i) := by
  obtain ⟨t,he,hv,hf⟩ := full_grid_correct "x" "out" n n n rows s (by decide)
  refine ⟨t,he,?_,?_⟩
  · intro r c hr hc
    have covered := every_source_write_covered n n rows r c hr hc
    cases ho : lastWriter n n (List.range rows) (r*n+c) with
    | none => exact False.elim (covered ho)
    | some owner =>
      have hit := (lastWriter_sound n n _ _ _ ho).2
      have eq := row_unique n r c owner hc hit.1 hit.2
      subst owner
      simpa [expected] using hv r (r*n+c) ho
  · intro region i outside
    apply hf region i
    rcases outside with h | h
    · exact Or.inl h
    · right
      apply (lastWriter_none n n _ i).mpr
      intro row hr hit
      have next := Nat.mul_le_mul_right n (List.mem_range.mp hr)
      have upper := Nat.mul_le_mul_right n (Nat.succ_le_of_lt (List.mem_range.mp hr))
      rw [Nat.succ_mul] at upper
      unfold writesAt at hit
      omega

/-- The original transpose launch misses offset three: stride one makes
its two output rows overlap at offset one. -/
theorem original_transpose_missing : lastWriter 1 2 (List.range 2) 3=none := by
  norm_num [lastWriter,writesAt,List.range_succ]

#axiomsClean original_transpose_missing
#axiomsClean source_projection
#axiomsClean logical_grid
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SquareMatrix.RealProof
