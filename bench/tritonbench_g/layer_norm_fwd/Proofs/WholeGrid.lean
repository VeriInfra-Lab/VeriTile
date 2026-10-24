import bench.tritonbench_g.layer_norm_fwd.Proofs.Candidate
import VeriTile.Triton.Launch.Composition
import VeriTile.Meta.StatementAudit

/-! Whole `(M,)` grid composition of the complete source programs. Every
program executes, the seven write paths are pairwise disjoint across rows,
and the merged memory has the independent initial-snapshot formulas and a
full tagged frame. This theorem uses the source Real semantics; it does not
claim a native compiler, fixed-width-address, or rounded arithmetic bridge. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.WholeGrid
open VeriTile.Triton
noncomputable section
open scoped Classical
set_option maxHeartbeats 2000000

def grid (M : Nat) : Grid := ⟨[M]⟩

def row {M : Nat} (idx : GridIndex (grid M)) : Nat :=
  (idx ⟨0, by simp [grid, Grid.rank]⟩).val

theorem row_lt {M : Nat} (idx : GridIndex (grid M)) : row idx < M :=
  (idx ⟨0, by simp [grid, Grid.rank]⟩).isLt

def index {M : Nat} (i : Nat) (hi : i < M) : GridIndex (grid M) :=
  fun a => ⟨i, by
    have ha : a.val = 0 := by have := a.isLt; change a.val < 1 at this; omega
    simpa [grid, Grid.dim, ha] using hi⟩

@[simp] theorem row_index {M : Nat} (i : Nat) (hi : i < M) :
    row (index i hi) = i := rfl

@[simp] theorem pid_index {M : Nat} (s : BlockState) (idx : GridIndex (grid M)) :
    (s.withGridIndex idx).pid = row idx := rfl

theorem index_ext {M : Nat} {a b : GridIndex (grid M)} (h : row a = row b) : a = b := by
  funext ax
  have ha : ax = ⟨0, by simp [grid, Grid.rank]⟩ := by
    apply Fin.ext
    change ax.val = 0
    have := ax.isLt
    change ax.val < 1 at this
    omega
  subst ax
  exact Fin.ext h

/-- Fresh output allocation names and the nonoverlapping rows supplied by
`empty`/`empty_like`. Input rows remain unrestricted and may overlap. Unused
output names can always be chosen fresh; no inactive pointer is accessed. -/
structure Geometry (r : Regions) (stride : Strides) (N : Nat) : Prop where
  outputs : [r.y1, r.y, r.rstd, r.mean, r.residualOut, r.dropoutMask].Nodup
  y : N ≤ stride.y
  y1 : N ≤ stride.y1
  residualOut : N ≤ stride.residualOut

theorem inRow_unique {row other step N offset : Nat} (hs : N ≤ step)
    (ha : Contract.inRow (row * step) N offset)
    (hb : Contract.inRow (other * step) N offset) : row = other := by
  obtain ⟨a, ha, hea⟩ := Contract.inRow_iff_col.mp ha
  obtain ⟨b, hb, heb⟩ := Contract.inRow_iff_col.mp hb
  exact Contract.rows_disjoint hs ha hb (hea.symm.trans heb)

theorem mask_halves_disjoint {M N row other offset : Nat} (hr : row < M)
    (ha : Contract.inRow (row * N) N offset)
    (hb : Contract.inRow ((M + other) * N) N offset) : False := by
  obtain ⟨a, ha, hea⟩ := Contract.inRow_iff_col.mp ha
  have hlt := Domain.first_mask_lt hr ha
  have hlo : M * N ≤ offset := by
    have := hb.1
    rw [Nat.add_mul] at this
    omega
  omega

theorem written_row_unique (f : Flags) (r : Regions) (stride : Strides)
    (M N row other : Nat) (hgeo : Geometry r stride N)
    (hr : row < M) (ho : other < M) (region : RegionName) (offset : Nat)
    (ha : Contract.written f r stride M N row region offset)
    (hb : Contract.written f r stride M N other region offset) : row = other := by
  have hn := hgeo.outputs
  simp only [List.nodup_cons, List.mem_cons, not_or,
    List.nodup_nil, and_true] at hn
  rcases ha with ⟨_, ha, hia⟩ | ⟨ha, hia⟩ | ⟨ha, hia⟩ | ⟨_, ha, hia⟩ |
    ⟨_, ha, hia⟩ | ⟨_, _, _, ha, hia⟩ | ⟨_, _, ha, hia⟩ <;>
    rcases hb with ⟨_, hb, hib⟩ | ⟨hb, hib⟩ | ⟨hb, hib⟩ | ⟨_, hb, hib⟩ |
      ⟨_, hb, hib⟩ | ⟨_, _, _, hb, hib⟩ | ⟨_, _, hb, hib⟩
  all_goals try simp_all
  all_goals first
    | exact inRow_unique hgeo.y1 hia hib
    | exact inRow_unique hgeo.y hia hib
    | exact inRow_unique hgeo.residualOut hia hib
    | exact inRow_unique (Nat.le_refl N) hia hib
    | (have h := inRow_unique (Nat.le_refl N) hia hib; omega)

/-- All active row cells and the complement, specified without execution. -/
def Post (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (initial final : BlockState) : Prop :=
  (∀ i, i < M → ∀ region offset, Contract.written f r stride M N i region offset →
    final.mem region offset =
      (Contract.expectedCell f r stride (snapshot r stride initial)
        M N eps p i region offset).getD (initial.mem region offset)) ∧
  (∀ region offset, (∀ i, i < M → ¬ Contract.written f r stride M N i region offset) →
    final.mem region offset = initial.mem region offset)

private theorem compose (k : Kernel) (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (hgeo : Geometry r stride N)
    (hrun : ∀ s, ∃ t, exec k s = some t ∧ Contract.Post f r stride M N eps p s t)
    (initial : BlockState) :
    ∃ final, Nonempty (Kernel.GridLaunchedOrdinary k (grid M) initial final) ∧
      Post f r stride M N eps p initial final := by
  choose finish hexec hpost using hrun
  let frames : Kernel.GridFrames k (grid M) initial := fun idx =>
    { final := finish (initial.withGridIndex idx)
      writes := fun a => Contract.written f r stride M N (row idx) a.1 a.2
      h_exec := hexec _
      h_writeWithin := by
        intro region offset hw
        exact (Contract.post_frame f r stride M N eps p _ _
          (hpost _) region offset hw).symm }
  have hd : Kernel.GridWritesDisjoint frames := by
    intro a b hab address ha hb
    apply hab
    apply index_ext
    exact written_row_unique f r stride M N (row a) (row b) hgeo
      (row_lt a) (row_lt b) address.1 address.2 ha hb
  refine ⟨Kernel.mergeFrames (grid M) initial frames, ⟨⟨frames, hd, rfl⟩⟩, ?_, ?_⟩
  · intro i hi region offset hw
    rw [Kernel.mergeFrames_mem_written hd (index i hi) region offset hw]
    exact hpost (initial.withGridIndex (index i hi)) region offset
  · intro region offset hw
    apply Kernel.mergeFrames_mem_unwritten
    rintro ⟨idx, hidx⟩
    exact hw (row idx) (row_lt idx) hidx

theorem original_correct (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block)
    (hN : N ≤ block) (hsep : Domain.ReadWriteSeparated f r)
    (hgeo : Geometry r stride N) (initial : BlockState) :
    ∃ final, Nonempty (Kernel.GridLaunchedOrdinary
        (Execution.source f r stride M N eps p block hBlock) (grid M) initial final) ∧
      Post f r stride M N eps p initial final :=
  compose _ f r stride M N eps p hgeo
    (fun s => RealExecution.source_correct f r stride M N eps p block hBlock s hN hsep)
    initial

theorem candidate_correct (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block)
    (hN : N ≤ block) (hsep : Domain.ReadWriteSeparated f r)
    (hgeo : Geometry r stride N) (initial : BlockState) :
    ∃ final, Nonempty (Kernel.GridLaunchedOrdinary
        (CandidateExecution.source f r stride M N eps p block hBlock)
        (grid M) initial final) ∧ Post f r stride M N eps p initial final :=
  compose _ f r stride M N eps p hgeo
    (fun s => CandidateExecution.source_correct f r stride M N eps p block hBlock s hN hsep)
    initial

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.WholeGrid
