import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_RealIO
import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_HostReturn

/-! Finite allocation execution for the whole grid, retaining the shared
tagged memory, independent row strides and all output frames. Addresses here
are flat *cells*, as in the existing flatten bridge; native byte-address and
fixed-width-integer correspondence is a separate obligation. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.WholeFlatGrid
open VeriTile.Triton
noncomputable section
open scoped Classical
set_option maxHeartbeats 2000000

theorem flatten_grid_index (A : FlatAlloc) (s : BlockState) {g : Grid} (idx : GridIndex g) :
    A.flattenState (s.withGridIndex idx) = (A.flattenState s).withGridIndex idx := rfl

private theorem lift (A : FlatAlloc) (hd : A.Disjoint)
    (hc : ∀ r, r ∉ A.regions → A.extent r = 0) (k : Kernel) (g : Grid)
    (initial final : BlockState) (hl : Kernel.GridLaunchedOrdinary k g initial final)
    (hs : ∀ idx : GridIndex g, k.TraceSafe A.extent (initial.withGridIndex idx))
    (hok : k.FlattenOk) (hu : initial.undef = (fun _ _ => 0)) :
    Nonempty (Kernel.GridLaunchedOrdinary (A.flattenKernel k) g
      (A.flattenState initial) (A.flattenState final)) := by
  let frames : Kernel.GridFrames (A.flattenKernel k) g (A.flattenState initial) := fun idx =>
    { final := A.flattenState (hl.frames idx).final
      writes := fun address => address.1 = A.flat ∧
        ∃ q z, A.decode address.2 = some (q, z) ∧ (hl.frames idx).writes (q, z)
      h_exec := by
        rw [← flatten_grid_index, A.exec_flatten hd hc k _ (hs idx) hok hu,
          (hl.frames idx).h_exec]
        rfl
      h_writeWithin := by
        intro region offset hn
        change (A.flattenState initial).mem region offset =
          (A.flattenState (hl.frames idx).final).mem region offset
        simp only [FlatAlloc.flattenState]
        split
        · unfold FlatAlloc.readFlat
          cases hdec : A.decode offset with
          | none => rfl
          | some addr =>
            obtain ⟨q, z⟩ := addr
            apply congrArg A.trCell
            exact (hl.frames idx).h_writeWithin q z
              (fun hw => hn ⟨by assumption, q, z, hdec, hw⟩)
        · rfl }
  have hdis : Kernel.GridWritesDisjoint frames := by
    intro a b hab address ha hb
    obtain ⟨_, q, z, hq, hqa⟩ := ha
    obtain ⟨_, q', z', hq', hqb⟩ := hb
    have hpair := Option.some.inj (hq.symm.trans hq')
    cases hpair
    exact hl.h_disjoint a b hab (q, z) hqa hqb
  refine ⟨⟨frames, hdis, ?_⟩⟩
  rw [hl.h_final]
  apply BlockState.ext
  · intro region offset
    by_cases hr : region = A.flat
    · subst region
      cases hdec : A.decode offset with
      | none =>
        have hn : ¬ Kernel.GridWriteFootprint frames (A.flat, offset) := by
          rintro ⟨idx, _, q, z, hq, _⟩
          rw [hdec] at hq
          contradiction
        rw [Kernel.mergeFrames_mem_unwritten A.flat offset hn]
        simp [FlatAlloc.flattenState, FlatAlloc.readFlat, hdec]
      | some addr =>
        obtain ⟨q, z⟩ := addr
        by_cases hw : ∃ idx, (hl.frames idx).writes (q, z)
        · obtain ⟨idx, hidx⟩ := hw
          rw [Kernel.mergeFrames_mem_written hdis idx A.flat offset ⟨rfl, q, z, hdec, hidx⟩]
          simp only [frames, FlatAlloc.flattenState, FlatAlloc.readFlat, hdec, ite_true]
          rw [Kernel.mergeFrames_mem_written hl.h_disjoint idx q z hidx]
        · have hn : ¬ Kernel.GridWriteFootprint frames (A.flat, offset) := by
            rintro ⟨idx, _, q', z', hq, hidx⟩
            have he := Option.some.inj (hdec.symm.trans hq)
            cases he
            exact hw ⟨idx, hidx⟩
          rw [Kernel.mergeFrames_mem_unwritten A.flat offset hn]
          simp only [FlatAlloc.flattenState, FlatAlloc.readFlat, hdec]
          rw [Kernel.mergeFrames_mem_unwritten q z hw]
    · have hn : ¬ Kernel.GridWriteFootprint frames (region, offset) := by
        rintro ⟨idx, hidx, _⟩
        exact hr hidx
      rw [Kernel.mergeFrames_mem_unwritten region offset hn]
      simp [FlatAlloc.flattenState, hr]
  · intros; rfl
  · intros; rfl
  · intros; rfl
  · intros; rfl

theorem original_correct (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block)
    (hN : N ≤ block) (hsep : Domain.ReadWriteSeparated f r)
    (hg : WholeGrid.Geometry r stride N) (A : FlatAlloc) (hd : A.Disjoint)
    (hc : ∀ q, q ∉ A.regions → A.extent q = 0)
    (hb : FlatDomain.Extents f r stride M N A.extent)
    (initial : BlockState) (hu : initial.undef = (fun _ _ => 0)) :
    ∃ final, Nonempty (Kernel.GridLaunchedOrdinary
        (A.flattenKernel (Execution.source f r stride M N eps p block hBlock))
        (WholeGrid.grid M) (A.flattenState initial) (A.flattenState final)) ∧
      WholeGrid.Post f r stride M N eps p initial final ∧
      HostReturn.observe f r stride M N final = HostReturn.oracle f r stride M N eps p initial := by
  obtain ⟨final, ⟨hl⟩, hp⟩ := WholeGrid.original_correct f r stride M N eps p block hBlock hN hsep hg initial
  refine ⟨final, ?_, hp, HostReturn.observe_eq_oracle f r stride M N eps p initial final hg hsep hp⟩
  apply lift A hd hc _ _ initial final hl
  · intro idx
    exact Safety.source_safe f r stride M N eps p block hBlock _ hN hsep A.extent
      (FlatDomain.Extents.rowBounds f r stride M N _ A.extent hb (WholeGrid.row_lt idx))
  · exact Safety.source_flatten_ok f r stride M N eps p block hBlock
  · exact hu

theorem candidate_correct (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block)
    (hN : N ≤ block) (hsep : Domain.ReadWriteSeparated f r)
    (hg : WholeGrid.Geometry r stride N) (A : FlatAlloc) (hd : A.Disjoint)
    (hc : ∀ q, q ∉ A.regions → A.extent q = 0)
    (hb : FlatDomain.Extents f r stride M N A.extent)
    (initial : BlockState) (hu : initial.undef = (fun _ _ => 0)) :
    ∃ final, Nonempty (Kernel.GridLaunchedOrdinary
        (A.flattenKernel (CandidateExecution.source f r stride M N eps p block hBlock))
        (WholeGrid.grid M) (A.flattenState initial) (A.flattenState final)) ∧
      WholeGrid.Post f r stride M N eps p initial final ∧
      HostReturn.observe f r stride M N final = HostReturn.oracle f r stride M N eps p initial := by
  obtain ⟨final, ⟨hl⟩, hp⟩ := WholeGrid.candidate_correct f r stride M N eps p block hBlock hN hsep hg initial
  refine ⟨final, ?_, hp, HostReturn.observe_eq_oracle f r stride M N eps p initial final hg hsep hp⟩
  apply lift A hd hc _ _ initial final hl
  · intro idx
    exact Safety.candidate_safe f r stride M N eps p block hBlock _ hN hsep A.extent
      (FlatDomain.Extents.rowBounds f r stride M N _ A.extent hb (WholeGrid.row_lt idx))
  · exact Safety.candidate_flatten_ok f r stride M N eps p block hBlock
  · exact hu

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.WholeFlatGrid
