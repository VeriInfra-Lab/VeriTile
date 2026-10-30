import bench.tritonbench_g.layer_norm_fwd.Proofs.AliasGrid
import bench.tritonbench_g.layer_norm_fwd.Proofs.NativeGeometry

/-! Actual active-output geometry: disabled outputs need no row stride.
Reuses the existing scalar source proof, physical trace and independent formula.
Only the grid composition and readback signatures change. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.RealRevision.ActiveGrid
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open WholeGrid HostReturn
noncomputable section
open scoped Classical
set_option maxHeartbeats 3000000

theorem compose (k : Kernel) (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (hgeo : NativeGeometry.Geometry f r stride M N)
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
    exact NativeGeometry.written_row_unique f r stride M N (row a) (row b) hgeo
      (row_lt a) (row_lt b) address.1 address.2 ha hb
  refine ⟨Kernel.mergeFrames (grid M) initial frames, ⟨⟨frames, hd, rfl⟩⟩, ?_, ?_⟩
  · intro i hi region offset hw
    rw [Kernel.mergeFrames_mem_written hd (index i hi) region offset hw]
    exact hpost (initial.withGridIndex (index i hi)) region offset
  · intro region offset hw
    apply Kernel.mergeFrames_mem_unwritten
    rintro ⟨idx, hidx⟩
    exact hw (row idx) (row_lt idx) hidx

theorem source_correct (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block)
    (hN : N ≤ block) (hsep : Domain.ReadWriteSeparated f r)
    (hgeo : NativeGeometry.Geometry f r stride M N) (initial : BlockState) :
    ∃ final, Nonempty (Kernel.GridLaunchedOrdinary
        (Execution.source f r stride M N eps p block hBlock) (grid M) initial final) ∧
      Post f r stride M N eps p initial final :=
  compose _ f r stride M N eps p hgeo
    (fun s => RealExecution.source_correct f r stride M N eps p block hBlock s hN hsep)
    initial

theorem y_cell (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (initial final : BlockState) (hg : NativeGeometry.Geometry f r stride M N)
    (hp : WholeGrid.Post f r stride M N eps p initial final) (i : Fin M) (j : Fin N) :
    final.mem r.y (i.val * stride.y + j.val) =
      MemCell.real (Formula.y f (snapshot r stride initial) M N eps p i.val j.val) := by
  have hi := Contract.inRow_of_col (base := i.val * stride.y) j.isLt
  have hw : Contract.written f r stride M N i.val r.y (i.val * stride.y + j.val) :=
    Or.inr (Or.inl ⟨rfl, hi⟩)
  rw [hp.1 i.val i.isLt _ _ hw]
  have hn := hg.outputs
  simp only [List.nodup_cons, List.mem_cons, List.not_mem_nil, not_or,
    List.nodup_nil, and_true] at hn
  unfold Contract.expectedCell
  split_ifs <;> simp_all [Bool.and_eq_true]

theorem y1_cell (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (initial final : BlockState)
    (hp : WholeGrid.Post f r stride M N eps p initial final) (hf : f.w1 = Bool.true)
    (i : Fin M) (j : Fin N) :
    final.mem r.y1 (i.val * stride.y1 + j.val) =
      MemCell.real (Formula.y1 f (snapshot r stride initial) M N eps p i.val j.val) := by
  have hi := Contract.inRow_of_col (base := i.val * stride.y1) j.isLt
  have hw : Contract.written f r stride M N i.val r.y1 (i.val * stride.y1 + j.val) :=
    Or.inl ⟨hf, rfl, hi⟩
  rw [hp.1 i.val i.isLt _ _ hw]
  simp [Contract.expectedCell, hf, hi]

theorem rstd_cell (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (initial final : BlockState) (hg : NativeGeometry.Geometry f r stride M N)
    (hp : WholeGrid.Post f r stride M N eps p initial final) (i : Fin M) :
    final.mem r.rstd i.val =
      MemCell.real (Formula.rstd f (snapshot r stride initial) M N eps p i.val) := by
  have hw : Contract.written f r stride M N i.val r.rstd i.val := by simp [Contract.written]
  rw [hp.1 i.val i.isLt _ _ hw]
  have hn := hg.outputs
  simp only [List.nodup_cons, List.mem_cons, List.not_mem_nil, not_or,
    List.nodup_nil, and_true] at hn
  unfold Contract.expectedCell
  split_ifs <;> simp_all [Bool.and_eq_true]

theorem mean_cell (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (initial final : BlockState) (hg : NativeGeometry.Geometry f r stride M N)
    (hp : WholeGrid.Post f r stride M N eps p initial final) (hf : f.rms = Bool.false) (i : Fin M) :
    final.mem r.mean i.val =
      MemCell.real (Formula.mean f (snapshot r stride initial) M N p i.val) := by
  have hw : Contract.written f r stride M N i.val r.mean i.val := by simp [Contract.written, hf]
  rw [hp.1 i.val i.isLt _ _ hw]
  have hn := hg.outputs
  simp only [List.nodup_cons, List.mem_cons, List.not_mem_nil, not_or,
    List.nodup_nil, and_true] at hn
  unfold Contract.expectedCell
  split_ifs <;> simp_all [Bool.and_eq_true]

theorem residual_cell (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (initial final : BlockState) (hg : NativeGeometry.Geometry f r stride M N)
    (hp : WholeGrid.Post f r stride M N eps p initial final) (hf : f.residualOut = Bool.true)
    (i : Fin M) (j : Fin N) :
    final.mem r.residualOut (i.val * stride.residualOut + j.val) =
      MemCell.real (Formula.combined f (snapshot r stride initial) M p i.val j.val) := by
  have hi := Contract.inRow_of_col (base := i.val * stride.residualOut) j.isLt
  have hw : Contract.written f r stride M N i.val r.residualOut
      (i.val * stride.residualOut + j.val) := by simp [Contract.written, hf, hi]
  rw [hp.1 i.val i.isLt _ _ hw]
  have hn := hg.outputs
  simp only [List.nodup_cons, List.mem_cons, List.not_mem_nil, not_or,
    List.nodup_nil, and_true] at hn
  unfold Contract.expectedCell
  split_ifs <;> simp_all [Bool.and_eq_true]

theorem mask_cell (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (initial final : BlockState) (hg : NativeGeometry.Geometry f r stride M N)
    (hp : WholeGrid.Post f r stride M N eps p initial final)
    (hd : f.dropout = Bool.true) (hm : f.dropoutMask = Bool.true) (i : Fin M) (j : Fin N) :
    final.mem r.dropoutMask (i.val * N + j.val) =
      MemCell.of .bool (Formula.keep ((snapshot r stride initial).seeds i.val) p j.val) := by
  have hi := Contract.inRow_of_col (base := i.val * N) j.isLt
  have hsecond : ¬ Contract.inRow ((M + i.val) * N) N (i.val * N + j.val) :=
    fun h => WholeGrid.mask_halves_disjoint i.isLt hi h
  have hw : Contract.written f r stride M N i.val r.dropoutMask (i.val * N + j.val) := by
    simp [Contract.written, hd, hm, hi]
  rw [hp.1 i.val i.isLt _ _ hw]
  have hn := hg.outputs
  simp only [List.nodup_cons, List.mem_cons, List.not_mem_nil, not_or,
    List.nodup_nil, and_true] at hn
  unfold Contract.expectedCell
  split_ifs <;> simp_all [Bool.and_eq_true]

theorem mask1_cell (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (initial final : BlockState) (hg : NativeGeometry.Geometry f r stride M N)
    (hp : WholeGrid.Post f r stride M N eps p initial final)
    (hd : f.dropout = Bool.true) (hm : f.dropoutMask = Bool.true) (hx : f.x1 = Bool.true)
    (i : Fin M) (j : Fin N) :
    final.mem r.dropoutMask ((M + i.val) * N + j.val) =
      MemCell.of .bool (Formula.keep ((snapshot r stride initial).seeds (M + i.val)) p j.val) := by
  have hi := Contract.inRow_of_col (base := (M + i.val) * N) j.isLt
  have hw : Contract.written f r stride M N i.val r.dropoutMask ((M + i.val) * N + j.val) := by
    simp [Contract.written, hd, hm, hx, hi]
  rw [hp.1 i.val i.isLt _ _ hw]
  have hn := hg.outputs
  simp only [List.nodup_cons, List.mem_cons, List.not_mem_nil, not_or,
    List.nodup_nil, and_true] at hn
  unfold Contract.expectedCell
  split_ifs <;> simp_all [Bool.and_eq_true]

theorem observe_eq_oracle (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (initial final : BlockState) (hg : NativeGeometry.Geometry f r stride M N)
    (hsep : Domain.ReadWriteSeparated f r)
    (hp : WholeGrid.Post f r stride M N eps p initial final) :
    observe f r stride M N final = oracle f r stride M N eps p initial := by
  have hy := funext fun i => funext fun j => y_cell f r stride M N eps p initial final hg hp i j
  have hy1 := fun hf => funext fun i => funext fun j =>
    y1_cell f r stride M N eps p initial final hp hf i j
  have hs := funext fun i => rstd_cell f r stride M N eps p initial final hg hp i
  have hm := fun hf => funext fun i => mean_cell f r stride M N eps p initial final hg hp hf i
  have hr := fun hf => funext fun i => funext fun j =>
    residual_cell f r stride M N eps p initial final hg hp hf i j
  have hx := funext fun (i : Fin M) => funext fun (j : Fin N) =>
    preserves_input f r stride M N eps p initial final hsep hp r.x
      (by simp [Domain.readRegion]) (i.val * stride.x + j.val)
  have hseed := fun (hd : f.dropout = Bool.true) => funext fun (i : Fin (seedCount f M)) =>
    preserves_input f r stride M N eps p initial final hsep hp r.seeds
      (by simp [Domain.readRegion, hd]) i.val
  have hmask := fun hd hm => funext fun i => funext fun j =>
    mask_cell f r stride M N eps p initial final hg hp hd hm i j
  have hmask1 := fun hd hm hx => funext fun i => funext fun j =>
    mask1_cell f r stride M N eps p initial final hg hp hd hm hx i j
  apply Result.ext
  all_goals dsimp only [observe, oracle]
  · exact hy
  · cases hf : f.w1 <;> simp [hf, hy1]
  · cases hf : f.rms <;> simp [hf, hm]
  · exact hs
  · cases hf : f.residualOut <;> simp [hf, hr, hx]
  · cases hf : f.dropout <;> simp [hf, hseed]
  · cases hd : f.dropout <;> cases hm : f.dropoutMask <;> simp [hd, hm, hmask]
  · cases hd : f.dropout <;> cases hm : f.dropoutMask <;> cases hx : f.x1 <;>
      simp [hd, hm, hx, hmask1]


theorem grid_return (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block)
    (a : AliasMemory.Layout) (initial : AliasMemory.Physical a) (template : BlockState)
    (hN : N≤block) (hsep : Domain.ReadWriteSeparated f r)
    (hg : NativeGeometry.Geometry f r stride M N)
    (hf : AliasMemory.FreshWrites a (AliasMemory.Writes f r stride M N)) :
    ∃ final physical,
      Nonempty (Kernel.GridLaunchedOrdinary
        ((repaired f r stride M N eps p block hBlock).eraseDType.toAlgKernel)
        (WholeGrid.grid M) (AliasMemory.pull a initial template) final) ∧
      AliasMemory.pull a physical final = final ∧
      WholeGrid.Post f r stride M N eps p (AliasMemory.pull a initial template) final ∧
      HostReturn.observe f r stride M N (AliasMemory.pull a physical final) =
        HostReturn.oracle f r stride M N eps p (AliasMemory.pull a initial template) ∧
      (∀ i : Fin a.capacity, (∀ k, AliasMemory.Writes f r stride M N k →
        a.address k ≠ i.val) → physical i = initial i) := by
  obtain ⟨final,hrun,hpost⟩ := source_correct f r stride M N eps p block hBlock
    hN hsep hg (AliasMemory.pull a initial template)
  let physical := AliasMemory.runPhysical a initial (AliasGrid.events f r stride M N final)
  have hpull := AliasGrid.source_trace_pull a initial template final f r stride M N eps p hf hpost
  refine ⟨final,physical,hrun,hpull,hpost,?_,?_⟩
  · rw [hpull]
    exact observe_eq_oracle f r stride M N eps p _ _ hg hsep hpost
  · intro i hi
    exact AliasMemory.trace_frame a initial (AliasGrid.events f r stride M N final) i
      (fun e he => Ne.symm (hi e.key (AliasGrid.events_written f r stride M N final e he)))

#axiomsClean grid_return
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.RealRevision.ActiveGrid
