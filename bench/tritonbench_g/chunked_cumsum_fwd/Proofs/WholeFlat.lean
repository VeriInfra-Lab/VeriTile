import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.WholeGrid

namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Whole
open VeriTile.Triton Math Program Initialization
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 10000

def AllBounds (c : Host) (r : Regions) (hasBias : Bool) (A : FlatAlloc) : Prop :=
  ∀ l, c.valid l → Safety.Bounds c.layout l r hasBias A.extent

@[simp] theorem flatten_tileState (A : FlatAlloc) (l : Launch) (s : BlockState) :
    A.flattenState (tileState l s) = tileState l (A.flattenState s) := rfl

theorem flatten_run (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ rg, rg ∉ A.regions → A.extent rg = 0)
    (c : Host) (r : Regions) (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound)
    (hb : AllBounds c r hasBias A) (ls : List Launch) (valid : ∀ l ∈ ls, c.valid l)
    (s : BlockState) :
    runTiles (A.flattenKernel (source32 c r hasBias useSoftplus lower upper)) ls (A.flattenState s) =
      (runTiles (source32 c r hasBias useSoftplus lower upper) ls s).map A.flattenState := by
  induction ls generalizing s with
  | nil => rfl
  | cons l rest ih =>
    have safe := Safety.source_safe c.layout r .fp32 Bool.false .fp32 .fp32 hasBias useSoftplus
      lower upper (tileState l s) A.extent (by simpa using hb l (valid l List.mem_cons_self))
    have bridge := A.exec_flatten hd hcov (source32 c r hasBias useSoftplus lower upper)
      (tileState l s) safe
      (Safety.source_flatten_ok c.layout r .fp32 Bool.false .fp32 .fp32 hasBias useSoftplus lower upper) rfl
    simp only [flatten_tileState] at bridge
    simp only [runTiles,bridge]
    cases he : exec (source32 c r hasBias useSoftplus lower upper) (tileState l s) with
    | none => rfl
    | some v =>
      simp only [Option.map_some,Option.bind_some]
      exact ih (fun q hq => valid q (List.mem_cons_of_mem l hq)) v

def flatFootprint (A : FlatAlloc) (P : WriteFootprint) : WriteFootprint :=
  fun cell => cell.1 = A.flat ∧ ∃ rg off, P (rg,off) ∧ A.addr rg off = cell.2

theorem frame_flatten (A : FlatAlloc) (P : WriteFootprint) (u v : BlockState)
    (hf : BlockState.WriteWithin P u v) :
    BlockState.WriteWithin (flatFootprint A P) (A.flattenState u) (A.flattenState v) := by
  intro rg off hn
  by_cases hr : rg = A.flat
  · subst rg
    simp only [FlatAlloc.flattenState,if_pos rfl,FlatAlloc.readFlat]
    cases he : A.decode off with
    | none => rfl
    | some cell =>
      rcases cell with ⟨rg',off'⟩
      obtain ⟨_,heq,_⟩ := A.decode_sound he
      apply congrArg A.trCell
      exact hf rg' off' (fun h => hn ⟨rfl,rg',off',h,heq.symm⟩)
  · simp only [FlatAlloc.flattenState,if_neg hr]

def FlatValues (A : FlatAlloc) (c : Host) (r : Regions) (a : Inputs)
    (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound) (v : BlockState) : Prop :=
  ∀ l ∈ c.grid, ∀ (i : Fin c.layout.BH) (j : Fin c.layout.BC),
    activeOutput c.layout l i.val j.val →
    v.readMem A.flat (A.addr r.out (outAddress c.layout l i.val j.val)) =
      prepared c.layout l a hasBias useSoftplus lower upper i.val j.val ∧
    v.readMem A.flat (A.addr r.sum (sumAddress c.layout l i.val j.val)) =
      cumulative c.layout l a hasBias useSoftplus lower upper i j

/-- Complete original-source flat execution, both outputs, and every tagged
cell outside all stores. Bounds concern actual memory accesses, not a supplied
execution or an assumed postcondition. -/
theorem flat_grid_correct (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ rg, rg ∉ A.regions → A.extent rg = 0)
    (c : Host) (r : Regions) (fresh : Fresh r)
    (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound)
    (hb : AllBounds c r hasBias A) (ls : List Launch) (schedule : ls.Perm c.grid)
    (s : BlockState) :
    ∃ v, runTiles (A.flattenKernel (source32 c r hasBias useSoftplus lower upper)) ls
        (A.flattenState s) = some v ∧
      FlatValues A c r ⟨s.readMem r.dt,s.readMem r.rate,s.readMem r.bias⟩
        hasBias useSoftplus lower upper v ∧
      BlockState.WriteWithin (flatFootprint A (footprint c r c.grid)) (A.flattenState s) v := by
  obtain ⟨v,ran,values,frame⟩ := grid_correct c r fresh hasBias useSoftplus lower upper ls schedule s
  have bridge := flatten_run A hd hcov c r hasBias useSoftplus lower upper hb ls
    (fun l hl => (c.mem_grid l).1 (schedule.mem_iff.mp hl)) s
  rw [ran,Option.map_some] at bridge
  refine ⟨A.flattenState v,bridge,?_,frame_flatten A _ s v frame⟩
  have read (rg : RegionName) (off : Nat) (hbound : off < A.extent rg) :
      (A.flattenState v).readMem A.flat (A.addr rg off) = v.readMem rg off := by
    have hmem : rg ∈ A.regions := by
      by_contra hn
      rw [hcov rg hn] at hbound
      omega
    exact A.flattenState_readMem hd v hmem hbound
  intro l hl i j ha
  rw [read _ _ ((hb l ((c.mem_grid l).1 hl)).out i j ha),
    read _ _ ((hb l ((c.mem_grid l).1 hl)).sum i j ha)]
  exact values l hl i j ha

end
end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Whole
