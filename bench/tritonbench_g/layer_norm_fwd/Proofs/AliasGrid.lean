import bench.tritonbench_g.layer_norm_fwd.Proofs.AliasMemory

namespace VeriTile.Bench.Repairs.LayerNormFwd.AliasGrid
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open AliasMemory
noncomputable section
open scoped Classical
set_option maxHeartbeats 2000000

def rowCells (region : RegionName) (start N : Nat) : List Key :=
  (List.range N).map (fun col => (region,start+col))

theorem mem_rowCells (k : Key) (region : RegionName) (start N : Nat) :
    k ∈ rowCells region start N ↔ k.1=region ∧ Contract.inRow start N k.2 := by
  constructor
  · intro h
    obtain ⟨col,hc,he⟩ := List.mem_map.mp h
    have hk : k=(region,start+col) := he.symm
    rw [hk]
    exact ⟨rfl,Contract.inRow_of_col (List.mem_range.mp hc)⟩
  · rintro ⟨hr,hrow⟩
    obtain ⟨col,hc,he⟩ := Contract.inRow_iff_col.mp hrow
    apply List.mem_map.mpr
    exact ⟨col,List.mem_range.mpr hc,Prod.ext hr.symm he.symm⟩

def optional (flag : Bool) (xs : List α) : List α := if flag then xs else []
@[simp] theorem mem_optional (flag : Bool) (xs : List α) (x : α) :
    x ∈ optional flag xs ↔ flag=Bool.true ∧ x∈xs := by cases flag <;> simp [optional]

/-- Exact store-site order of the complete production source. -/
def rowKeys (f : Flags) (r : Regions) (stride : Strides) (M N row : Nat) : List Key :=
  optional (f.dropout && f.dropoutMask) (rowCells r.dropoutMask (row*N) N) ++
  optional (f.x1 && f.dropout && f.dropoutMask) (rowCells r.dropoutMask ((M+row)*N) N) ++
  optional f.residualOut (rowCells r.residualOut (row*stride.residualOut) N) ++
  optional (!f.rms) [(r.mean,row)] ++ [(r.rstd,row)] ++
  rowCells r.y (row*stride.y) N ++ optional f.w1 (rowCells r.y1 (row*stride.y1) N)

theorem mem_rowKeys (f : Flags) (r : Regions) (stride : Strides) (M N row : Nat) (k : Key) :
    k ∈ rowKeys f r stride M N row ↔ Contract.written f r stride M N row k.1 k.2 := by
  simp only [rowKeys,List.mem_append,mem_optional,mem_rowCells,List.mem_cons,List.not_mem_nil,
    or_false,Bool.and_eq_true,Bool.not_eq_true,Prod.mk.injEq]
  unfold Contract.written
  have hk : k=(r.mean,row) ↔ k.1=r.mean ∧ k.2=row := Prod.ext_iff
  have hk' : k=(r.rstd,row) ↔ k.1=r.rstd ∧ k.2=row := Prod.ext_iff
  have hn : ((!f.rms)=Bool.true) ↔ f.rms=Bool.false := by cases f.rms <;> decide
  rw [hk,hk',hn]
  simp only [and_assoc,and_left_comm,and_comm,or_assoc,or_left_comm,or_comm]

def keys (f : Flags) (r : Regions) (stride : Strides) (M N : Nat) : List Key :=
  (List.range M).flatMap (fun row => rowKeys f r stride M N row)

theorem mem_keys (f : Flags) (r : Regions) (stride : Strides) (M N : Nat) (k : Key) :
    k ∈ keys f r stride M N ↔ Writes f r stride M N k := by
  simp [keys,List.mem_flatMap,mem_rowKeys,Writes]

/-- Values are taken from the proved successful source result at the uniquely
written cells. The independently proved source postcondition later identifies
these values with the input-only oracle; no desired-value premise is supplied. -/
def events (f : Flags) (r : Regions) (stride : Strides) (M N : Nat) (final : BlockState) : List Event :=
  (keys f r stride M N).map fun k => ⟨k,final.mem k.1 k.2⟩

/-- Exact effect of the ordered source store sites, retaining arbitrary initial
memory and repeated equal-value events if a source schedule duplicates a store. -/
theorem tagged_events_mem (ks : List Key) (initial final : BlockState) (k : Key) :
    (runTagged initial (ks.map fun q => ⟨q,final.mem q.1 q.2⟩)).mem k.1 k.2 =
      if k∈ks then final.mem k.1 k.2 else initial.mem k.1 k.2 := by
  induction ks generalizing initial with
  | nil => simp [runTagged]
  | cons q qs ih =>
    simp only [List.map_cons,runTagged,List.foldl_cons]
    change (runTagged (Memory.writeCell initial q.1 q.2 (final.mem q.1 q.2))
      (qs.map fun z => ⟨z,final.mem z.1 z.2⟩)).mem k.1 k.2 = _
    rw [ih]
    by_cases hk : k∈qs
    · simp [hk]
    · by_cases he : k=q
      · subst q; simp [hk,Memory.writeCell]
      · have hp : ¬(k.1=q.1 ∧ k.2=q.2) := by intro h; exact he (Prod.ext h.1 h.2)
        simp [hk,he,Memory.writeCell,hp]

theorem events_written (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (final : BlockState) (e : Event) (he : e∈events f r stride M N final) :
    Writes f r stride M N e.key := by
  obtain ⟨k,hk,rfl⟩ := List.mem_map.mp he
  exact (mem_keys f r stride M N k).mp hk

/-- Successful source execution supplies the emitted values; store simulation
then proves the actual ordered physical scatter equals that result on every
view, including overlapping reads and all untouched physical cells. -/
theorem source_trace_pull (a : Layout) (initial : Physical a) (template final : BlockState)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat) (eps p : ℝ)
    (hf : FreshWrites a (Writes f r stride M N))
    (hp : WholeGrid.Post f r stride M N eps p (pull a initial template) final) :
    pull a (runPhysical a initial (events f r stride M N final)) final = final := by
  rw [trace_commutes a initial final (Writes f r stride M N) hf _
    (events_written f r stride M N final)]
  apply BlockState.ext
  · intro region offset
    rw [events,tagged_events_mem _ _ final (region,offset)]
    by_cases hw : Writes f r stride M N (region,offset)
    · rw [if_pos ((mem_keys f r stride M N _).mpr hw)]
    · rw [if_neg (fun h => hw ((mem_keys f r stride M N _).mp h))]
      exact (hp.2 region offset (by intro row hr h; exact hw ⟨row,hr,h⟩)).symm
  · intro dtype shape name
    have h : ∀ (es : List Event) (s : BlockState),
        (runTagged s es).regs dtype shape name = s.regs dtype shape name := by
      intro es; induction es with
      | nil => intro s; rfl
      | cons e es ih => intro s; exact ih _
    exact h _ _
  · intro ax
    have h : ∀ (es : List Event) (s : BlockState), (runTagged s es).pids ax = s.pids ax := by
      intro es; induction es with
      | nil => intro s; rfl
      | cons e es ih => intro s; exact ih _
    exact h _ _
  · intro region offset
    have h : ∀ (es : List Event) (s : BlockState), (runTagged s es).undef region offset = s.undef region offset := by
      intro es; induction es with
      | nil => intro s; rfl
      | cons e es ih => intro s; exact ih _
    exact h _ _
  · intro ax
    have h : ∀ (es : List Event) (s : BlockState), (runTagged s es).numPids ax = s.numPids ax := by
      intro es; induction es with
      | nil => intro s; rfl
      | cons e es ih => intro s; exact ih _
    exact h _ _

/-- The whole source grid has a finite physical realization and full physical
frame for arbitrary overlapping read views. Only fresh output writes are
unique. The source success, physical frame and eight outputs are conclusions. -/
theorem grid_return (q : HostReturn.Request) (r : Regions) (stride : Strides)
    (M N : Nat) (eps pHost p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block)
    (a : Layout) (initial : Physical a) (template : BlockState) (hN : N≤block)
    (hsep : Domain.ReadWriteSeparated (HostReturn.flags q pHost) r)
    (hg : WholeGrid.Geometry r stride N)
    (hf : FreshWrites a (Writes (HostReturn.flags q pHost) r stride M N)) :
    ∃ final physical,
      Nonempty (Kernel.GridLaunchedOrdinary
        ((repaired (HostReturn.flags q pHost) r stride M N eps p block hBlock).eraseDType.toAlgKernel)
        (WholeGrid.grid M) (pull a initial template) final) ∧
      pull a physical final = final ∧
      WholeGrid.Post (HostReturn.flags q pHost) r stride M N eps p (pull a initial template) final ∧
      HostReturn.observe (HostReturn.flags q pHost) r stride M N (pull a physical final) =
        HostReturn.oracle (HostReturn.flags q pHost) r stride M N eps p (pull a initial template) ∧
      (∀ i : Fin a.capacity, (∀ k, Writes (HostReturn.flags q pHost) r stride M N k →
        a.address k ≠ i.val) → physical i = initial i) := by
  let f := HostReturn.flags q pHost
  obtain ⟨final,hrun,hpost⟩ := WholeGrid.original_correct f r stride M N eps p block hBlock
    hN hsep hg (pull a initial template)
  let physical := runPhysical a initial (events f r stride M N final)
  have hpull := source_trace_pull a initial template final f r stride M N eps p hf hpost
  refine ⟨final,physical,hrun,hpull,hpost,?_,?_⟩
  · rw [hpull]
    exact HostReturn.observe_eq_oracle f r stride M N eps p _ _ hg hsep hpost
  · intro i hi
    exact trace_frame a initial (events f r stride M N final) i
      (fun e he => Ne.symm (hi e.key (events_written f r stride M N final e he)))

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.AliasGrid
