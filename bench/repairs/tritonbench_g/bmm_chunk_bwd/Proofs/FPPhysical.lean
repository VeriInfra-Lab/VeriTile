import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.FPRoles

namespace VeriTile.Bench.Repairs.BmmChunkBwd.FPPhysical
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
open FPDefault FPReference
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 6000000
open Classical

/-- Output occupies a different physical region from each input role.
The direct boundary derives this from admitted disjoint spans, for either the
supplied output or the ordinary fresh allocation. -/
def FreshStorage (h : HostCall) (a : FPFlat.Placement) : Prop :=
  ∀ r, r = h.A ∨ r = h.Dout ∨ r = h.Residual → a.region r ≠ a.region h.Out

def ScratchTyped (h : HostCall) (a : FPFlat.Placement) (s : State α) : Prop :=
  ∀ off, off < a.extent h.Out →
    s.mem (a.region h.Out) (FPFlat.address a h.Out off) =
      .mk .fp32 ((s.mem (a.region h.Out) (FPFlat.address a h.Out off)).read .fp32)

/-- Actual physical load/cast/dot recurrence followed by individual physical
FP32 stores. NativeExecution lowers the arithmetic controlling these accesses. -/
def tile (M : Algebra α) (D : DotPrimitive α) (h : HostCall) (c : Tuning)
    (a : FPFlat.Placement) (q : HostFP.Pid) (physical : State α) : State α :=
  let source := HostFP.launch h c q (FPFlat.lift h a physical)
  (TileShape.allIndices [c.BM,c.BN]).foldl (fun t i => if active source c.BM c.BN (HostContract.parameters h) i then
    t.write (a.region h.Out) (FPFlat.address a h.Out (FPFlat.outOffset h c q physical i))
      (.mk .fp32 (expected M D c.BM c.BN c.BCS (HostContract.parameters h) source i)) else t) physical

def grid (M : Algebra α) (D : DotPrimitive α) (h : HostCall) (c : Tuning) (a : FPFlat.Placement) :
    List HostFP.Pid → State α → State α
  | [],s => s
  | q::rest,s => grid M D h c a rest (tile M D h c a q s)

theorem publish_mem_congr (a : FPFlat.Placement) (out : RegionName) (p s t : State α) (hm : s.mem = t.mem) :
    (FPFlat.publish a out p s).mem = (FPFlat.publish a out p t).mem := by
  simp only [FPFlat.publish,hm]

theorem fold_commutes {ι : Type} (a : FPFlat.Placement) (out : RegionName)
    (initial source physical : State α) (l : List ι) (off : ι → Nat) (value : ι → Cell α)
    (act : ι → Prop) [DecidablePred act] (bounds : ∀ i ∈ l, act i → off i < a.extent out)
    (hm : physical.mem = (FPFlat.publish a out initial source).mem) :
    (l.foldl (fun t i => if act i then t.write (a.region out) (FPFlat.address a out (off i)) (value i) else t) physical).mem =
    (FPFlat.publish a out initial (l.foldl (fun t i => if act i then t.write out (off i) (value i) else t) source)).mem := by
  induction l generalizing source physical with
  | nil => exact hm
  | cons i rest ih =>
    simp only [List.foldl_cons]
    by_cases hi : act i
    · simp only [if_pos hi]
      apply ih
      · intro j hj; exact bounds j (by simp [hj])
      · rw [FPFlat.write_commutes a out initial source (off i) (value i) (bounds i (by simp) hi)]
        simp only [State.write,hm]
    · simp only [if_neg hi]
      exact ih source physical (fun j hj => bounds j (by simp [hj])) hm

theorem lift_published_input (h : HostCall) (a : FPFlat.Placement) (fresh : FreshStorage h a)
    (initial source physical : State α) (hm : physical.mem = (FPFlat.publish a h.Out initial source).mem)
    (r : RegionName) (hr : r = h.A ∨ r = h.Dout ∨ r = h.Residual) :
    (FPFlat.lift h a physical).mem r = (FPFlat.lift h a initial).mem r := by
  funext off
  have hx : ¬ FPFlat.scratchCell a h.Out (a.region r) (FPFlat.address a r off) :=
    fun hh => fresh r hr hh.1
  simp only [FPFlat.lift,hm,FPFlat.publish_frame a h.Out initial source _ _ hx]

def InitialInputs (h : HostCall) (a : FPFlat.Placement) (initial source : State α) : Prop :=
  ∀ r, r = h.A ∨ r = h.Dout ∨ r = h.Residual → source.mem r = (FPFlat.lift h a initial).mem r

theorem tile_commutes (M : Algebra α) (D : DotPrimitive α) (h : HostCall) (c : Tuning)
    (a : FPFlat.Placement) (alloc : FlatAlloc) (layout : HostContract.Layout h c alloc)
    (extent : a.extent = alloc.extent) (fresh : FreshStorage h a)
    (q : HostFP.Pid) (hq : q ∈ HostFP.grid h c) (initial source physical : State α)
    (inputs : InitialInputs h a initial source)
    (hm : physical.mem = (FPFlat.publish a h.Out initial source).mem) :
    (tile M D h c a q physical).mem =
      (FPFlat.publish a h.Out initial
        (FPFinish.write M D c.BM c.BN c.BCS (HostContract.parameters h) (HostFP.launch h c q source))).mem := by
  have he : expected M D c.BM c.BN c.BCS (HostContract.parameters h)
      (HostFP.launch h c q (FPFlat.lift h a physical)) =
      expected M D c.BM c.BN c.BCS (HostContract.parameters h) (HostFP.launch h c q source) := by
    apply FPGridOutput.expected_congr M D c.BM c.BN c.BCS (HostContract.parameters h)
      (HostFP.launch h c q (FPFlat.lift h a physical)) (HostFP.launch h c q source) rfl
    · exact (lift_published_input h a fresh initial source physical hm h.A (Or.inl rfl)).trans (inputs h.A (Or.inl rfl)).symm
    · exact (lift_published_input h a fresh initial source physical hm h.Dout (Or.inr (Or.inl rfl))).trans
        (inputs h.Dout (Or.inr (Or.inl rfl))).symm
    · exact (lift_published_input h a fresh initial source physical hm h.Residual (Or.inr (Or.inr rfl))).trans
        (inputs h.Residual (Or.inr (Or.inr rfl))).symm
  unfold tile
  dsimp only
  rw [he]
  unfold FPFinish.write
  apply fold_commutes
  · intro i hi ha
    rw [extent]
    exact FPFlat.output_bound h c alloc layout q hq physical i ha
  · exact hm

theorem input_frame (M : Algebra α) (D : DotPrimitive α) (h : HostCall) (c : Tuning)
    (a : FPFlat.Placement) (inputs : HostGrid.InputsSeparate (HostContract.parameters h))
    (q : HostFP.Pid) (initial source : State α) (hs : InitialInputs h a initial source) :
    InitialInputs h a initial
      (FPFinish.write M D c.BM c.BN c.BCS (HostContract.parameters h) (HostFP.launch h c q source)) := by
  intro r hr
  funext off
  have hn : r ≠ h.Out := by rcases hr with rfl | rfl | rfl; exact inputs.1; exact inputs.2.1; exact inputs.2.2
  exact (FPOutput.frame M D c.BM c.BN c.BCS (HostContract.parameters h)
    (HostFP.launch h c q source) r off (Or.inl hn)).trans (congrFun (hs r hr) off)

theorem grid_commutes (M : Algebra α) (D : DotPrimitive α) (h : HostCall) (c : Tuning)
    (a : FPFlat.Placement) (alloc : FlatAlloc) (layout : HostContract.Layout h c alloc)
    (extent : a.extent = alloc.extent) (fresh : FreshStorage h a)
    (jobs : List HostFP.Pid) (within : ∀ q ∈ jobs, q ∈ HostFP.grid h c)
    (initial source physical : State α) (inputs : InitialInputs h a initial source)
    (hm : physical.mem = (FPFlat.publish a h.Out initial source).mem) :
    (grid M D h c a jobs physical).mem =
      (FPFlat.publish a h.Out initial (FPGridOutput.reference M D h c jobs source)).mem := by
  induction jobs generalizing source physical with
  | nil => exact hm
  | cons q rest ih =>
    apply ih
    · intro u hu; exact within u (by simp [hu])
    · exact input_frame M D h c a layout.inputsSeparate q initial source inputs
    · exact tile_commutes M D h c a alloc layout extent fresh q (within q (by simp)) initial source physical inputs hm

theorem publish_initial (h : HostCall) (a : FPFlat.Placement) (s : State α)
    (names : h.Out ≠ h.A ∧ h.Out ≠ h.Dout) (typed : ScratchTyped h a s) :
    (FPFlat.publish a h.Out s (FPFlat.lift h a s)).mem = s.mem := by
  funext r off
  by_cases hx : FPFlat.scratchCell a h.Out r off
  · have bound : off-a.base h.Out < a.extent h.Out := by have := hx.2; omega
    have addr : FPFlat.address a h.Out (off-a.base h.Out) = off := by unfold FPFlat.address; have := hx.2.1; omega
    have ht := typed (off-a.base h.Out) bound
    rw [addr,←hx.1] at ht
    simp only [FPFlat.publish,if_pos hx,FPFlat.lift]
    simp only [names.1,names.2,or_self,↓reduceIte,addr,←hx.1]
    exact ht.symm
  · exact FPFlat.publish_frame a h.Out s _ r off hx

/-- Full physical execution is the store-by-store image of the complete
actual typed source grid. The equality is proved from local loads and stores,
not supplied as a kernel-output or flattening assumption. -/
theorem source_grid (M : Algebra α) (D : DotPrimitive α) (h : HostCall) (c : Tuning)
    (precision : FPGrid.DefaultPrecision h) (a : FPFlat.Placement) (alloc : FlatAlloc)
    (layout : HostContract.Layout h c alloc) (extent : a.extent = alloc.extent)
    (fresh : FreshStorage h a) (s : State α) (typed : ScratchTyped h a s) :
    ∃ source, HostFP.run (backend M D) h c Bool.false (HostFP.grid h c) (FPFlat.lift h a s) = some source ∧
      (grid M D h c a (HostFP.grid h c) s).mem = (FPFlat.publish a h.Out s source).mem := by
  obtain ⟨source,he,hm⟩ := FPGridOutput.run_memory M D h c precision (HostFP.grid h c) (FPFlat.lift h a s)
  refine ⟨source,he,?_⟩
  rw [publish_mem_congr a h.Out s _ _ hm]
  apply grid_commutes M D h c a alloc layout extent fresh _ (fun q hq => hq) s (FPFlat.lift h a s) s
  · intro r hr; rfl
  · exact (publish_initial h a s ⟨Ne.symm layout.inputsSeparate.1,Ne.symm layout.inputsSeparate.2.1⟩ typed).symm

#axiomsClean tile_commutes
#axiomsClean grid_commutes
#axiomsClean source_grid
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkBwd.FPPhysical
