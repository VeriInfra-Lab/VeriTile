import bench.tritonbench_g.chunk_gla_simple.Proofs.HostLayout

namespace VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Host
open VeriTile.Triton Math Real
noncomputable section
set_option maxHeartbeats 2000000

def flatKernel (A : FlatAlloc) (c : Shape) (scale : ℝ) (swapped : Bool) : Kernel :=
  A.flattenKernel (kernel c scale swapped).eraseDType.toAlgKernel

theorem perProgram (c : Shape) (scale : ℝ) (swapped : Bool)
    (A : FlatAlloc) (ha : Allocated c A) (s : BlockState)
    (hu : s.undef = fun _ _ => 0) (idx : GridIndex (grid c)) :
    ∃ t, exec (flatKernel A c scale swapped) ((A.flattenState s).withGridIndex idx) = some t ∧
      (∀ i : TileIndex [c.BT,c.BV], active (layout c) (launchIndex c idx) i.1.val i.2.1.val →
        t.readMem A.flat (A.addr "o" (outputAddress (layout c) (launchIndex c idx) i.1.val i.2.1.val)) =
          expected (layout c) (launchIndex c idx) (inputs regions s) scale i.1.val i.2.1.val) ∧
      BlockState.WriteWithin (Flat.footprint A (layout c) (launchIndex c idx) regions)
        ((A.flattenState s).withGridIndex idx) t := by
  have h := Flat.source_executes (layout c) regions .fp32 .fp32 .fp32 .fp32 scale swapped A
    ha.1 ha.2.1 (s.withGridIndex idx) (by simpa using hu)
    (noCollision c _) (by simpa using bounds c idx A ha.2.2)
  simpa only [source_binding,launch_grid,flatKernel] using h

def frames (c : Shape) (scale : ℝ) (swapped : Bool)
    (A : FlatAlloc) (ha : Allocated c A) (s : BlockState) (hu : s.undef = fun _ _ => 0) :
    Kernel.GridFrames (flatKernel A c scale swapped) (grid c) (A.flattenState s) :=
  fun idx => {
    final := (perProgram c scale swapped A ha s hu idx).choose
    writes := Flat.footprint A (layout c) (launchIndex c idx) regions
    h_exec := (perProgram c scale swapped A ha s hu idx).choose_spec.1
    h_writeWithin := (perProgram c scale swapped A ha s hu idx).choose_spec.2.2 }

theorem frames_disjoint (c : Shape) (scale : ℝ) (swapped : Bool)
    (A : FlatAlloc) (ha : Allocated c A) (s : BlockState) (hu : s.undef = fun _ _ => 0) :
    Kernel.GridWritesDisjoint (frames c scale swapped A ha s hu) := by
  intro x y hne cell hx hy
  rcases hx with ⟨_,i,hi,hei⟩
  rcases hy with ⟨_,j,hj,hej⟩
  have he : outputAddress (layout c) (launchIndex c x) i.1.val i.2.1.val =
      outputAddress (layout c) (launchIndex c y) j.1.val j.2.1.val := by
    have h := hei.symm.trans hej
    exact Nat.add_left_cancel h
  exact hne (launchIndex_injective c (address_injective c _ _ i j hi hj he).1)

def outputFootprint (A : FlatAlloc) (c : Shape) : WriteFootprint :=
  fun cell => cell.1 = A.flat ∧ ∃ off, off < c.heads*c.T*c.V ∧ cell.2 = A.addr "o" off

theorem frames_footprint (c : Shape) (scale : ℝ) (swapped : Bool)
    (A : FlatAlloc) (ha : Allocated c A) (s : BlockState) (hu : s.undef = fun _ _ => 0) :
    ∀ cell, Kernel.GridWriteFootprint (frames c scale swapped A ha s hu) cell →
      outputFootprint A c cell := by
  rintro cell ⟨idx,hr,i,hi,he⟩
  refine ⟨hr,_,?_,he⟩
  have hh := (launch_lt c idx).2.2
  simpa [outputAddress,layout] using triple_lt hh hi.1 hi.2

def lane (c : Shape) (ht : 0 < c.BT) (t : Fin c.T) (v : Fin c.V) : TileIndex [c.BT,c.BV] :=
  (⟨t.val%c.BT,Nat.mod_lt _ ht⟩,⟨v.val%c.BV,Nat.mod_lt _ (block_pos c.V)⟩,⟨⟩)

@[simp] theorem owner_launch (c : Shape) (ht : 0 < c.BT) (h : Fin c.heads) (t : Fin c.T) (v : Fin c.V) :
    launchIndex c (owner c ht h t v) = ⟨v.val/c.BV,t.val/c.BT,h.val⟩ := rfl

theorem owner_active (c : Shape) (ht : 0 < c.BT) (h : Fin c.heads) (t : Fin c.T) (v : Fin c.V) :
    active (layout c) (launchIndex c (owner c ht h t v))
      (lane c ht t v).1.val (lane c ht t v).2.1.val := by
  simpa [active,layout,row,column,lane,Nat.div_add_mod'] using And.intro t.isLt v.isLt

theorem owner_address (c : Shape) (ht : 0 < c.BT) (h : Fin c.heads) (t : Fin c.T) (v : Fin c.V) :
    outputAddress (layout c) (launchIndex c (owner c ht h t v))
      (lane c ht t v).1.val (lane c ht t v).2.1.val = h.val*(c.T*c.V)+t.val*c.V+v.val := by
  simp [outputAddress,layout,row,column,lane,Nat.div_add_mod']

/-- Readback of the actual returned contiguous `empty_like(v)` tensor. -/
def returned (A : FlatAlloc) (c : Shape) (s : BlockState)
    (h : Fin c.heads) (t : Fin c.T) (v : Fin c.V) : ℝ :=
  s.readMem A.flat (A.addr "o" (h.val*(c.T*c.V)+t.val*c.V+v.val))

/-- Independent finite Real contraction, on the initial input arrays. No
execution result appears in this definition. The existing zero-extended
mathematical formula covers K blocks and time/value tails. -/
def formula (c : Shape) (ht : 0 < c.BT) (a : Inputs) (scale : ℝ)
    (h : Fin c.heads) (t : Fin c.T) (v : Fin c.V) : ℝ :=
  expected (layout c) (launchIndex c (owner c ht h t v)) a scale
    (lane c ht t v).1.val (lane c ht t v).2.1.val

/-- Successful whole-grid execution, every returned cell and a tagged memory
frame. The launch witness is constructed, never a premise. -/
theorem executes (c : Shape) (ht : 0 < c.BT) (scale : ℝ) (swapped : Bool)
    (A : FlatAlloc) (ha : Allocated c A) (s : BlockState) (hu : s.undef = fun _ _ => 0) :
    ∃ t, Nonempty (Kernel.GridLaunchedOrdinary (flatKernel A c scale swapped)
        (grid c) (A.flattenState s) t) ∧
      (∀ h row v, returned A c t h row v = formula c ht (inputs regions s) scale h row v) ∧
      BlockState.WriteWithin (outputFootprint A c) (A.flattenState s) t := by
  let fs := frames c scale swapped A ha s hu
  let t := Kernel.mergeFrames (grid c) (A.flattenState s) fs
  have hd := frames_disjoint c scale swapped A ha s hu
  refine ⟨t,⟨⟨fs,hd,rfl⟩⟩,?_,?_⟩
  · intro h row v
    let idx := owner c ht h row v
    let i := lane c ht row v
    have hi := owner_active c ht h row v
    have hw : (fs idx).writes (A.flat,A.addr "o"
        (outputAddress (layout c) (launchIndex c idx) i.1.val i.2.1.val)) := ⟨rfl,i,hi,rfl⟩
    have hm := Kernel.mergeFrames_mem_written hd idx _ _ hw
    have hv := (perProgram c scale swapped A ha s hu idx).choose_spec.2.1 i hi
    change (fs idx).final.readMem _ _ = _ at hv
    have hr : t.readMem A.flat (A.addr "o"
        (outputAddress (layout c) (launchIndex c idx) i.1.val i.2.1.val)) =
        expected (layout c) (launchIndex c idx) (inputs regions s) scale i.1.val i.2.1.val := by
      unfold BlockState.readMem
      rw [hm]
      exact hv
    simpa only [idx,i,owner_address,returned,formula] using hr
  · intro region off hn
    exact Kernel.mergeFrames_writeWithin fs region off
      (fun hw => hn (frames_footprint c scale swapped A ha s hu _ hw))

/-- The output allocation cannot overwrite any supplied input. This includes
history and gates; there is no hidden intermediate producer in this wrapper. -/
theorem inputs_stable (c : Shape) (A : FlatAlloc) (ha : Allocated c A) (s t : BlockState)
    (hf : BlockState.WriteWithin (outputFootprint A c) (A.flattenState s) t)
    (r : RegionName) (hr : r ∈ (["q","k","v","h","g"] : List RegionName))
    (off : Nat) (ho : off < capacity c r) :
    t.mem A.flat (A.addr r off) = (A.flattenState s).mem A.flat (A.addr r off) := by
  apply Eq.symm
  apply hf
  rintro ⟨_,o,ho',he⟩
  have hro : r ≠ "o" := by simp only [List.mem_cons,List.not_mem_nil,or_false] at hr; rcases hr with rfl|rfl|rfl|rfl|rfl <;> decide
  have hir : off < A.extent r := lt_of_lt_of_le ho (ha.2.2 r)
  have hio : o < A.extent "o" := lt_of_lt_of_le ho' (by simpa [capacity] using ha.2.2 "o")
  have hmr : r ∈ A.regions := by by_contra hn; rw [ha.2.1 r hn] at hir; omega
  have hmo : ("o" : RegionName) ∈ A.regions := by by_contra hn; rw [ha.2.1 "o" hn] at hio; omega
  have hd := congrArg A.decode he
  rw [A.decode_addr ha.1 hmr hir,A.decode_addr ha.1 hmo hio] at hd
  cases hd
  exact hro rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Host
