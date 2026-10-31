import bench.tritonbench_g.chunk_gla_simple.Sources.Mathematical_HostLayout

namespace VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Host
open VeriTile.Triton Math Real
noncomputable section
set_option maxHeartbeats 2000000

def outputFootprint (A : FlatAlloc) (c : Shape) : WriteFootprint :=
  fun cell => cell.1 = A.flat ∧ ∃ off, off < c.heads*c.T*c.V ∧ cell.2 = A.addr "o" off

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
