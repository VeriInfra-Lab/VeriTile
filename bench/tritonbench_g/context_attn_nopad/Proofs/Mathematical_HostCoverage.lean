import bench.tritonbench_g.context_attn_nopad.Proofs.Mathematical_HostFormula

namespace VeriTile.Bench.TritonBenchG.ContextAttnNopad.Host
open VeriTile.Triton Formula Program
noncomputable section
set_option maxHeartbeats 1000000

def Call.index (c : Call) (b h block : Nat) (hb : b < c.batches)
    (hh : h < c.q.heads) (ht : block < c.blocks) : GridIndex c.grid := by
  rintro ⟨ax, ha⟩
  rcases ax with _ | (_ | (_ | ax))
  · exact ⟨b, hb⟩
  · exact ⟨h, hh⟩
  · exact ⟨block, ht⟩
  · have : ax + 3 < 3 := ha
    omega

@[simp] theorem index_launch (c : Call) (b h block : Nat) (hb : b < c.batches)
    (hh : h < c.q.heads) (ht : block < c.blocks) :
    c.launch (c.index b h block hb hh ht) = ⟨block, b, h⟩ := rfl

theorem blocks_cover (c : Call) : c.maxInputLen ≤ c.blocks * 128 := by
  unfold Call.blocks
  omega

variable {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}

/-- A logical row is selected exactly up to the rounded host grid extent. -/
theorem covered (hm : Memory c r A s) (b h t d : Nat) (hb : b < c.batches)
    (hh : h < c.q.heads) (ht : t < c.length b) (hd : d < c.q.dims)
    (hg : t < c.blocks*128) :
    ∃ idx : GridIndex c.grid, ∃ i : RealIO.Lane c.layout,
      (c.launch idx).batch = b ∧ (c.launch idx).head = h ∧
      queryRow c.layout (c.launch idx) i.1.val = t ∧ i.2.1.val = d ∧
      RealIO.active c.layout (c.launch idx) (inputs s r) i := by
  have hblock : t / 128 < c.blocks := by omega
  let idx := c.index b h (t/128) hb hh hblock
  let i : RealIO.Lane c.layout := (⟨t%128, Nat.mod_lt _ (by decide)⟩, ⟨d, hd⟩, PUnit.unit)
  have htime : queryRow c.layout (c.launch idx) i.1.val = t := by
    simp only [idx, index_launch, queryRow, Call.layout, i]
    omega
  refine ⟨idx, i, rfl, rfl, htime, rfl, ?_⟩
  apply (active_iff hm idx i).mpr
  simpa only [htime] using ht

def Call.inputRegions (_c : Call) (r : Regions) : List RegionName :=
  [r.query, r.key, r.value, r.start, r.sequence]

/-- The wrapper's execution boundary: the complete actual grid is launched over
the caller-owned finite buffers, and Python returns `None`. -/
def Call.Returns (c : Call) (r : Regions) (A : FlatAlloc) (s : BlockState) (swapped : Bool)
    (result : BlockState × Option Unit) : Prop :=
  Nonempty (Kernel.GridLaunchedOrdinary (A.flattenKernel (c.source r swapped).eraseDType.toAlgKernel)
    c.grid (A.flattenState s) result.1) ∧ result.2 = pythonReturn

end
end VeriTile.Bench.TritonBenchG.ContextAttnNopad.Host
