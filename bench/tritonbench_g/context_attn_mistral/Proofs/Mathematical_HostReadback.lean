import bench.tritonbench_g.context_attn_mistral.Proofs.Mathematical_Host

namespace VeriTile.Bench.TritonBenchG.ContextAttnMistral.Host
open VeriTile.Triton Formula Program Call
noncomputable section
set_option maxHeartbeats 1000000

def Call.indexOf (c : Call) (b : Fin c.batches) (h : Fin c.heads) (m : Fin c.blocks) :
    GridIndex c.grid := fun ax =>
  if h0 : ax.val=0 then ⟨b.val,by simpa [grid,Grid.dim,h0] using b.isLt⟩ else
  if h1 : ax.val=1 then ⟨h.val,by simpa [grid,Grid.dim,h1] using h.isLt⟩ else
    ⟨m.val,by
      have ha : ax.val=2 := by have := ax.isLt; change ax.val<3 at this; omega
      simpa [grid,Grid.dim,ha] using m.isLt⟩

theorem launch_indexOf (c : Call) (b : Fin c.batches) (h : Fin c.heads) (m : Fin c.blocks) :
    c.launchAt (c.indexOf b h m) = ⟨m.val,b.val,h.val⟩ := by
  simp [launchAt,indexOf,GridIndex.toPids,grid,Grid.rank]

def Call.lane (c : Call) (row : Nat) (d : Fin c.layout.D) : RealIO.Lane c.layout :=
  (⟨row % 128,by rw [c.blockM]; omega⟩,d,())

theorem covered (c : Call) (b : Fin c.batches) (h : Fin c.heads)
    (row : Nat) (hr : row < c.length b) (hg : row < c.blocks*128) (d : Fin c.layout.D) :
    ∃ idx : GridIndex c.grid,
      c.launchAt idx = ⟨row/128,b.val,h.val⟩ ∧
      c.active idx (c.lane row d) ∧
      c.offset idx (c.lane row d) = c.output b h row d := by
  have hm : row/128 < c.blocks := (Nat.div_lt_iff_lt_mul (by omega)).mpr hg
  refine ⟨c.indexOf b h ⟨row/128,hm⟩,launch_indexOf c b h _,?_,?_⟩
  · dsimp only [active]
    rw [launch_indexOf]
    simpa [queryRow,lane,c.blockM,Nat.mul_comm,Nat.div_add_mod] using hr
  · dsimp only [offset]
    rw [launch_indexOf]
    simp [queryRow,lane,c.blockM,Nat.mul_comm,Nat.div_add_mod]

/-- If max_input_len covers all lengths, every logical row has a writer.
The general postcondition also describes smaller grids exactly. -/
theorem full_coverage (c : Call) (b : Nat) (hb : c.length b ≤ c.maxInput)
    (row : Nat) (hr : row < c.length b) : row < c.blocks*128 := by
  unfold blocks
  omega

def Call.logicalExpected (c : Call) (s : BlockState) (b h row d : Nat) : ℝ :=
  Formula.expected c.layout ⟨row/128,b,h⟩ (inputs s c.regions) (scale c.layout.D) (row%128) d

def Call.inputs (c : Call) : List RegionName :=
  [c.regions.query,c.regions.key,c.regions.value,c.regions.start,c.regions.sequence]

/-- The finite launch, supplied output object and Python None return. -/
def Call.Exec (c : Call) (A : FlatAlloc) (s : BlockState) (swapped : Bool)
    (result : Option Unit × BlockState) : Prop :=
  result.1 = none ∧ Nonempty (Kernel.GridLaunchedOrdinary
    (A.flattenKernel (kernel c.layout c.regions .fp32 swapped (scale c.layout.D)).eraseDType.toAlgKernel)
    c.grid (A.flattenState s) result.2)

/-- Ordinary tensor-coordinate readback, the complete output write frame,
and stability of all bounded Q/K/V and metadata cells. -/
def Call.Result (c : Call) (A : FlatAlloc) (s final : BlockState) : Prop :=
  (∀ b : Fin c.batches, ∀ h : Fin c.heads, ∀ row, row < c.length b →
    row < c.blocks*128 → ∀ d : Fin c.layout.D,
    final.readMem A.flat (A.addr c.regions.out (c.output b h row d)) =
      c.logicalExpected s b h row d) ∧
  BlockState.WriteWithin (c.writes A) (A.flattenState s) final ∧
  (∀ rg, rg ∈ c.inputs → ∀ off, off < A.extent rg →
    final.mem A.flat (A.addr rg off) = (A.flattenState s).mem A.flat (A.addr rg off))

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnMistral.Host
