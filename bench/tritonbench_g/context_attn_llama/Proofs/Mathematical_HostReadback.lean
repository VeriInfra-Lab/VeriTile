import bench.tritonbench_g.context_attn_llama.Proofs.Mathematical_Host

namespace VeriTile.Bench.TritonBenchG.ContextAttnLlama.Host
open VeriTile.Triton Formula Program Call
noncomputable section
set_option maxHeartbeats 1000000

def Call.indexOf (c : Call) (b : Fin c.batches) (h : Fin c.layout.heads) (m : Fin c.blocks) :
    GridIndex c.grid := fun ax =>
  if h0 : ax.val=0 then ⟨m.val,by simpa [grid,Grid.dim,h0] using m.isLt⟩ else
  if h1 : ax.val=1 then ⟨b.val*c.layout.heads+h.val,by
    simp only [grid,Grid.dim,h1,List.getD_cons_succ,List.getD_cons_zero]
    calc b.val*c.layout.heads+h.val < b.val*c.layout.heads+c.layout.heads := Nat.add_lt_add_left h.isLt _
      _ = (b.val+1)*c.layout.heads := by ring
      _ ≤ c.batches*c.layout.heads := Nat.mul_le_mul_right _ b.isLt⟩ else
    ⟨0,by
      have ha : ax.val=2 := by have := ax.isLt; change ax.val<3 at this; omega
      simp [grid,Grid.dim,ha]⟩

theorem launch_indexOf (c : Call) (b : Fin c.batches) (h : Fin c.layout.heads) (m : Fin c.blocks) :
    c.launchAt (c.indexOf b h m) = ⟨m.val,b.val*c.layout.heads+h.val⟩ := by
  simp [launchAt,indexOf,GridIndex.toPids,grid,Grid.rank]

theorem batch_index (c : Call) (b : Nat) (h : Fin c.layout.heads) (m : Nat) :
    batch c.layout ⟨m,b*c.layout.heads+h.val⟩ = b := by
  simp [batch,Nat.add_div,Nat.div_eq_of_lt h.isLt,c.headsPositive]
  exact Nat.mod_lt _ c.headsPositive

theorem head_index (c : Call) (b : Nat) (h : Fin c.layout.heads) (m : Nat) :
    head c.layout ⟨m,b*c.layout.heads+h.val⟩ = h.val := by
  simp [head,Nat.add_mod,Nat.mod_eq_of_lt h.isLt]

def Call.lane (c : Call) (row : Nat) (d : Fin c.layout.D) : RealIO.Lane c.layout :=
  (⟨row % c.layout.BM,Nat.mod_lt _ (bm_positive c)⟩,d,())

theorem covered (c : Call) (b : Fin c.batches) (h : Fin c.layout.heads)
    (row : Nat) (hr : row < c.length b) (hg : row < c.blocks*c.layout.BM) (d : Fin c.layout.D) :
    ∃ idx : GridIndex c.grid,
      c.launchAt idx = ⟨row/c.layout.BM,b.val*c.layout.heads+h.val⟩ ∧
      c.active idx (c.lane row d) ∧
      c.offset idx (c.lane row d) = c.output b h row d := by
  have hm : row/c.layout.BM < c.blocks := (Nat.div_lt_iff_lt_mul (bm_positive c)).mpr hg
  refine ⟨c.indexOf b h ⟨row/c.layout.BM,hm⟩,launch_indexOf c b h _,?_,?_⟩
  · dsimp only [active]
    rw [launch_indexOf,batch_index]
    simpa [queryRow,lane,Nat.mul_comm,Nat.div_add_mod] using hr
  · dsimp only [offset]
    rw [launch_indexOf,batch_index,head_index]
    simp [queryRow,lane,Nat.mul_comm,Nat.div_add_mod]

/-- If max_input_len covers all lengths, every logical row has a writer.
The general postcondition also describes smaller grids exactly. -/
theorem full_coverage (c : Call) (b : Nat) (hb : c.length b ≤ c.maxInput)
    (row : Nat) (hr : row < c.length b) : row < c.blocks*c.layout.BM := by
  unfold blocks
  rcases c.blockM with hm | hm <;> rw [hm] <;> omega

def Call.logicalExpected (c : Call) (s : BlockState) (b h row d : Nat) : ℝ :=
  Formula.expected c.layout ⟨row/c.layout.BM,b*c.layout.heads+h⟩ (inputs s c.regions) (scale c.layout.D) (row%c.layout.BM) d

def Call.inputs (c : Call) : List RegionName :=
  [c.regions.query,c.regions.key,c.regions.value,c.regions.start,c.regions.sequence,
    c.regions.prompt,c.regions.request,c.regions.pages]

/-- The finite launch, supplied output object and Python None return. -/
def Call.Exec (c : Call) (A : FlatAlloc) (s : BlockState) (swapped : Bool)
    (result : Option Unit × BlockState) : Prop :=
  result.1 = none ∧ Nonempty (Kernel.GridLaunchedOrdinary
    (A.flattenKernel (kernel c.layout c.regions .fp32 swapped (scale c.layout.D)).eraseDType.toAlgKernel)
    c.grid (A.flattenState s) result.2)

/-- Ordinary tensor-coordinate readback, the complete output write frame,
and stability of all bounded Q/K/V and metadata cells. -/
def Call.Result (c : Call) (A : FlatAlloc) (s final : BlockState) : Prop :=
  (∀ b : Fin c.batches, ∀ h : Fin c.layout.heads, ∀ row, row < c.length b →
    row < c.blocks*c.layout.BM → ∀ d : Fin c.layout.D,
    final.readMem A.flat (A.addr c.regions.out (c.output b h row d)) =
      c.logicalExpected s b h row d) ∧
  BlockState.WriteWithin (c.writes A) (A.flattenState s) final ∧
  (∀ rg, rg ∈ c.inputs → ∀ off, off < A.extent rg →
    final.mem A.flat (A.addr rg off) = (A.flattenState s).mem A.flat (A.addr rg off))

theorem whole_call (c : Call) (A : FlatAlloc) (s : BlockState)
    (v : Valid c A s) (ha : A.Disjoint) (hc : ∀ rg, rg ∉ A.regions → A.extent rg = 0)
    (swapped : Bool) : ∃ result, c.Exec A s swapped result ∧ c.Result A s result.2 := by
  obtain ⟨final,he,ho,hf⟩ := whole_grid c A s v ha hc swapped
  refine ⟨(none,final),⟨rfl,he⟩,?_,hf,?_⟩
  · intro b h row hr hg d
    obtain ⟨idx,hl,hi,hoff⟩ := covered c b h row hr hg d
    have hv := ho idx (c.lane row d) hi
    rw [hoff,hl,←RealIO.expected_closed] at hv
    exact hv
  · intro rg hr off hb
    apply Composition.input_stable c.grid A s c.active c.regions.out c.offset ha hc
      (fun idx i hi => v.out _ (batch_bound c idx) _ (head_bound c idx) _ hi _ i.2.1.isLt)
      final hf rg (v.input_separate rg hr) off hb

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnLlama.Host
