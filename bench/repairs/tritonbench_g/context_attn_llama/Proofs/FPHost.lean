import bench.repairs.tritonbench_g.context_attn_llama.Proofs.FPGrid
import bench.repairs.tritonbench_g.context_attn_llama.Proofs.FPTotalBackend
import bench.tritonbench_g.context_attn_llama.Proofs.HostReadback

namespace VeriTile.Bench.Repairs.ContextAttnLlama.FPHost
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnLlama
open FPSuccess
noncomputable section

/-- Logical tensor coordinates. The captured active rows are finite even though
row coordinates themselves are not artificially restricted by a small fixture. -/
def Coordinate (c : Host.Call) := Fin c.batches × Fin c.layout.heads × Nat × Fin c.layout.D

def copyRows (c : Host.Call) (b : Nat) : Nat := min (c.length b) (c.blocks*c.layout.BM)

/-- The Python wrapper computes these ranges from the resolved metadata before
launch. Replacing physical metadata during copyback cannot alter this value. -/
def capture (c : Host.Call) (o : FPViews.Output (Coordinate c)) : FPViews.Output (Coordinate c) :=
  {o with
    active := fun i => i.2.2.1 < copyRows c i.1
    scratchOffset := fun i => c.output i.1 i.2.1 i.2.2.1 i.2.2.2}

theorem captured_writes (c : Host.Call) (o : FPViews.Output (Coordinate c))
    (i : Coordinate c) (hi : (capture c o).active i) :
    writes c (allPrograms c) c.regions.out ((capture c o).scratchOffset i) := by
  have hs : i.2.2.1 < c.length i.1 ∧ i.2.2.1 < c.blocks*c.layout.BM :=
    Nat.lt_min.mp hi
  obtain ⟨idx,_,ha,hoff⟩ := Host.covered c i.1 i.2.1 i.2.2.1 hs.1 hs.2 i.2.2.2
  exact ⟨idx,allPrograms_complete c idx,c.lane i.2.2.1 i.2.2.2,ha,rfl,hoff.symm⟩

theorem grid_writes_captured (c : Host.Call) (o : FPViews.Output (Coordinate c))
    (idx : GridIndex c.grid) (lane : TileIndex [c.layout.BM,c.layout.D])
    (ha : c.active idx lane) :
    ∃ i : Coordinate c, (capture c o).active i ∧
      (capture c o).scratchOffset i = c.offset idx lane := by
  let b : Fin c.batches := ⟨Formula.batch c.layout (c.launchAt idx),Host.batch_bound c idx⟩
  let h : Fin c.layout.heads := ⟨Formula.head c.layout (c.launchAt idx),Host.head_bound c idx⟩
  let row := Formula.queryRow c.layout (c.launchAt idx) lane.1.val
  refine ⟨(b,h,row,lane.2.1),?_,rfl⟩
  apply Nat.lt_min.mpr
  constructor
  · exact ha
  · have hb := Host.block_bound c idx
    have hl := lane.1.isLt
    have hm := Nat.mul_le_mul_right c.layout.BM (Nat.succ_le_iff.mpr hb)
    dsimp [capture,copyRows,row,Formula.queryRow]
    simp only [Nat.succ_mul] at hm
    omega

def scratch (c : Host.Call) (o : FPViews.Output (Coordinate c))
    (t : State (BitVec 32)) (i : Coordinate c) : BitVec 32 :=
  (t.mem c.regions.out ((capture c o).scratchOffset i)).read .real

/-- Observable result of the ordinary `copy_` contract after private launch.
The original supplied object identity and the Python None return are retained. -/
def result (c : Host.Call) (o : FPViews.Output (Coordinate c))
    (t : State (BitVec 32)) :
    Option Unit × Nat × ((RegionName × Nat) → Cell (BitVec 32)) :=
  (none,o.owner,(capture c o).copyBack (scratch c o t))

/-- Whole finite-grid source execution and binary32 logical copyback. `M` and
`D` supply arbitrary total primitive values; no exact arithmetic, numerical
accuracy, source-success or final-output hypothesis appears. Metadata and
scratch separation are ordinary initial-memory/ownership conditions. -/
theorem whole_call (c : Host.Call) (M : Algebra (BitVec 32)) (D : TotalDots (BitVec 32))
    (s : State (BitVec 32)) (hm : Metadata c s) (sep : Separate c)
    (o : FPViews.Output (Coordinate c))
    (unique : ∀ i j, (capture c o).active i → (capture c o).active j →
      o.address i = o.address j → i = j) :
    ∃ t, runGrid c (D.install M) (allPrograms c) s = some t ∧
      (∀ i, (capture c o).active i →
        (capture c o).logical (result c o t).2.2 i = scratch c o t i) ∧
      (∀ rg n, ¬ writes c (allPrograms c) rg n → t.mem rg n = s.mem rg n) ∧
      (∀ cell, (¬ ∃ i, (capture c o).active i ∧ o.address i = cell) →
        (result c o t).2.2 cell = o.initial cell) ∧
      (result c o t).1 = none ∧ (result c o t).2.1 = o.owner := by
  obtain ⟨t,ht,_,hf⟩ := grid_success_frame c (D.install M) (D.support M c.layout)
    sep (allPrograms c) s hm
  refine ⟨t,ht,?_,hf,?_,rfl,rfl⟩
  · intro i hi
    exact (capture c o).copyBack_read (scratch c o t) unique i hi
  · intro cell hmiss
    exact (capture c o).copyBack_frame (scratch c o t) cell hmiss

end
end VeriTile.Bench.Repairs.ContextAttnLlama.FPHost
