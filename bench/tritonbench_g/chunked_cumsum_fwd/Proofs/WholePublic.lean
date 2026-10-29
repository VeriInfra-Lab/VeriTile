import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.WholeFormula

namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Whole
open VeriTile.Triton Math Program Initialization
noncomputable section
set_option maxHeartbeats 1800000

structure HostResult where
  state : BlockState
  first : RegionName
  second : RegionName
  shape : List Nat

/-- Successful allocation/launch path of the wrapper; the original return
order is cumulative output followed by prepared dt. Allocation identities and
extents are supplied by the ordinary fresh-allocation contract. -/
def runHost (A : FlatAlloc) (c : Host) (r : Regions) (hasBias useSoftplus : Bool)
    (lower : ℝ) (upper : UpperBound) (ls : List Launch) (s : BlockState) : Option HostResult :=
  (runTiles (A.flattenKernel (source32 c r hasBias useSoftplus lower upper)) ls (A.flattenState s)).map
    fun v => ⟨v,r.sum,r.out,[c.batches,c.heads,c.chunks,c.chunk]⟩

def LogicalOutputs (A : FlatAlloc) (c : Host) (r : Regions) (a : Inputs)
    (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound) (v : BlockState) : Prop :=
  ∀ b h k j, b < c.batches → h < c.heads → k < c.chunks → j < c.chunk →
    v.readMem A.flat (A.addr r.out (packed c b h k j)) =
      logicalPrepared c a hasBias useSoftplus lower upper b h k j ∧
    v.readMem A.flat (A.addr r.sum (packed c b h k j)) =
      logicalCumulative c a hasBias useSoftplus lower upper b h k j

theorem logical_outputs (A : FlatAlloc) (c : Host) (r : Regions) (a : Inputs)
    (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound) (v : BlockState)
    (hv : FlatValues A c r a hasBias useSoftplus lower upper v) :
    LogicalOutputs A c r a hasBias useSoftplus lower upper v := by
  intro b h k j hb hh hk hj
  obtain ⟨l,hl,i,t,ha,hb',hh',hk',hj'⟩ := cover c b h k j hb hh hk hj
  have ho := hv l hl i t ha
  have valid := (c.mem_grid l).1 hl
  rw [prepared_logical c l valid,cumulative_logical c l valid _ _ _ _ _ i t ha.1] at ho
  simpa only [sum_eq_out,out_packed,hb',hh',hk',hj'] using ho

/-- Public whole-task mathematical contract. Its numeric interpretation is the
precision erasure of the original FP32 source; IEEE accuracy is not asserted. -/
def HostCorrect (A : FlatAlloc) (c : Host) (r : Regions) (hasBias useSoftplus : Bool)
    (lower : ℝ) (upper : UpperBound) (ls : List Launch) (s : BlockState) : Prop :=
  ∃ result, runHost A c r hasBias useSoftplus lower upper ls s = some result ∧
    result.first = r.sum ∧ result.second = r.out ∧
    result.shape = [c.batches,c.heads,c.chunks,c.chunk] ∧
    LogicalOutputs A c r ⟨s.readMem r.dt,s.readMem r.rate,s.readMem r.bias⟩
      hasBias useSoftplus lower upper result.state ∧
    BlockState.WriteWithin (flatFootprint A (footprint c r c.grid)) (A.flattenState s) result.state ∧
    A.extent r.out = c.outputSize ∧ A.extent r.sum = c.outputSize ∧
    (∀ l, c.valid l → ∀ (i : Fin c.layout.BH) (j : Fin c.layout.BC),
      ∀ x ∈ integerTerms c l i.val j.val,
        (BitVec.ofNat 32 x).toNat = x ∧ (BitVec.ofNat 32 x).toInt = (x : Int)) ∧
    (∀ l, c.valid l → ∀ rg off, Access c r hasBias l rg off →
      byteAddress A rg off % 4 = 0 ∧ byteAddress A rg off + 4 ≤ byteEnd A rg ∧
      byteAddress A rg off + 3 < 2^64)

theorem host_correct (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ rg, rg ∉ A.regions → A.extent rg = 0)
    (c : Host) (r : Regions) (fresh : Fresh r)
    (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound)
    (bounds : AllBounds c r hasBias A) (indices : Signed32Domain c) (bytes : ByteDomain A)
    (outSize : A.extent r.out = c.outputSize) (sumSize : A.extent r.sum = c.outputSize)
    (ls : List Launch) (schedule : ls.Perm c.grid) (s : BlockState) :
    HostCorrect A c r hasBias useSoftplus lower upper ls s := by
  obtain ⟨v,ran,values,frame⟩ := flat_grid_correct A hd hcov c r fresh hasBias useSoftplus lower upper
    bounds ls schedule s
  refine ⟨⟨v,r.sum,r.out,[c.batches,c.heads,c.chunks,c.chunk]⟩,?_,rfl,rfl,rfl,?_,frame,
    outSize,sumSize,?_,?_⟩
  · simp only [runHost,ran,Option.map_some]
  · exact logical_outputs A c r _ hasBias useSoftplus lower upper v values
  · exact source_indices_exact c indices
  · exact all_accesses_full_bytes A bytes hcov c r hasBias bounds

end
end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Whole
