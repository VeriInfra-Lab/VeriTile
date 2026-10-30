import bench.tritonbench_g.rotary_emb.Proofs.HostCoverage

namespace VeriTile.Bench.TritonBenchG.RotaryEmb.HostGrid
open VeriTile Triton FullMath
noncomputable section

/-- Independent whole-launch result. Unvisited logical K heads and odd/partial
dimension tails are preserved as raw cells, not only decoded real values. -/
def Plan.Output (c : Plan) (Q K Cos Sin : RegionName) (s t : BlockState) : Prop :=
  (∀ x n h d, c.covered x n h d →
    t.readMem (region Q K x) ((c.tensor x).address n h d) =
      c.logicalResult Q K Cos Sin s x n h d) ∧
  BlockState.WriteWithin (c.foot Q K c.schedule) s t ∧
  (∀ x n h d, n < (c.tensor x).length → h < (c.tensor x).heads →
    d < (c.tensor x).width → c.headBlocks*4 ≤ h ∨ c.evenWidth ≤ d →
    t.mem (region Q K x) ((c.tensor x).address n h d) =
      s.mem (region Q K x) ((c.tensor x).address n h d))

/-- Public host-to-grid theorem for the original 34-statement source in the
retained exact-real semantics. The trusted conversion is explicit; all kernel
success, trace safety, grid coverage, initial-memory values and frames are
proved. Native integer/arange admission and rounded arithmetic are separate. -/
specification rotary_emb_host_grid_correctness
    (op : PythonIntProduct) (a : Request) (c : Plan) (hp : prepare op a = some c)
    (q k cos sin : FullSource.Storage) (Q K Cos Sin : RegionName)
    (A : FlatAlloc) (hd : A.Disjoint) (hcov : ∀ r, r∉A.regions → A.extent r=0)
    (D : Domain c A.extent Q K Cos Sin) (hQr : Q∈A.regions) (hKr : K∈A.regions)
    (s : BlockState) :
    c.request = a ∧ a.assertions ∧
    op a.q.width a.partialRotaryFactor = some c.signedHeadDim ∧
    (c.parameters.dimension : Int) = c.signedHeadDim ∧
    c.parameters.blockHead = 4 ∧ c.parameters.blockSeq = 16 ∧
    c.grid.dims = [(a.q.heads+3)/4,(a.q.length+15)/16] ∧
    c.numWarps = (if c.signedHeadDim ≥ 128 then 8 else 4) ∧ c.numStages = 1 ∧
    c.SafeRun (c.kernel q k cos sin Q K Cos Sin) A.extent c.schedule s ∧
    ∃ t, c.run (c.kernel q k cos sin Q K Cos Sin) c.schedule s = some t ∧
      c.Output Q K Cos Sin s t := by
  obtain ⟨ha,has,hop,hbh,hbs,hg,hw,hst⟩ := prepare_arguments op a c hp
  obtain ⟨hs,t,ht,hout⟩ := grid_correct c q k cos sin Q K Cos Sin A hd hcov D hQr hKr s
  exact ⟨ha,has,hop,dimension_binding c D.headDim_nonneg,hbh,hbs,hg,hw,hst,hs,t,ht,hout⟩

#specNonCircular Plan.Output avoiding [Plan.run,Plan.kernel,exec]
#axiomsClean prepare_arguments
#axiomsClean dimension_binding
#axiomsClean address_injective
#axiomsClean block_geometry
#axiomsClean block_bounds
#axiomsClean block_disjoint
#axiomsClean result_unchanged
#axiomsClean block_run
#axiomsClean run_correct
#axiomsClean rotary_emb_host_grid_correctness
#print rotary_emb_host_grid_correctness

end
end VeriTile.Bench.TritonBenchG.RotaryEmb.HostGrid
