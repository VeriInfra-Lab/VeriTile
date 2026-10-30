import bench.tritonbench_g.chunk_gla_fwd.Proofs.SourceBridge
import bench.tritonbench_g.chunk_gla_fwd.Proofs.ZeroPaddingSafety

/-! Reuse the five established source trace-safety proofs for the repaired
zero-padding bodies. These are proofs of actual evaluated accesses, not merely
declared windows or a post-execution write frame. -/
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.LocalRealSafety
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.ChunkGlaFwd
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 4000000

theorem inter (p : RealInterMath.Layout) (r : RealInterProgram.Regions)
    (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _=>0) (bounds : RegionBounds)
    (hb : RealInterSafety.Bounds p (RealInterProgram.launch s) r bounds) :
    (SourceBridge.interNew p r scale).eraseDType.toAlgKernel.TraceSafe bounds s := by
  rw [Kernel.TraceSafe,SourceBridge.inter_body]
  apply (ZeroPadding.statements_safe _ bounds s ?_ hu).2
  · exact RealInterSafety.source_safe p r .fp32 .fp32 .fp32 .fp32 scale Bool.false s
      (by intro r i; rw [hu]) bounds hb
  · exact (RealInterSafety.source_flatten_ok p r .fp32 .fp32 .fp32 .fp32 scale Bool.false)

theorem intra (p : RealIntraMath.Layout) (r : RealIntraProgram.Regions)
    (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _=>0)
    (hkey : r.key≠r.output) (hgate : r.gate≠r.output)
    (hcols : RealIntraMath.columns p (RealIntraProgram.launch s)≤p.BT)
    (bounds : RegionBounds) (hb : RealIntraSafety.Bounds p (RealIntraProgram.launch s) r bounds) :
    (SourceBridge.intraNew p r scale).eraseDType.toAlgKernel.TraceSafe bounds s := by
  rw [Kernel.TraceSafe,SourceBridge.intra_body]
  apply (ZeroPadding.statements_safe _ bounds s ?_ hu).2
  · exact RealIntraSafety.source_safe p r .fp32 .fp32 .fp32 scale Bool.false s hkey hgate hcols bounds hb
  · exact (RealIntraSafety.source_flatten_ok p r .fp32 .fp32 .fp32 scale Bool.false)

theorem split (p : RealSplitMath.Layout) (r : RealSplitProgram.Regions)
    (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _=>0)
    (hkey : r.key≠r.output) (hgate : r.gate≠r.output)
    (bounds : RegionBounds) (hb : RealSplitSafety.Bounds p (RealSplitProgram.launch p s) r bounds) :
    (SourceBridge.splitNew p r scale).eraseDType.toAlgKernel.TraceSafe bounds s := by
  rw [Kernel.TraceSafe,SourceBridge.split_body]
  apply (ZeroPadding.statements_safe _ bounds s ?_ hu).2
  · exact RealSplitSafety.source_safe p r .fp32 .fp32 .fp32 scale Bool.false s hkey hgate bounds hb
  · exact (RealSplitSafety.source_flatten_ok p r .fp32 .fp32 .fp32 scale Bool.false)

theorem merge (p : RealMergeProgram.Layout) (input output : RegionName)
    (s : BlockState) (hu : s.undef=fun _ _=>0) (bounds : RegionBounds)
    (hb : RealMergeSafety.Bounds p (RealMergeProgram.launch s) input output bounds) :
    (SourceBridge.mergeNew p input output).eraseDType.toAlgKernel.TraceSafe bounds s := by
  rw [Kernel.TraceSafe,SourceBridge.merge_body]
  apply (ZeroPadding.statements_safe _ bounds s ?_ hu).2
  · exact RealMergeSafety.source_safe p input output .fp32 Bool.false s bounds hb
  · exact (RealMergeSafety.source_flatten_ok p input output .fp32 Bool.false)

theorem output (p : RealOutputIO.Layout) (r : RealOutputIO.Regions)
    (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _=>0) (bounds : RegionBounds)
    (hb : RealOutputSafety.Bounds s r.query r.value r.gate r.history r.output r.attention
      p.keyHead p.keyTime p.valueHead p.valueTime p.historyHead p.historyKey
      p.T p.K p.V p.BT p.BK p.BV bounds) :
    (SourceBridge.outputNew p r scale).eraseDType.toAlgKernel.TraceSafe bounds s := by
  rw [Kernel.TraceSafe,SourceBridge.output_body]
  apply (ZeroPadding.statements_safe _ bounds s ?_ hu).2
  · exact RealOutputSafety.source_safe s .fp32 .fp32 .fp32 .fp32 .fp32
      r.query r.value r.gate r.history r.output r.attention
      p.keyHead p.keyTime p.valueHead p.valueTime p.historyHead p.historyKey
      scale p.T p.K p.V p.BT p.BK p.BV bounds hb hu
  · exact (RealOutputSafety.source_flatten_ok .fp32 .fp32 .fp32 .fp32 .fp32
      r.query r.value r.gate r.history r.output r.attention
      p.keyHead p.keyTime p.valueHead p.valueTime p.historyHead p.historyKey
      scale p.T p.K p.V p.BT p.BK p.BV)

#axiomsClean inter
#axiomsClean intra
#axiomsClean split
#axiomsClean merge
#axiomsClean output
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.LocalRealSafety
