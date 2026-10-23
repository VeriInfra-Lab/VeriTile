import bench.repairs.tritonbench_g.chunk_gla_fwd.Kernels
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.ZeroPadding
import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealOutputIO
import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealInterIO
import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealIntraIO
import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealSplitIO
import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealMergeIO

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.SourceBridge
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

def interOld (p : VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealInterMath.Layout) (r : VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealInterProgram.Regions) (scale : ℝ) : ComputeKernel :=
  VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedInter.original .fp32 .fp32 .fp32 .fp32 r.query r.key r.gate r.output p.headStride p.timeStride scale p.T p.K p.BT p.BC p.BK p.NC

def interNew (p : VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealInterMath.Layout) (r : VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealInterProgram.Regions) (scale : ℝ) : ComputeKernel :=
  VeriTile.Bench.Repairs.ChunkGlaFwd.TypedInter.original .fp32 .fp32 .fp32 .fp32 r.query r.key r.gate r.output p.headStride p.timeStride scale p.T p.K p.BT p.BC p.BK p.NC

theorem inter_body (p : VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealInterMath.Layout) (r : VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealInterProgram.Regions) (scale : ℝ) :
    (interNew p r scale).eraseDType.toAlgKernel.body =
      ZeroPadding.statements (interOld p r scale).eraseDType.toAlgKernel.body := by
  simp only [interNew, interOld, VeriTile.Bench.Repairs.ChunkGlaFwd.TypedInter.original, VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedInter.original,
    VeriTile.Bench.Repairs.ChunkGlaFwd.TypedInter.source, VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedInter.source, ComputeKernel.surfaceBody, ComputeKernel.eraseDType, ComputeKernel.toAlgKernel,
    ComputeKernel.toAlgorithm?, ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType,
    MaskOpt.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    Except.bind, bind, List.map_cons, List.map_nil, Bool.false_eq_true, ↓reduceIte]
  simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, FloatDType.toTileDType]
  simp only [ZeroPadding.statements, ZeroPadding.statement, ZeroPadding.operation, ZeroPadding.loadZero]
  all_goals rfl

theorem inter_exec (p : VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealInterMath.Layout) (r : VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealInterProgram.Regions) (scale : ℝ) (s : BlockState)
    (hok : StmtList.FlattenOk (interOld p r scale).eraseDType.toAlgKernel.body)
    (hu : s.undef = fun _ _ => 0) :
    exec (interNew p r scale).eraseDType.toAlgKernel s =
      exec (interOld p r scale).eraseDType.toAlgKernel s := by
  unfold exec
  rw [inter_body]
  exact ZeroPadding.statements_eq _ s hok hu

def intraOld (p : VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealIntraMath.Layout) (r : VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealIntraProgram.Regions) (scale : ℝ) : ComputeKernel :=
  VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedIntra.original .fp32 .fp32 .fp32 r.query r.key r.gate r.output p.headStride p.timeStride scale p.T p.K p.BT p.BC p.BK

def intraNew (p : VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealIntraMath.Layout) (r : VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealIntraProgram.Regions) (scale : ℝ) : ComputeKernel :=
  VeriTile.Bench.Repairs.ChunkGlaFwd.TypedIntra.original .fp32 .fp32 .fp32 r.query r.key r.gate r.output p.headStride p.timeStride scale p.T p.K p.BT p.BC p.BK

theorem intra_body (p : VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealIntraMath.Layout) (r : VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealIntraProgram.Regions) (scale : ℝ) :
    (intraNew p r scale).eraseDType.toAlgKernel.body =
      ZeroPadding.statements (intraOld p r scale).eraseDType.toAlgKernel.body := by
  simp only [intraNew, intraOld, VeriTile.Bench.Repairs.ChunkGlaFwd.TypedIntra.original, VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedIntra.original,
    VeriTile.Bench.Repairs.ChunkGlaFwd.TypedIntra.source, VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedIntra.source, ComputeKernel.surfaceBody, ComputeKernel.eraseDType, ComputeKernel.toAlgKernel,
    ComputeKernel.toAlgorithm?, ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType,
    MaskOpt.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    Except.bind, bind, List.map_cons, List.map_nil, Bool.false_eq_true, ↓reduceIte]
  simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, FloatDType.toTileDType]
  simp only [ZeroPadding.statements, ZeroPadding.statement, ZeroPadding.operation, ZeroPadding.loadZero]
  all_goals rfl

theorem intra_exec (p : VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealIntraMath.Layout) (r : VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealIntraProgram.Regions) (scale : ℝ) (s : BlockState)
    (hok : StmtList.FlattenOk (intraOld p r scale).eraseDType.toAlgKernel.body)
    (hu : s.undef = fun _ _ => 0) :
    exec (intraNew p r scale).eraseDType.toAlgKernel s =
      exec (intraOld p r scale).eraseDType.toAlgKernel s := by
  unfold exec
  rw [intra_body]
  exact ZeroPadding.statements_eq _ s hok hu

def splitOld (p : VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealSplitMath.Layout) (r : VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealSplitProgram.Regions) (scale : ℝ) : ComputeKernel :=
  VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedSplit.original .fp32 .fp32 .fp32 r.query r.key r.gate r.output p.headStride p.timeStride scale p.T p.K p.BT p.BC p.BK p.NC

def splitNew (p : VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealSplitMath.Layout) (r : VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealSplitProgram.Regions) (scale : ℝ) : ComputeKernel :=
  VeriTile.Bench.Repairs.ChunkGlaFwd.TypedSplit.original .fp32 .fp32 .fp32 r.query r.key r.gate r.output p.headStride p.timeStride scale p.T p.K p.BT p.BC p.BK p.NC

theorem split_body (p : VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealSplitMath.Layout) (r : VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealSplitProgram.Regions) (scale : ℝ) :
    (splitNew p r scale).eraseDType.toAlgKernel.body =
      ZeroPadding.statements (splitOld p r scale).eraseDType.toAlgKernel.body := by
  simp only [splitNew, splitOld, VeriTile.Bench.Repairs.ChunkGlaFwd.TypedSplit.original, VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedSplit.original,
    VeriTile.Bench.Repairs.ChunkGlaFwd.TypedSplit.source, VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedSplit.source, ComputeKernel.surfaceBody, ComputeKernel.eraseDType, ComputeKernel.toAlgKernel,
    ComputeKernel.toAlgorithm?, ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType,
    MaskOpt.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    Except.bind, bind, List.map_cons, List.map_nil, Bool.false_eq_true, ↓reduceIte]
  simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, FloatDType.toTileDType]
  simp only [ZeroPadding.statements, ZeroPadding.statement, ZeroPadding.operation, ZeroPadding.loadZero]
  all_goals rfl

theorem split_exec (p : VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealSplitMath.Layout) (r : VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealSplitProgram.Regions) (scale : ℝ) (s : BlockState)
    (hok : StmtList.FlattenOk (splitOld p r scale).eraseDType.toAlgKernel.body)
    (hu : s.undef = fun _ _ => 0) :
    exec (splitNew p r scale).eraseDType.toAlgKernel s =
      exec (splitOld p r scale).eraseDType.toAlgKernel s := by
  unfold exec
  rw [split_body]
  exact ZeroPadding.statements_eq _ s hok hu

def mergeOld (p : VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealMergeProgram.Layout) (A A2 : RegionName) : ComputeKernel :=
  VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedMerge.original .fp32 A A2 p.T p.BT p.BC p.NK

def mergeNew (p : VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealMergeProgram.Layout) (A A2 : RegionName) : ComputeKernel :=
  VeriTile.Bench.Repairs.ChunkGlaFwd.TypedMerge.original .fp32 A A2 p.T p.BT p.BC p.NK

theorem merge_body (p : VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealMergeProgram.Layout) (A A2 : RegionName) :
    (mergeNew p A A2).eraseDType.toAlgKernel.body =
      ZeroPadding.statements (mergeOld p A A2).eraseDType.toAlgKernel.body := by
  simp only [mergeNew, mergeOld, VeriTile.Bench.Repairs.ChunkGlaFwd.TypedMerge.original, VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedMerge.original,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType, ComputeKernel.toAlgKernel,
    ComputeKernel.toAlgorithm?, ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType,
    MaskOpt.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    Except.bind, bind, List.map_cons, List.map_nil, Bool.false_eq_true, ↓reduceIte]
  simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, FloatDType.toTileDType]
  simp only [ZeroPadding.statements, ZeroPadding.statement, ZeroPadding.operation, ZeroPadding.loadZero]
  all_goals rfl

theorem merge_exec (p : VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealMergeProgram.Layout) (A A2 : RegionName) (s : BlockState)
    (hok : StmtList.FlattenOk (mergeOld p A A2).eraseDType.toAlgKernel.body)
    (hu : s.undef = fun _ _ => 0) :
    exec (mergeNew p A A2).eraseDType.toAlgKernel s =
      exec (mergeOld p A A2).eraseDType.toAlgKernel s := by
  unfold exec
  rw [merge_body]
  exact ZeroPadding.statements_eq _ s hok hu

def outputOld (p : VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealOutputIO.Layout) (r : VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealOutputIO.Regions) (scale : ℝ) : ComputeKernel :=
  VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedOutput.original .fp32 .fp32 .fp32 .fp32 .fp32 r.query r.value r.gate r.history r.output r.attention p.keyHead p.keyTime p.valueHead p.valueTime p.historyHead p.historyKey scale p.T p.K p.V p.BT p.BK p.BV

def outputNew (p : VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealOutputIO.Layout) (r : VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealOutputIO.Regions) (scale : ℝ) : ComputeKernel :=
  VeriTile.Bench.Repairs.ChunkGlaFwd.TypedOutput.original .fp32 .fp32 .fp32 .fp32 .fp32 r.query r.value r.gate r.history r.output r.attention p.keyHead p.keyTime p.valueHead p.valueTime p.historyHead p.historyKey scale p.T p.K p.V p.BT p.BK p.BV

theorem output_body (p : VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealOutputIO.Layout) (r : VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealOutputIO.Regions) (scale : ℝ) :
    (outputNew p r scale).eraseDType.toAlgKernel.body =
      ZeroPadding.statements (outputOld p r scale).eraseDType.toAlgKernel.body := by
  simp only [outputNew, outputOld, VeriTile.Bench.Repairs.ChunkGlaFwd.TypedOutput.original, VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedOutput.original,
    VeriTile.Bench.Repairs.ChunkGlaFwd.TypedOutput.source, VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedOutput.source, ComputeKernel.surfaceBody, ComputeKernel.eraseDType, ComputeKernel.toAlgKernel,
    ComputeKernel.toAlgorithm?, ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType,
    MaskOpt.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    Except.bind, bind, List.map_cons, List.map_nil, Bool.false_eq_true, ↓reduceIte]
  simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, FloatDType.toTileDType]
  simp only [ZeroPadding.statements, ZeroPadding.statement, ZeroPadding.operation, ZeroPadding.loadZero]
  all_goals rfl

theorem output_exec (p : VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealOutputIO.Layout) (r : VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealOutputIO.Regions) (scale : ℝ) (s : BlockState)
    (hok : StmtList.FlattenOk (outputOld p r scale).eraseDType.toAlgKernel.body)
    (hu : s.undef = fun _ _ => 0) :
    exec (outputNew p r scale).eraseDType.toAlgKernel s =
      exec (outputOld p r scale).eraseDType.toAlgKernel s := by
  unfold exec
  rw [output_body]
  exact ZeroPadding.statements_eq _ s hok hu

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.SourceBridge
