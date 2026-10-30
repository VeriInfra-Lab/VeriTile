import bench.repairs.tritonbench_g.matmul_dequantize.Proofs.FirstPublicResult
import bench.repairs.tritonbench_g.matmul_dequantize.Proofs.SecondPublicResult
import bench.repairs.tritonbench_g.matmul_dequantize.Proofs.Dequant32IOResult
import bench.repairs.tritonbench_g.matmul_dequantize.Proofs.Dequant16IOResult

namespace VeriTile.Bench.Repairs.MatmulDequantize.FPEquiv
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulDequantizeWhole001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section

/-! Source-to-native observation under the original scalar ISA and compiler
routing plan. Arbitrary symbolic native inputs remain admitted. This is not an
optimization equivalence or a claim that native arithmetic equals Real; there
are no floating transformation assumptions or new experiments. -/

/-- Complete original arbitrary-order source grid, finite typed inputs and
outputs, proved actual-source access safety and exact exterior frame. -/
theorem gptq_kernel_equiv (p : VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedFirst.Parameters)
    (valid : Geometry.Valid (FirstGrid.geometry p))
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (FirstGrid.geometry p))))
    (plan : Native.Plan) (isa : Native.WordISA) :
    (FirstIO.nativeIO p jobs).Observes (FirstIO.observe p plan isa) (FirstIO.nativeTarget p plan isa) :=
  FirstIO.native_observes p valid ha hb hs hz jobs schedule plan isa

#stmtSurfaceSubset gptq_kernel_equiv ⊆
  [VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedFirst.Parameters,
   VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedFirst.Parameters.weight,
   VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedFirst.Parameters.scale,
   VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedFirst.Parameters.zero,
   VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedFirst.Parameters.output,
   RegionName,Geometry.Valid,
   Geometry.count,
   FirstGrid.geometry,
   VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedFirst.Parameters.input,
   Native.Plan,
   Native.WordISA,
   Native.Expr,
   Native.instInhabitedExpr,
   FirstIO.nativeIO,
   FirstIO.observe,
   FirstIO.nativeTarget,
   ProgramIO.Observes]
#specNonCircular FirstIO.nativeTarget avoiding [FP.Structural.exec,Grid.run,runSchedule]
#axiomsClean gptq_kernel_equiv

/-- Complete original arbitrary-order source grid, finite typed inputs and
outputs, proved actual-source access safety and exact exterior frame. -/
theorem s2_kernel_equiv (p : FPTypedSecond.Parameters)
    (valid : Geometry.Valid (SecondGrid.geometry p))
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (SecondGrid.geometry p))))
    (plan : Native.Plan) (isa : Native.WordISA) :
    (SecondIO.nativeIO p jobs).Observes (SecondIO.observe p plan isa) (SecondIO.nativeTarget p plan isa) :=
  SecondIO.native_observes p valid ha hb hs hz jobs schedule plan isa

#stmtSurfaceSubset s2_kernel_equiv ⊆
  [FPTypedSecond.Parameters,
   FPTypedSecond.Parameters.weight,
   FPTypedSecond.Parameters.scale,
   FPTypedSecond.Parameters.zero,
   FPTypedSecond.Parameters.output,
   RegionName,Geometry.Valid,
   Geometry.count,
   SecondGrid.geometry,
   FPTypedSecond.Parameters.input,
   Native.Plan,
   Native.WordISA,
   Native.Expr,
   Native.instInhabitedExpr,
   SecondIO.nativeIO,
   SecondIO.observe,
   SecondIO.nativeTarget,
   ProgramIO.Observes]
#specNonCircular SecondIO.nativeTarget avoiding [FP.Structural.exec,Grid.run,runSchedule]
#axiomsClean s2_kernel_equiv

/-- Complete original arbitrary-order source grid, finite typed inputs and
outputs, proved actual-source access safety and exact exterior frame. -/
theorem dequant32_kernel_equiv (p : Dequant32.Parameters)
    (valid : Geometry.Valid (DequantGrid32.geometry p))
    (hb : p.weight≠p.output) (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (DequantGrid32.geometry p))))
    (plan : Native.Plan) (isa : Native.WordISA) :
    (D32IO.nativeIO p jobs).Observes (D32IO.observe p plan isa) (D32IO.nativeTarget p plan isa) :=
  D32IO.native_observes p valid hb hs hz jobs schedule plan isa

#stmtSurfaceSubset dequant32_kernel_equiv ⊆
  [Dequant32.Parameters,
   Dequant32.Parameters.weight,
   Dequant32.Parameters.scale,
   Dequant32.Parameters.zero,
   Dequant32.Parameters.output,
   RegionName,Geometry.Valid,
   Geometry.count,
   DequantGrid32.geometry,
   Native.Plan,
   Native.WordISA,
   Native.Expr,
   Native.instInhabitedExpr,
   D32IO.nativeIO,
   D32IO.observe,
   D32IO.nativeTarget,
   ProgramIO.Observes]
#specNonCircular D32IO.nativeTarget avoiding [FP.Structural.exec,Grid.run,runSchedule]
#axiomsClean dequant32_kernel_equiv

/-- Complete original arbitrary-order source grid, finite typed inputs and
outputs, proved actual-source access safety and exact exterior frame. -/
theorem dequant16_kernel_equiv (p : VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.Parameters)
    (valid : Geometry.Valid (DequantGrid16.geometry p))
    (hb : p.weight≠p.output) (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (Geometry.count (DequantGrid16.geometry p))))
    (plan : Native.Plan) (isa : Native.WordISA) :
    (D16IO.nativeIO p jobs).Observes (D16IO.observe p plan isa) (D16IO.nativeTarget p plan isa) :=
  D16IO.native_observes p valid hb hs hz jobs schedule plan isa

#stmtSurfaceSubset dequant16_kernel_equiv ⊆
  [VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.Parameters,
   VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.Parameters.weight,
   VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.Parameters.scale,
   VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.Parameters.zero,
   VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.Parameters.output,
   RegionName,Geometry.Valid,
   Geometry.count,
   DequantGrid16.geometry,
   Native.Plan,
   Native.WordISA,
   Native.Expr,
   Native.instInhabitedExpr,
   D16IO.nativeIO,
   D16IO.observe,
   D16IO.nativeTarget,
   ProgramIO.Observes]
#specNonCircular D16IO.nativeTarget avoiding [FP.Structural.exec,Grid.run,runSchedule]
#axiomsClean dequant16_kernel_equiv

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulDequantize.FPEquiv
