import VeriTile.Triton.Source
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.Mathematical_RealSource

/-! Symbolic parameters and named slices of the complete source. Splitting at
the contraction loop preserves the in-body chunk limit, the residual branch
and every final store. These source identities assert no execution premise. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealProgram
open VeriTile.Triton
set_option maxRecDepth 18000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

structure Parameters where
  A : RegionName
  Dout : RegionName
  Db : RegionName
  Res : RegionName
  seqlen : Nat
  chunk : Nat
  K : Nat
  groups : Nat
  aBatch : Nat
  aRow : Nat
  aHead : Nat
  aColumn : Nat
  dBatch : Nat
  dChunk : Nat
  dHead : Nat
  dRow : Nat
  dColumn : Nat
  oBatch : Nat
  oRow : Nat
  oHead : Nat
  oColumn : Nat
  rBatch : Nat
  rRow : Nat
  rHead : Nat
  rColumn : Nat
  hasResidual : Bool

def source (aStorage doutStorage : TypedSource.Storage) (swapped : Bool)
    (residualDType outputDType : FloatDType) (BM BN BCS : Nat) (p : Parameters) : ComputeKernel :=
  TypedSource.specialize aStorage doutStorage swapped residualDType outputDType p.A p.Dout p.Db p.Res
    p.seqlen p.chunk p.K p.groups p.aBatch p.aRow p.aHead p.aColumn
    p.dBatch p.dChunk p.dHead p.dRow p.dColumn p.oBatch p.oRow p.oHead p.oColumn
    p.rBatch p.rRow p.rHead p.rColumn BM BN BCS p.hasResidual

def kernel (swapped : Bool) (BM BN BCS : Nat) (p : Parameters) : ComputeKernel :=
  (source .fp32 .fp32 swapped .real .real BM BN BCS p).eraseDType

theorem source_projects (aStorage doutStorage : TypedSource.Storage) (swapped : Bool)
    (residualDType outputDType : FloatDType) (BM BN BCS : Nat) (p : Parameters) :
    (source aStorage doutStorage swapped residualDType outputDType BM BN BCS p).toAlgorithm? =
      .ok (source aStorage doutStorage swapped residualDType outputDType BM BN BCS p).toAlgKernel := by
  cases aStorage <;> cases doutStorage <;> cases swapped <;> rfl

theorem storage_erasure (aStorage doutStorage : TypedSource.Storage) (swapped : Bool)
    (residualDType outputDType : FloatDType) (BM BN BCS : Nat) (p : Parameters) :
    (source aStorage doutStorage swapped residualDType outputDType BM BN BCS p).eraseDType =
      kernel swapped BM BN BCS p :=
  RealSource.storage_erasure aStorage doutStorage swapped residualDType outputDType p.A p.Dout p.Db p.Res
    p.seqlen p.chunk p.K p.groups p.aBatch p.aRow p.aHead p.aColumn
    p.dBatch p.dChunk p.dHead p.dRow p.dColumn p.oBatch p.oRow p.oHead p.oColumn
    p.rBatch p.rRow p.rHead p.rColumn BM BN BCS p.hasResidual

def parts (swapped : Bool) (BM BN BCS : Nat) (p : Parameters) : Source.IntRange :=
  (Source.intRange? (source .fp32 .fp32 swapped .real .real BM BN BCS p).surfaceBody "cs").getD
    ⟨[], .constInt 0, .constInt 0, .constNat 0, [], []⟩

def realPart (code : List ComputeStmt) : List Stmt :=
  Stmt.eraseDTypeList (ComputeKernel.mk [] [] code).toAlgKernel.body

def loopBound (BCS : Nat) : Op .int [] :=
  .truncDiv .nil
    (.add .int .nil (.ref .int [] "chunk_size_limit")
      (.sub .int .nil (.constInt BCS) (.constInt 1)))
    (.constInt BCS)

theorem body_split (swapped : Bool) (BM BN BCS : Nat) (p : Parameters) :
    (kernel swapped BM BN BCS p).toAlgKernel.body =
      realPart (parts swapped BM BN BCS p).before ++
        [.forRangeInt "cs" (.constInt 0) (loopBound BCS) (.constNat 1)
          (realPart (parts swapped BM BN BCS p).body)] ++
      realPart (parts swapped BM BN BCS p).after := by
  cases swapped
  all_goals
    simp [kernel, source, parts, realPart, loopBound, TypedSource.specialize, VeriTile.Bench.Repairs.BmmChunkBwd.Kernels.family,
      VeriTile.Bench.Repairs.BmmChunkBwd.Kernels.sourceKernel,
      VeriTile.Bench.Repairs.BmmChunkBwd.Kernels.dotDType,
      Source.intRange?, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
      ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?_map_alg, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType,
      Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
      MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, Except.bind, bind, eraseDType_float]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealProgram
