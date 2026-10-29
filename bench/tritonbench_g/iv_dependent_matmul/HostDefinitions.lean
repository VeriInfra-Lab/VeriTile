import bench.tritonbench_g.iv_dependent_matmul.ScheduleDefinitions
namespace VeriTile.Bench.TritonBenchG.IvDependentMatmul.FPValue
open VeriTile.Triton FP.Structural
abbrev Word := FP.Value .fp32
end VeriTile.Bench.TritonBenchG.IvDependentMatmul.FPValue
namespace VeriTile.Bench.TritonBenchG.IvDependentMatmul.Whole
open VeriTile.Triton FP.Structural FPValue Kernels
structure Request where
  (A B C : RegionName)
  (rows inner columns blockRows blockColumns blockInner : Nat)
  positiveRowsBlock : 0<blockRows
  positiveColumnsBlock : 0<blockColumns
  positiveInnerBlock : 0<blockInner
  schedule : Schedule
  device : String

def Request.parameters (r : Request) : Parameters :=
  ⟨r.A,r.B,r.C,r.rows,r.columns,r.inner,max r.inner 1,1,max r.columns 1,1,
    max r.columns 1,1,r.blockRows,r.blockColumns,r.blockInner⟩
def tileRows (r : Request) := (r.rows+r.blockRows-1)/r.blockRows
def tileColumns (r : Request) := (r.columns+r.blockColumns-1)/r.blockColumns
def programs (r : Request) := tileRows r*tileColumns r

structure TensorMetadata where
  region : RegionName
  (shape strides : List Nat)
  dtype : FloatDType
  device : String

def inputMetadata (r : Request) : TensorMetadata :=
  ⟨r.A,[r.rows,r.inner],[max r.inner 1,1],.fp32,r.device⟩
def weightMetadata (r : Request) : TensorMetadata :=
  ⟨r.B,[r.inner,r.columns],[max r.columns 1,1],.fp32,r.device⟩
def outputMetadata (r : Request) : TensorMetadata :=
  ⟨r.C,[r.rows,r.columns],[max r.columns 1,1],.fp32,r.device⟩

abbrev CIndex (r : Request) := TileIndex [r.blockRows,r.blockColumns]
end VeriTile.Bench.TritonBenchG.IvDependentMatmul.Whole
