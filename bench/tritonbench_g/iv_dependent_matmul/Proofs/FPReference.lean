import bench.tritonbench_g.iv_dependent_matmul.HostDefinitions
import bench.tritonbench_g.iv_dependent_matmul.Proofs.FPSource
import bench.tritonbench_g.sgmv_expand_slice.Proofs.FPFusedSoftware
namespace VeriTile.Bench.TritonBenchG.IvDependentMatmul.FPValue
open VeriTile.Triton FP.Structural
open ParallelAttention.FPStages (foldRange range_values)
noncomputable section
abbrev Dot := SgmvExpandSlice.FPValue.Dot
abbrev DotAcc := SgmvExpandSlice.FPValue.DotAcc
abbrev castWord := LoraExpandGemv.FPValue.castWord
abbrev softwareDot := SgmvExpandSlice.FPValue.softwareDot
abbrev softwareDotAcc := SgmvExpandSlice.FPValue.softwareDotAcc

def backend (rest : Algebra Word) (cfg : FP.Config) (D : Dot) (F : DotAcc) : Algebra Word :=
  SgmvExpandSlice.FPValue.backend (SgmvExpandSlice.FPValue.installAcc rest F) cfg D

def columnsGrid (p : Parameters) := (p.columns+p.blockColumns-1)/p.blockColumns
def rowBlock (p : Parameters) (s : State Word) := s.pids 0/columnsGrid p
def colBlock (p : Parameters) (s : State Word) := s.pids 0%columnsGrid p
def row (p : Parameters) (s : State Word) (i : Fin p.blockRows) := rowBlock p s*p.blockRows+i.val
def col (p : Parameters) (s : State Word) (j : Fin p.blockColumns) := colBlock p s*p.blockColumns+j.val

def aBase (p : Parameters) (s : State Word) (i : TileIndex [p.blockRows,p.blockInner]) :=
  (row p s i.1%p.rows)*p.inputRowStride+i.2.1.val*p.inputInnerStride
def bBase (p : Parameters) (s : State Word) (i : TileIndex [p.blockInner,p.blockColumns]) :=
  i.1.val*p.weightInnerStride+(col p s i.2.1%p.columns)*p.weightColumnStride
def aOffset (p : Parameters) (s : State Word) (cur : Nat) (i : TileIndex [p.blockRows,p.blockInner]) :=
  aBase p s i+cur*p.blockInner*p.inputInnerStride
def bOffset (p : Parameters) (s : State Word) (cur : Nat) (i : TileIndex [p.blockInner,p.blockColumns]) :=
  bBase p s i+cur*p.blockInner*p.weightInnerStride
def outputOffset (p : Parameters) (s : State Word) (i : TileIndex [p.blockRows,p.blockColumns]) :=
  p.outputRowStride*row p s i.1+p.outputColumnStride*col p s i.2.1
def active (p : Parameters) (s : State Word) (i : TileIndex [p.blockRows,p.blockColumns]) : Bool :=
  (row p s i.1<p.rows)&&(col p s i.2.1<p.columns)
def count (p : Parameters) := (p.inner+p.blockInner-1)/p.blockInner
abbrev Accumulator (p : Parameters) := TileIndex [p.blockRows,p.blockColumns] → Word

def aValues [Inhabited Word] (p : Parameters) (s : State Word) (cur : Nat) :
    TileIndex [p.blockRows,p.blockInner] → Word := fun i =>
  if i.2.1.val<p.inner-cur*p.blockInner then (s.mem p.input (aOffset p s cur i)).read .fp32
  else FP.Value.zero .fp32

def bValues [Inhabited Word] (p : Parameters) (s : State Word) (cur : Nat) :
    TileIndex [p.blockInner,p.blockColumns] → Word := fun i =>
  if i.1.val<p.inner-cur*p.blockInner then (s.mem p.weight (bOffset p s cur i)).read .fp32
  else FP.Value.zero .fp32

def update [Inhabited Word] (cfg : FP.Config) (D : Dot) (F : DotAcc) (g : Graph)
    (p : Parameters) (s : State Word) (cur : Nat) (a : Accumulator p) : Accumulator p :=
  match g with
  | .source => fun i => FP.add cfg (a i)
      (D.apply (some .fp32) .fp32 .fp32 .fp32 {} (batch := []) (aValues p s cur) (bValues p s cur) i)
  | .accumulator => F.apply (some .fp32) .fp32 .fp32 .real {} (batch := [])
      (aValues p s cur) (bValues p s cur) a

def accumulated [Inhabited Word] (cfg : FP.Config) (D : Dot) (F : DotAcc) (g : Graph)
    (p : Parameters) (s : State Word) : Accumulator p :=
  foldRange (update cfg D F g p s) 0 (count p) 1 (fun _ => FP.Value.zero .fp32)

/-- Both physical conversions are required: accumulator -> half register ->
fp32 output tensor. The intermediate half bits are not fp32 bit encodings. -/
def outputValues (cfg : FP.Config) (p : Parameters) (a : Accumulator p) : Accumulator p :=
  fun i => castWord cfg .half .single (castWord cfg .single .half (a i))

def written [Inhabited Word] (cfg : FP.Config) (p : Parameters) (s : State Word)
    (a : Accumulator p) : State Word :=
  (TileShape.allIndices [p.blockRows,p.blockColumns]).foldl
    (fun t i => if active p s i then t.write p.output (outputOffset p s i)
      (.mk .fp32 (outputValues cfg p a i)) else t) s
end
end VeriTile.Bench.TritonBenchG.IvDependentMatmul.FPValue
