import bench.tritonbench_g.triton_linear_activation.Proofs.FPSource
import bench.tritonbench_g.triton_linear_activation.Proofs.FPLiterals
import bench.tritonbench_g.sgmv_expand_slice.Proofs.FPFusedSoftware
import bench.tritonbench_g.softmax_reducev.Proofs.FPSoftware
namespace VeriTile.Bench.TritonBenchG.TritonLinearActivation.FPValue
open VeriTile.Triton FP.Structural
open ParallelAttention.FPStages (foldRange range_values)
noncomputable section
abbrev Dot := SgmvExpandSlice.FPValue.Dot
abbrev DotAcc := SgmvExpandSlice.FPValue.DotAcc
abbrev castWord := LoraExpandGemv.FPValue.castWord
abbrev maximumWord := SoftmaxReducev.FPValue.maximumWord

/-- All reached scalar operations and storage conversions are concrete. Dot,
libdevice tanh and erf retain complete, explicit numerical callbacks. -/
def backend (rest : Algebra Word) (cfg : FP.Config) (D : Dot) (F : DotAcc)
    (tanh32 erf32 : Word → Word) (divide : Word → Word → Word) : Algebra Word :=
  let base := SgmvExpandSlice.FPValue.backend (SgmvExpandSlice.FPValue.installAcc rest F) cfg D
  { base with
    literal := fun p d x => literalWord (base.literal p d) x
    binary := fun p d op a b => match d,op with
      | .real,.add | .fp32,.add => FP.add cfg a b
      | .real,.sub | .fp32,.sub => FP.sub cfg a b
      | .real,.mul | .fp32,.mul => FP.mul cfg a b
      | .real,.div | .fp32,.div => divide a b
      | .real,.max | .fp32,.max => maximumWord cfg a b
      | _,_ => base.binary p d op a b
    unary := fun p op x => match op with
      | .tanh => tanh32 x | .erf => erf32 x | _ => base.unary p op x }

def gridRows (p : Parameters) := (p.rows+p.blockRows-1)/p.blockRows
def gridCols (p : Parameters) := (p.columns+p.blockColumns-1)/p.blockColumns
def width (p : Parameters) := p.groupRows*gridCols p
def groupIdx (p : Parameters) (s : State Word) := s.pids 0/width p
def groupSize (p : Parameters) (s : State Word) :=
  if gridRows p-groupIdx p s*p.groupRows < p.groupRows then
    gridRows p-groupIdx p s*p.groupRows else p.groupRows
def blockRow (p : Parameters) (s : State Word) := groupIdx p s*p.groupRows+s.pids 0%groupSize p s
def blockCol (p : Parameters) (s : State Word) := s.pids 0%width p/groupSize p s
def row (p : Parameters) (s : State Word) (i : Fin p.blockRows) := (blockRow p s*p.blockRows+i.val)%p.rows
def col (p : Parameters) (s : State Word) (j : Fin p.blockColumns) := (blockCol p s*p.blockColumns+j.val)%p.columns

def aOffset (p : Parameters) (s : State Word) (cur : Nat) (i : TileIndex [p.blockRows,p.blockInner]) :=
  row p s i.1*p.inputRowStride+i.2.1.val*p.inputInnerStride+cur*p.blockInner*p.inputInnerStride
def bOffset (p : Parameters) (s : State Word) (cur : Nat) (i : TileIndex [p.blockInner,p.blockColumns]) :=
  i.1.val*p.weightInnerStride+col p s i.2.1*p.weightColStride+cur*p.blockInner*p.weightInnerStride
def outputOffset (p : Parameters) (s : State Word) (i : TileIndex [p.blockRows,p.blockColumns]) :=
  row p s i.1*p.outputRowStride+col p s i.2.1*p.outputColStride
def savedOffset (p : Parameters) (s : State Word) (i : TileIndex [p.blockRows,p.blockColumns]) :=
  row p s i.1*p.savedRowStride+col p s i.2.1*p.savedColStride
def active (p : Parameters) (s : State Word) (i : TileIndex [p.blockRows,p.blockColumns]) : Bool :=
  (row p s i.1<p.rows)&&(col p s i.2.1<p.columns)
def readWord [Inhabited Word] (fmt : Format) (s : State Word) (r : RegionName) (o : Nat) : Word :=
  match fmt with
  | .half => (s.mem r o).read .fp16 | .bfloat => (s.mem r o).read .bf16 | .single => (s.mem r o).read .fp32
abbrev Accumulator (p : Parameters) := TileIndex [p.blockRows,p.blockColumns] → Word

def biasValue [Inhabited Word] (cfg : FP.Config) (fmt : Format) (p : Parameters) (s : State Word)
    (j : Fin p.blockColumns) := castWord cfg fmt .single
      (if col p s j<p.columns then readWord fmt s p.bias (col p s j) else FP.Value.zero .fp32)
def initial [Inhabited Word] (cfg : FP.Config) (g : Graph) (fmt : Format) (p : Parameters) (s : State Word) : Accumulator p :=
  fun i => if p.hasBias then
    match g with
    | .source => FP.add cfg (FP.Value.zero .fp32) (biasValue cfg fmt p s i.2.1)
    | .accumulator => FP.add cfg (biasValue cfg fmt p s i.2.1) (FP.Value.zero .fp32)
    else FP.Value.zero .fp32

def aValues [Inhabited Word] (fmt : Format) (p : Parameters) (s : State Word) (cur : Nat) :
    TileIndex [p.blockRows,p.blockInner] → Word := fun i => readWord fmt s p.input (aOffset p s cur i)
def bValues [Inhabited Word] (fmt : Format) (p : Parameters) (s : State Word) (cur : Nat) :
    TileIndex [p.blockInner,p.blockColumns] → Word := fun i => readWord fmt s p.weight (bOffset p s cur i)
def update [Inhabited Word] (cfg : FP.Config) (D : Dot) (F : DotAcc) (g : Graph)
    (fmt : Format) (p : Parameters) (s : State Word) (cur : Nat) (acc : Accumulator p) : Accumulator p :=
  match g with
  | .source => fun i => FP.add cfg (acc i)
      (D.apply (some .fp32) (dtype fmt) (dtype fmt) .fp32 {} (batch := [])
        (aValues fmt p s cur) (bValues fmt p s cur) i)
  | .accumulator => F.apply (some .fp32) (dtype fmt) (dtype fmt) .real {} (batch := [])
        (aValues fmt p s cur) (bValues fmt p s cur) acc

def accumulated [Inhabited Word] (cfg : FP.Config) (D : Dot) (F : DotAcc) (g : Graph)
    (fmt : Format) (p : Parameters) (s : State Word) : Accumulator p :=
  foldRange (update cfg D F g fmt p s) 0 p.blocks 1 (initial cfg g fmt p s)

def geluWord (cfg : FP.Config) (g : Graph) (erf32 : Word → Word)
    (divide : Word → Word → Word) (x : Word) : Word :=
  let e := erf32 (divide x sqrt2Word)
  FP.mul cfg (FP.mul cfg x halfWord)
    (match g with | .source => FP.add cfg oneWord e | .accumulator => FP.add cfg e oneWord)
def fastGeluWord (cfg : FP.Config) (g : Graph) (tanh32 : Word → Word) (x : Word) : Word :=
  match g with
  | .source => FP.mul cfg (FP.mul cfg halfWord x) (FP.add cfg oneWord (tanh32
      (FP.mul cfg sqrt2piWord (FP.add cfg x (FP.mul cfg (FP.mul cfg (FP.mul cfg cubicWord x) x) x)))))
  | .accumulator => FP.mul cfg (FP.mul cfg x halfWord) (FP.add cfg (tanh32
      (FP.mul cfg (FP.add cfg x (FP.mul cfg (FP.mul cfg (FP.mul cfg x cubicWord) x) x)) sqrt2piWord)) oneWord)
def activate (cfg : FP.Config) (g : Graph) (tanh32 erf32 : Word → Word)
    (divide : Word → Word → Word) (name : String) (x : Word) : Word :=
  let x1 := if name=="tanh" then tanh32 x else x
  let x2 := if name=="gelu" then geluWord cfg g erf32 divide x1 else x1
  let x3 := if name=="fast_gelu" then fastGeluWord cfg g tanh32 x2 else x2
  if name=="relu" then
    match g with
    | .source => maximumWord cfg (FP.Value.zero .fp32) x3
    | .accumulator => maximumWord cfg x3 (FP.Value.zero .fp32)
  else x3

def outputCell (fmt : Format) (x : Word) : Cell Word := match fmt with
  | .half => .mk .fp16 x | .bfloat => .mk .bf16 x | .single => .mk .fp32 x

def saveWritten [Inhabited Word] (cfg : FP.Config) (saved : Format) (p : Parameters) (s : State Word)
    (acc : Accumulator p) : State Word :=
  if p.saveInputs then (TileShape.allIndices [p.blockRows,p.blockColumns]).foldl
    (fun t i => t.write p.saved (savedOffset p s i) (outputCell saved (castWord cfg .single saved (acc i)))) s
  else s

def storeResult [Inhabited Word] (cfg : FP.Config) (fmt : Format) (p : Parameters)
    (geometry base : State Word) (acc : Accumulator p) : State Word :=
  (TileShape.allIndices [p.blockRows,p.blockColumns]).foldl
    (fun t i => if active p geometry i then t.write p.output (outputOffset p geometry i)
      (outputCell fmt (castWord cfg .single fmt (acc i))) else t) base

def written [Inhabited Word] (cfg : FP.Config) (g : Graph) (tanh32 erf32 : Word → Word)
    (divide : Word → Word → Word) (fmt saved : Format) (p : Parameters) (s : State Word)
    (acc : Accumulator p) : State Word :=
  storeResult cfg fmt p s (saveWritten cfg saved p s acc)
    (fun i => activate cfg g tanh32 erf32 divide p.activation (acc i))

end
end VeriTile.Bench.TritonBenchG.TritonLinearActivation.FPValue
