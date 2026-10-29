import bench.tritonbench_g.sgmv_expand_slice.Proofs.FPSource
import bench.tritonbench_g.lora_expand_gemv.Proofs.FPModel
import bench.tritonbench_g.parallel_attention.Proofs.FPRange
namespace VeriTile.Bench.TritonBenchG.SgmvExpandSlice.FPValue
open VeriTile.Triton FP.Structural
open ParallelAttention.FPStages (foldRange range_values)
noncomputable section
abbrev Word := FP.Value .fp32
namespace Convert
abbrev Format := LoraExpandGemv.FPValue.OutputFormat
abbrev cast := LoraExpandGemv.FPValue.castWord
end Convert

def storageFormat : Storage → Convert.Format
  | .half => .half | .bfloat => .bfloat
def outputFormat : Output → Convert.Format
  | .half => .half | .bfloat => .bfloat | .single => .single

/-- The complete typed-dot primitive. Actual operand/output formats, options,
precision and layout are part of the callback; no widening or scalar sum is
silently substituted for a native low-precision tensor dot. -/
structure Dot where
  apply : Option ComputeDType → (left right output : FloatDType) → DotOptions →
    {batch : TileShape} → {m k n : Nat} →
    (TileIndex (batch++[m,k]) → Word) → (TileIndex (batch++[k,n]) → Word) →
    (TileIndex (batch++[m,n]) → Word)

def backend (rest : Algebra Word) (cfg : FP.Config) (D : Dot) : Algebra Word :=
  let scalar := LoraExpandGemv.FPValue.backend rest cfg FP.Equational.ReductionPlan.withSeed
  { scalar with
    dotTyped := fun p l r o opts {_} {_} {_} {_} a b =>
      some (fun i => ofFloat o (D.apply p l r o opts (fun j => toFloat l (a j)) (fun j => toFloat r (b j)) i)) }

def seqLen [Inhabited Word] (p : Parameters) (s : State Word) := (s.mem p.lengths.name (s.pids 2)).read .nat
def seqStart [Inhabited Word] (p : Parameters) (s : State Word) := (s.mem p.starts.name (s.pids 2)).read .nat
def adapter [Inhabited Word] (p : Parameters) (s : State Word) := (s.mem p.adapters.name (s.pids 2)).read .nat

def row (p : Parameters) (s : State Word) (i : Fin p.blockRows) := i.val+s.pids 0*p.blockRows
def col (p : Parameters) (s : State Word) (j : Fin p.blockColumns) := j.val+s.pids 1*p.blockColumns

def aOffset [Inhabited Word] (p : Parameters) (s : State Word) (cur : Nat)
    (i : TileIndex [p.blockRows,p.blockInner]) :=
  seqStart p s*p.inputRowStride+(row p s i.1 % seqLen p s)*p.inputRowStride+
    i.2.1.val*p.inputInnerStride+cur*p.inputInnerStride

def bOffset [Inhabited Word] (p : Parameters) (s : State Word) (cur : Nat)
    (i : TileIndex [p.blockInner,p.blockColumns]) :=
  p.adapterStride*adapter p s+i.1.val*p.weightInnerStride+
    (col p s i.2.1 % p.columns)*p.weightColumnStride+cur*p.weightInnerStride

def outputOffset [Inhabited Word] (p : Parameters) (s : State Word)
    (i : TileIndex [p.blockRows,p.blockColumns]) :=
  (seqStart p s+row p s i.1)*p.outputRowStride+
    (col p s i.2.1+p.sliceOffset)*p.outputColumnStride

def active [Inhabited Word] (p : Parameters) (s : State Word)
    (i : TileIndex [p.blockRows,p.blockColumns]) : Bool :=
  (row p s i.1 < seqLen p s) && (col p s i.2.1 < p.columns)

def readWord [Inhabited Word] (fmt : Storage) (s : State Word) (r : RegionName) (o : Nat) : Word :=
  match fmt with | .half => (s.mem r o).read .fp16 | .bfloat => (s.mem r o).read .bf16

def aValues [Inhabited Word] (fmt : Storage) (p : Parameters) (s : State Word) (cur : Nat) :
    TileIndex [p.blockRows,p.blockInner] → Word := fun i => readWord fmt s p.input (aOffset p s cur i)
def bValues [Inhabited Word] (fmt : Storage) (p : Parameters) (s : State Word) (cur : Nat) :
    TileIndex [p.blockInner,p.blockColumns] → Word := fun i => readWord fmt s p.weight (bOffset p s cur i)

abbrev Accumulator (p : Parameters) := TileIndex [p.blockRows,p.blockColumns] → Word

def update [Inhabited Word] (cfg : FP.Config) (D : Dot) (fmt : Storage) (p : Parameters)
    (s : State Word) (cur : Nat) (acc : Accumulator p) : Accumulator p :=
  let delta := D.apply (some .fp32) (storageType fmt) (storageType fmt) .fp32 {} (batch := [])
    (aValues fmt p s cur) (bValues fmt p s cur)
  fun i => FP.add cfg (acc i) (delta i)

def initial (p : Parameters) : Accumulator p := fun _ => FP.Value.zero .fp32

def accumulated [Inhabited Word] (cfg : FP.Config) (D : Dot) (fmt : Storage) (p : Parameters)
    (s : State Word) : Accumulator p := foldRange (update cfg D fmt p s) 0 p.inner p.blockInner (initial p)

/-- Original two storage boundaries: first round the fp32 accumulator to the
weight format, then apply the output tensor's store conversion. Direct exact
value conversion is used, with no invented fp32 multiplication or extra round. -/
def outputValues (cfg : FP.Config) (fmt : Storage) (out : Output) (p : Parameters)
    (acc : Accumulator p) : Accumulator p := fun i =>
  Convert.cast cfg (storageFormat fmt) (outputFormat out)
    (Convert.cast cfg .single (storageFormat fmt) (acc i))

def outputCell (out : Output) (x : Word) : Cell Word := match out with
  | .half => .mk .fp16 x | .bfloat => .mk .bf16 x | .single => .mk .fp32 x

def written [Inhabited Word] (out : Output) (p : Parameters) (s : State Word) (v : Accumulator p) : State Word :=
  (TileShape.allIndices [p.blockRows,p.blockColumns]).foldl
    (fun t i => if active p s i then t.write p.output (outputOffset p s i) (outputCell out (v i)) else t) s

def Post [Inhabited Word] (out : Output) (p : Parameters) (s final : State Word) (v : Accumulator p) : Prop :=
  (∀ i, active p s i=Bool.true → final.mem p.output (outputOffset p s i)=outputCell out (v i)) ∧
  (∀ r o, (r≠p.output ∨ ∀ i, active p s i=Bool.true → o≠outputOffset p s i) → final.mem r o=s.mem r o)
end
end VeriTile.Bench.TritonBenchG.SgmvExpandSlice.FPValue
