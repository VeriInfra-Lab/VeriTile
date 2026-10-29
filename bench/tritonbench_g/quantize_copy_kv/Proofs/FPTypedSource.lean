import bench.tritonbench_g.quantize_copy_kv.QuantizeCopyKv
import bench.tritonbench_g.quantize_copy_kv.Proofs.FPMaxProfile
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy

/-! The complete original head-masked quantizer, with binary16 input/scale
storage, binary32 arithmetic and signed eight-bit conversion. There is no
column mask: every BLOCK_DMODEL column participates. Undefined conversions
use an explicit arbitrary policy; they do not make padded heads fail.
This software profile is separate from native promotion/maximum/cast binding. -/
namespace VeriTile.Bench.TritonBenchG.QuantizeCopyKv.TypedQuantizer
open VeriTile.Triton FP.Structural FP.IntegerConversion
open QuantizeKvTransform
noncomputable section
set_option maxHeartbeats 7000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
abbrev Word := FP.Value .fp32
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

structure Parameters where
  (input output scales : RegionName)
  destination : Region .nat
  (strideKbs strideKh strideKd strideObs strideOh strideOd
    strideSbs strideSh strideSd heads height width : Nat)

def source (p : Parameters) : ComputeKernel :=
  let K := p.input
  let DestLoc := p.destination
  let Out := p.output
  let OutScale := p.scales
  triton {
  tl.region K = tl.float16, Out = tl.int8, OutScale = tl.float16
  cur_index = tl.program_id(0)
  offs_h = tl.arange(0, $(p.height))
  offs_d = tl.arange(0, $(p.width))
  dest_index = tl.load(DestLoc + cur_index)
  src_data = tl.load(K + cur_index * $(p.strideKbs) +
      offs_h[:, None] * $(p.strideKh) + $(p.strideKd) * offs_d[None, :],
    mask=offs_h[:, None] < $(p.heads), other=0.0)
  abs_data = tl.abs(src_data)
  data_scale = ((tl.max(abs_data, axis=1) / 127.0).to(tl.float16))[:, None]
  q_src_data = (src_data / data_scale).to(tl.int8)
  o_ptrs = Out + dest_index * $(p.strideObs) +
    $(p.strideOh) * offs_h[:, None] + $(p.strideOd) * offs_d[None, :]
  os_ptrs = OutScale + dest_index * $(p.strideSbs) + $(p.strideSh) * offs_h[:, None]
  tl.store(o_ptrs, q_src_data, mask=offs_h[:, None] < $(p.heads))
  tl.store(os_ptrs, data_scale, mask=offs_h[:, None] < $(p.heads))
}

def active (p : Parameters) (i : TileIndex [p.height, p.width]) : Prop :=
  i.1.val < p.heads
instance (p : Parameters) : DecidablePred (active p) := fun _ => inferInstanceAs (Decidable (_ < _))

def inputOffset (p : Parameters) (s : State Word) (i : TileIndex [p.height, p.width]) : Nat :=
  s.pids 0 * p.strideKbs + i.1.val * p.strideKh + p.strideKd * i.2.1.val

def loaded (p : Parameters) (s : State Word) (i : TileIndex [p.height, p.width]) : Word :=
  if active p i then (s.mem p.input (inputOffset p s i)).read .fp16
  else FP.Value.zero .fp32

def magnitude (p : Parameters) (s : State Word) (i : TileIndex [p.height, p.width]) : Word :=
  MixedBits.widen (MixedBits.absHalf (loaded p s i))

def maximum (cfg : FP.Config) (p : Parameters) (s : State Word) (g : Fin p.height) : Word :=
  FPProfile.maximum cfg (fun j : Fin p.width => magnitude p s (g, j, PUnit.unit))

def scale (cfg : FP.Config) (p : Parameters) (s : State Word) (g : Fin p.height) : Word :=
  MixedBits.narrow cfg (FP.div cfg (maximum cfg p s g) (FP.Value.ofBits .fp32 0x42fe0000))

def quotient (cfg : FP.Config) (p : Parameters) (s : State Word)
    (i : TileIndex [p.height, p.width]) : Word :=
  FP.div cfg (MixedBits.widen (loaded p s i)) (MixedBits.widen (scale cfg p s i.1))

/-- Every input bit pattern is covered. Only undefined conversion results
retain the explicit policy; the finite in-range branch is checked truncation. -/
def quantized (cfg : FP.Config) (u : Nat → Word → Int) (p : Parameters) (s : State Word)
    (i : TileIndex [p.height, p.width]) : Int :=
  totalSigned (u 8) 8 (quotient cfg p s i)

def InputDefined (cfg : FP.Config) (p : Parameters) (s : State Word) : Prop :=
  ∀ i, active p i → (toSigned 8 (quotient cfg p s i)).isSome

def destination (p : Parameters) (s : State Word) : Nat :=
  (s.mem p.destination.cast (s.pids 0)).read .nat

def outputOffset (p : Parameters) (s : State Word) (i : TileIndex [p.height, p.width]) : Nat :=
  destination p s * p.strideObs + p.strideOh * i.1.val + p.strideOd * i.2.1.val

def scaleOffset (p : Parameters) (s : State Word) (i : TileIndex [p.height, 1]) : Nat :=
  destination p s * p.strideSbs + p.strideSh * i.1.val

def valuesWritten (cfg : FP.Config) (u : Nat → Word → Int) (p : Parameters) (s : State Word) : State Word :=
  (TileShape.allIndices [p.height, p.width]).foldl (fun t i => if active p i then
    t.write p.output (outputOffset p s i) (.mk .int (quantized cfg u p s i)) else t) s

/-- Independent initial-memory reference: all reads and arithmetic use s,
then the two original masked output scatters run in their source order. -/
def written (cfg : FP.Config) (u : Nat → Word → Int) (p : Parameters) (s : State Word) : State Word :=
  (TileShape.allIndices [p.height, 1]).foldl (fun t i => if i.1.val < p.heads then
    t.write p.scales (scaleOffset p s i) (.mk .fp16 (scale cfg p s i.1)) else t)
    (valuesWritten cfg u p s)

def execution (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (p : Parameters) (s : State Word) (hw : 0 < p.width) :
    {t : State Word // FP.Structural.exec (FPProfile.backend rest cfg u) (source p) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [source, FP.Structural.exec, ComputeKernel.surfaceBody,
    FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr, FP.Structural.evalComputeOp,
    FP.Structural.evalOp_unfold, FP.Structural.store, FP.Structural.numeric,
    FP.Structural.bop, FP.Structural.natLt, FPProfile.backend, MixedBits.backend,
    QuantizeKvCopy.TypedQuantize.backend, fp32TotalBackend, fp32Source,
    FP.Software.scalarFP32, FP.Software.fp32Context, ComputeDType.eraseDType,
    FP.Structural.ofFloat, FP.Structural.toFloat, Region.cast, Option.bind,
    TileShape.axisDim, hw]
  try rfl

#axiomsClean execution
end
end VeriTile.Bench.TritonBenchG.QuantizeCopyKv.TypedQuantizer
