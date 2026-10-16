import bench.tritonbench_g.quantize_kv_transform.QuantizeKvTransform
import bench.tritonbench_g.quantize_kv_transform.Proofs.MixedBits
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy
namespace VeriTile.Bench.TritonBenchG.QuantizeKvTransform.TypedTransform
open VeriTile.Triton FP.Structural FP.IntegerConversion
noncomputable section
set_option maxHeartbeats 7000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
abbrev Word := FP.Value .fp32
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
inductive Storage where | fp16 | fp32 deriving DecidableEq, Repr
structure Parameters where
  (input output scales : RegionName)
  destination : Region .nat
  (strideKbs strideKh strideKd strideObs strideOh strideOd strideSbs strideSh
    heads dimension height width : Nat)

def source_fp16 (p : Parameters) : ComputeKernel :=
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
    mask=(offs_h[:, None] < $(p.heads)) & (offs_d[None, :] < $(p.dimension)), other=0.0)
  abs_data = tl.abs(src_data)
  data_scale = ((tl.max(abs_data, axis=1) / 127.0).to(OutScale.dtype.element_ty))[:, None]
  q_src_data = (src_data / data_scale).to(tl.int8)
  o_ptrs = Out + dest_index * $(p.strideObs) +
    $(p.strideOh) * offs_h[:, None] + $(p.strideOd) * offs_d[None, :]
  os_ptrs = OutScale + dest_index * $(p.strideSbs) + $(p.strideSh) * offs_h[:, None]
  tl.store(o_ptrs, q_src_data,
    mask=(offs_h[:, None] < $(p.heads)) & (offs_d[None, :] < $(p.dimension)))
  tl.store(os_ptrs, data_scale, mask=offs_h[:, None] < $(p.heads))
}

def source_fp32 (p : Parameters) : ComputeKernel :=
  let K := p.input
  let DestLoc := p.destination
  let Out := p.output
  let OutScale := p.scales
  triton {
  tl.region K = tl.float32, Out = tl.int8, OutScale = tl.float16
  cur_index = tl.program_id(0)
  offs_h = tl.arange(0, $(p.height))
  offs_d = tl.arange(0, $(p.width))
  dest_index = tl.load(DestLoc + cur_index)
  src_data = tl.load(K + cur_index * $(p.strideKbs) +
      offs_h[:, None] * $(p.strideKh) + $(p.strideKd) * offs_d[None, :],
    mask=(offs_h[:, None] < $(p.heads)) & (offs_d[None, :] < $(p.dimension)), other=0.0)
  abs_data = tl.abs(src_data)
  data_scale = ((tl.max(abs_data, axis=1) / 127.0).to(OutScale.dtype.element_ty))[:, None]
  q_src_data = (src_data / data_scale).to(tl.int8)
  o_ptrs = Out + dest_index * $(p.strideObs) +
    $(p.strideOh) * offs_h[:, None] + $(p.strideOd) * offs_d[None, :]
  os_ptrs = OutScale + dest_index * $(p.strideSbs) + $(p.strideSh) * offs_h[:, None]
  tl.store(o_ptrs, q_src_data,
    mask=(offs_h[:, None] < $(p.heads)) & (offs_d[None, :] < $(p.dimension)))
  tl.store(os_ptrs, data_scale, mask=offs_h[:, None] < $(p.heads))
}

def source : Storage → Parameters → ComputeKernel
  | .fp16 => source_fp16
  | .fp32 => source_fp32
def active (p : Parameters) (i : TileIndex [p.height, p.width]) : Prop :=
  i.1.val < p.heads ∧ i.2.1.val < p.dimension
instance (p : Parameters) : DecidablePred (active p) := fun _ => inferInstanceAs (Decidable (_ ∧ _))
def inputOffset (p : Parameters) (s : State Word) (i : TileIndex [p.height, p.width]) : Nat :=
  s.pids 0 * p.strideKbs + i.1.val * p.strideKh + p.strideKd * i.2.1.val

def loaded (d : Storage) (p : Parameters) (s : State Word)
    (i : TileIndex [p.height, p.width]) : Word :=
  if active p i then
    match d with
    | .fp16 => (s.mem p.input (inputOffset p s i)).read .fp16
    | .fp32 => (s.mem p.input (inputOffset p s i)).read .real
  else FP.Value.zero .fp32

def promoted (d : Storage) (x : Word) : Word :=
  match d with | .fp16 => MixedBits.widen x | .fp32 => x

def absPromoted (d : Storage) (x : Word) : Word :=
  match d with | .fp16 => MixedBits.widen (MixedBits.absHalf x) | .fp32 => x.abs

def maximum (d : Storage) (p : Parameters) (s : State Word) (g : Fin p.height) : Word :=
  QuantizeKvCopy.TypedQuantize.maximum (fun j : Fin p.width =>
    absPromoted d (loaded d p s (g, j, PUnit.unit)))

def scale (cfg : FP.Config) (d : Storage) (p : Parameters) (s : State Word)
    (g : Fin p.height) : Word :=
  MixedBits.narrow cfg (FP.div cfg (maximum d p s g) (FP.Value.ofBits .fp32 0x42fe0000))

def quotient (cfg : FP.Config) (d : Storage) (p : Parameters) (s : State Word)
    (i : TileIndex [p.height, p.width]) : Word :=
  FP.div cfg (promoted d (loaded d p s i)) (MixedBits.widen (scale cfg d p s i.1))

def InputDefined (cfg : FP.Config) (d : Storage) (p : Parameters) (s : State Word) : Prop :=
  ∀ i, active p i → (toSigned 8 (quotient cfg d p s i)).isSome

def expected (cfg : FP.Config) (d : Storage) (p : Parameters) (s : State Word)
    (i : TileIndex [p.height, p.width]) : Int :=
  (toSigned 8 (quotient cfg d p s i)).getD 0

def destination (p : Parameters) (s : State Word) : Nat :=
  (s.mem p.destination.cast (s.pids 0)).read .nat

def outputOffset (p : Parameters) (s : State Word) (i : TileIndex [p.height, p.width]) : Nat :=
  destination p s * p.strideObs + p.strideOh * i.1.val + p.strideOd * i.2.1.val

def scaleOffset (p : Parameters) (s : State Word) (i : TileIndex [p.height, 1]) : Nat :=
  destination p s * p.strideSbs + p.strideSh * i.1.val

def valuesWritten (cfg : FP.Config) (d : Storage) (p : Parameters) (s : State Word) : State Word :=
  (TileShape.allIndices [p.height, p.width]).foldl (fun t i => if active p i then
    t.write p.output (outputOffset p s i) (.mk .int (expected cfg d p s i)) else t) s

def written (cfg : FP.Config) (d : Storage) (p : Parameters) (s : State Word) : State Word :=
  (TileShape.allIndices [p.height, 1]).foldl (fun t i => if i.1.val < p.heads then
    t.write p.scales (scaleOffset p s i) (.mk .fp16 (scale cfg d p s i.1)) else t)
    (valuesWritten cfg d p s)

def execution (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (d : Storage) (p : Parameters) (s : State Word) (hw : 0 < p.width) :
    {t : State Word // FP.Structural.exec (MixedBits.backend rest cfg u) (source d p) s = some t} := by
  cases d
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp [source, source_fp16, source_fp32, FP.Structural.exec, ComputeKernel.surfaceBody,
      FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr, FP.Structural.evalComputeOp,
      FP.Structural.evalOp_unfold, FP.Structural.store, FP.Structural.numeric,
      FP.Structural.bop, FP.Structural.natLt, MixedBits.backend,
      QuantizeKvCopy.TypedQuantize.backend, fp32TotalBackend, fp32Source,
      FP.Software.scalarFP32, FP.Software.fp32Context, ComputeDType.eraseDType,
      FP.Structural.ofFloat, FP.Structural.toFloat, Region.cast, Option.bind,
      TileShape.axisDim, hw]
    try rfl

#axiomsClean execution

end
end VeriTile.Bench.TritonBenchG.QuantizeKvTransform.TypedTransform
