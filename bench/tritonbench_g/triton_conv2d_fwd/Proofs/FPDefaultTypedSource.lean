import bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPTypedSource
import VeriTile.Triton
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.parallel_attention.Proofs.FPRange
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy
namespace VeriTile.Bench.TritonBenchG.TritonConv2dFwd.FPDefault
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false

/-- The fp32 storage profile from the local Python inputs/output. Masked loads
have no `other`: their address-dependent undef is deliberately retained. The
original default fp16=false specialization, both tf32 settings and all three source loops remain intact. -/
def typed (dtype : FloatDType)
    (Input Weight Output : RegionName)
    (batch_dim in_feat_dim in_height in_width out_feat_dim out_height out_width
      input_batch_stride input_in_feat_stride input_height_stride input_width_stride
      weight_out_feat_stride weight_in_feat_stride weight_height_stride weight_width_stride
      output_batch_stride output_out_feat_stride output_height_stride output_width_stride
      kernel_height kernel_width stride_height stride_width padding_height padding_width groups : Nat)
    (_tf32 : Bool)
    (BLOCK_SIZE_BATCH_HEIGHT_WIDTH BLOCK_SIZE_IN_FEAT BLOCK_SIZE_OUT_FEAT : Nat) :
    ComputeKernel := triton {
  batch_height_width_pid = tl.program_id(0)
  out_feat_pid = tl.program_id(1)
  group_pid = tl.program_id(2)
  in_group_dim = $(in_feat_dim) // $(groups)
  out_group_dim = $(out_feat_dim) // $(groups)
  batch_height_width_offset =
    batch_height_width_pid * $(BLOCK_SIZE_BATCH_HEIGHT_WIDTH) +
      tl.arange(0, $(BLOCK_SIZE_BATCH_HEIGHT_WIDTH))
  batch_height_offset = batch_height_width_offset // $(out_width)
  batch_offset = batch_height_offset // $(out_height)
  output_feat_offset = out_feat_pid * $(BLOCK_SIZE_OUT_FEAT) +
    tl.arange(0, $(BLOCK_SIZE_OUT_FEAT))
  output_height_offset = batch_height_offset % $(out_height)
  output_width_offset = batch_height_width_offset % $(out_width)
    Input +=
      ($(input_batch_stride) * batch_offset +
        $(input_in_feat_stride) * group_pid * in_group_dim)[:, None]
    Weight +=
      ($(weight_out_feat_stride) * output_feat_offset +
        $(weight_out_feat_stride) * group_pid * out_group_dim)[None, :]
  accum = tl.zeros([$(BLOCK_SIZE_BATCH_HEIGHT_WIDTH), $(BLOCK_SIZE_OUT_FEAT)], dtype=tl.float32)
  for h in range($(0), $(kernel_height), $(1)) {
    for w in range($(0), $(kernel_width), $(1)) {
      for c in range($(0), in_group_dim, $(BLOCK_SIZE_IN_FEAT)) {
        input_feat_offset = c + tl.arange(0, $(BLOCK_SIZE_IN_FEAT))
        input_height_offset = h - $((padding_height : Int)) +
          $(stride_height) * output_height_offset
        input_width_offset = w - $((padding_width : Int)) +
          $(stride_width) * output_width_offset
          curr_input_pointer = Input +
            ($(input_in_feat_stride) * input_feat_offset)[None, :] +
            ($(input_height_stride) * input_height_offset)[:, None] +
            ($(input_width_stride) * input_width_offset)[:, None]
          curr_weight_pointer = Weight +
            ($(weight_in_feat_stride) * input_feat_offset)[:, None] +
            $(weight_height_stride) * h + $(weight_width_stride) * w
        input_mask = (batch_offset[:, None] < $(batch_dim)) &
          (input_feat_offset[None, :] < in_group_dim) &
          ($((0 : Int)) <= input_height_offset[:, None]) &
          (input_height_offset[:, None] < $(in_height)) &
          ($((0 : Int)) <= input_width_offset[:, None]) &
          (input_width_offset[:, None] < $(in_width))
        weight_mask = (input_feat_offset[:, None] < in_group_dim) &
          (output_feat_offset[None, :] < out_group_dim)
          input_block = tl.load(curr_input_pointer, mask=input_mask, dtype=DTYPE)
          weight_block = tl.load(curr_weight_pointer, mask=weight_mask, dtype=DTYPE)
        accum += tl.dot(input_block, weight_block, allow_tf32=_tf32)
      }
    }
  }
    Output += $(output_batch_stride) * batch_offset[:, None] +
      $(output_out_feat_stride) * (group_pid * out_group_dim + output_feat_offset)[None, :] +
      $(output_height_stride) * output_height_offset[:, None] +
      $(output_width_stride) * output_width_offset[:, None]
    output_mask = (batch_offset[:, None] < $(batch_dim)) &
      (output_feat_offset[None, :] < out_group_dim) &
      (output_height_offset[:, None] < $(out_height)) &
      (output_width_offset[:, None] < $(out_width))
    tl.store(Output, (accum).to(DTYPE), mask=output_mask)
}


abbrev Parameters := FPValue.Parameters

def source (p : Parameters) : ComputeKernel :=
  typed .fp32 p.Input p.Weight p.Output
    p.batch_dim p.in_feat_dim p.in_height p.in_width p.out_feat_dim p.out_height p.out_width p.input_batch_stride p.input_in_feat_stride p.input_height_stride p.input_width_stride p.weight_out_feat_stride p.weight_in_feat_stride p.weight_height_stride p.weight_width_stride p.output_batch_stride p.output_out_feat_stride p.output_height_stride p.output_width_stride p.kernel_height p.kernel_width p.stride_height p.stride_width p.padding_height p.padding_width p.groups p.tf32
    p.BLOCK_SIZE_BATCH_HEIGHT_WIDTH p.BLOCK_SIZE_IN_FEAT p.BLOCK_SIZE_OUT_FEAT

abbrev BHW (p : Parameters) := p.BLOCK_SIZE_BATCH_HEIGHT_WIDTH
abbrev BIN (p : Parameters) := p.BLOCK_SIZE_IN_FEAT
abbrev OF (p : Parameters) := p.BLOCK_SIZE_OUT_FEAT
abbrev AIndex (p : Parameters) := TileIndex [BHW p,BIN p]
abbrev BIndex (p : Parameters) := TileIndex [BIN p,OF p]
abbrev CIndex (p : Parameters) := TileIndex [BHW p,OF p]

def prelude (p : Parameters) := (source p).surfaceBody.take 14
def hBody (p : Parameters) : List ComputeStmt :=
  match (source p).surfaceBody[14]? with
  | some (ComputeStmt.forRange _ _ _ _ code) => code
  | _ => []
def wBody (p : Parameters) : List ComputeStmt :=
  match (hBody p)[0]? with
  | some (ComputeStmt.forRange _ _ _ _ code) => code
  | _ => []
def cBody (p : Parameters) : List ComputeStmt :=
  match (wBody p)[0]? with
  | some (ComputeStmt.forRangeDyn _ _ _ _ code) => code
  | _ => []
def epilogue (p : Parameters) := (source p).surfaceBody.drop 15

/-- Typed fp32-input, fp32-output dot with all original options visible. -/
structure Dot (α : Type) where
  apply : Option ComputeDType → DotOptions → {batch : TileShape} → {m k n : Nat} →
    (TileIndex (batch ++ [m,k]) → α) → (TileIndex (batch ++ [k,n]) → α) →
    (TileIndex (batch ++ [m,n]) → α)

def backend {α : Type} (rest : Algebra α) (D : Dot α) : Algebra α :=
  {rest with dotTyped := fun precision left right out options {batch} {m} {k} {n} a b =>
    match left,right,out with
    | .fp32,.fp32,.fp32 => some (D.apply precision options a b)
    | l,r,o => rest.dotTyped precision l r o options a b}

theorem decomposition (p : Parameters) :
    (source p).surfaceBody = prelude p ++
      [.forRange "h" 0 p.kernel_height 1 (hBody p)] ++ epilogue p := by rfl

theorem hBody_shape (p : Parameters) :
    hBody p = [.forRange "w" 0 p.kernel_width 1 (wBody p)] := by rfl

theorem wBody_shape (p : Parameters) :
    wBody p = [.forRangeDyn "c" (.constNat 0) (.ref .nat [] "in_group_dim")
      (.constNat (BIN p)) (cBody p)] := by rfl

#axiomsClean decomposition
#axiomsClean hBody_shape
#axiomsClean wBody_shape
end
end VeriTile.Bench.TritonBenchG.TritonConv2dFwd.FPDefault
