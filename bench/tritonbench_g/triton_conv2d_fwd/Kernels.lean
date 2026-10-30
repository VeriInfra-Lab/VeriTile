import VeriTile.Triton
import VeriTile.Triton.Float.Structural
import Mathlib.Tactic
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.Conv2d
open scoped BigOperators
structure Shape where
  n : Nat
  cg : Nat
  og : Nat
  h : Nat
  w : Nat
  kh : Nat
  kw : Nat
  sh : Nat
  sw : Nat
  ph : Nat
  pw : Nat
  groups : Nat
  oh : Nat
  ow : Nat
  sn : Nat
  sc : Nat
  sx : Nat
  sy : Nat
  so : Nat
  tc : Nat
  tx : Nat
  ty : Nat
  deriving DecidableEq, Repr

def Shape.channels (s : Shape) := s.groups*s.cg
def Shape.outputs (s : Shape) := s.groups*s.og
def Shape.plane (s : Shape) := s.oh*s.ow
def Shape.spatial (s : Shape) := s.n*s.plane
def blocks (n tile : Nat) := (n+tile-1)/tile
def Shape.padded (s : Shape) := blocks s.cg 32*32
def Shape.size (s : Shape) := s.n*s.outputs*s.plane
structure Valid (s : Shape) : Prop where
  cg : 0<s.cg
  og : 0<s.og
  h : 0<s.h
  w : 0<s.w
  kh : 0<s.kh
  kw : 0<s.kw
  sh : 0<s.sh
  sw : 0<s.sw
  groups : 0<s.groups
  oh : 0<s.oh
  ow : 0<s.ow
  shapeH : (s.oh : Int)=((s.h:Int)+2*s.ph-s.kh)/s.sh+1
  shapeW : (s.ow : Int)=((s.w:Int)+2*s.pw-s.kw)/s.sw+1
  strides : 0<s.sn ∧ 0<s.sc ∧ 0<s.sx ∧ 0<s.sy ∧
    0<s.so ∧ 0<s.tc ∧ 0<s.tx ∧ 0<s.ty
structure Flags where
  fp16 : Bool
  tf32 : Bool
  deriving DecidableEq, Repr
structure Cell where
  group : Nat
  spatial : Nat
  feature : Nat
  deriving DecidableEq, Repr
abbrev Program := Nat×Nat×Nat
abbrev Lane := Nat×Nat
def Shape.Active (s : Shape) (x : Cell) :=
  x.group<s.groups ∧ x.spatial<s.spatial ∧ x.feature<s.og
instance (s : Shape) (x : Cell) : Decidable (s.Active x) := inferInstanceAs (Decidable (_∧_∧_))
def cellOf (p : Program) (l : Lane) : Cell := ⟨p.2.2,p.1*128+l.1,p.2.1*32+l.2⟩
def programs (s : Shape) : List Program :=
  (List.range (blocks s.spatial 128)) ×ˢ
    ((List.range (blocks s.outputs 32)) ×ˢ (List.range s.groups))
def lanes : List Lane := (List.range 128) ×ˢ (List.range 32)
def batch (s : Shape) (r : Nat) := (r/s.ow)/s.oh
def outH (s : Shape) (r : Nat) := (r/s.ow)%s.oh
def outW (s : Shape) (r : Nat) := r%s.ow
def ih (s : Shape) (r h : Nat) : Int := (h:Int)-s.ph+s.sh*outH s r
def iw (s : Shape) (r w : Nat) : Int := (w:Int)-s.pw+s.sw*outW s r
/-- Complete signed source expression; conversion is after masked nonnegativity. -/
def inputPointer (s : Shape) (g r h w c : Nat) : Int :=
  s.sn*batch s r+s.sc*g*s.cg+s.sc*c+s.sx*ih s r h+s.sy*iw s r w
def weightPointer (s : Shape) (g f h w c : Nat) : Nat :=
  s.so*f+s.so*g*s.og+s.tc*c+s.tx*h+s.ty*w
def outputPointer (s : Shape) (x : Cell) : Nat :=
  batch s x.spatial*s.outputs*s.oh*s.ow+
    (x.group*s.og+x.feature)*s.oh*s.ow+outH s x.spatial*s.ow+outW s x.spatial
inductive Region where
  | input | weight | output | other (id : Nat)
  deriving DecidableEq, Repr
abbrev Memory (α : Type) := Region → Nat → α
abbrev Mat (m n : Nat) (α : Type) := Fin m → Fin n → α
abbrev Frag (α : Type) := Mat 64 32 α
/-- Fixed instruction ports: scalar FMA, one scalar FP16 RN conversion, and
one literal SM90 WGMMA instruction at each of its two fixed legal sizes.
There is no configurable dot/convolution or output-valued port. -/
structure ISA (α β : Type) where
  zero : α
  fma : α → α → α → α
  halfRN : α → β
  wgmma16 : Mat 64 16 β → Mat 16 32 β → Frag α → Frag α
  wgmma8 : Mat 64 8 α → Mat 8 32 α → Frag α → Frag α

def aLoad {α β} (s : Shape) (p : ISA α β) (m : Memory α)
    (g r h w c : Nat) : α :=
  if batch s r<s.n ∧ c<s.cg ∧ 0 ≤ ih s r h ∧ ih s r h<s.h ∧
      0 ≤ iw s r w ∧ iw s r w<s.w then
    m .input (inputPointer s g r h w c).toNat else p.zero
def bLoad {α β} (s : Shape) (p : ISA α β) (m : Memory α)
    (g f h w c : Nat) : α :=
  if c<s.cg ∧ f<s.og then m .weight (weightPointer s g f h w c) else p.zero

def fmaLanes {α β} (p : ISA α β) (a : Mat 64 32 α) (b : Mat 32 32 α) :
    Nat → Frag α → Frag α
  | 0,carry => carry
  | k+1,carry => fun i j => p.fma (a i ⟨k%32,Nat.mod_lt _ (by decide)⟩)
      (b ⟨k%32,Nat.mod_lt _ (by decide)⟩ j) (fmaLanes p a b k carry i j)
def mma16Step {α β} (p : ISA α β) (a : Mat 64 32 α) (b : Mat 32 32 α)
    (chunk : Nat) (carry : Frag α) : Frag α :=
  p.wgmma16 (fun i k => p.halfRN (a i ⟨(chunk*16+k.val)%32,Nat.mod_lt _ (by decide)⟩))
    (fun k j => p.halfRN (b ⟨(chunk*16+k.val)%32,Nat.mod_lt _ (by decide)⟩ j)) carry
def mma8Step {α β} (p : ISA α β) (a : Mat 64 32 α) (b : Mat 32 32 α)
    (chunk : Nat) (carry : Frag α) : Frag α :=
  p.wgmma8 (fun i k => a i ⟨(chunk*8+k.val)%32,Nat.mod_lt _ (by decide)⟩)
    (fun k j => b ⟨(chunk*8+k.val)%32,Nat.mod_lt _ (by decide)⟩ j) carry
/-- PTX order: two K16 instructions or four K8 instructions, with the previous
FP32 accumulator as input. fp16 takes precedence over tf32, as in source. -/
def dot32 {α β} (p : ISA α β) (flags : Flags) (a : Mat 64 32 α)
    (b : Mat 32 32 α) (carry : Frag α) : Frag α :=
  if flags.fp16 then mma16Step p a b 1 (mma16Step p a b 0 carry)
  else if flags.tf32 then mma8Step p a b 3
    (mma8Step p a b 2 (mma8Step p a b 1 (mma8Step p a b 0 carry)))
  else fmaLanes p a b 32 carry

def sourceStep {α β} (s : Shape) (p : ISA α β) (flags : Flags) (m : Memory α)
    (g r f h w cb : Nat) (carry : Frag α) :=
  dot32 p flags (fun i k => aLoad s p m g (r+i.val) h w (cb*32+k.val))
    (fun k j => bLoad s p m g (f+j.val) h w (cb*32+k.val)) carry
def channelLoop {α β} (s : Shape) (p : ISA α β) (flags : Flags) (m : Memory α)
    (g r f h w : Nat) : Nat → Frag α → Frag α
  | 0,carry => carry
  | c+1,carry => sourceStep s p flags m g r f h w c
      (channelLoop s p flags m g r f h w c carry)
def widthLoop {α β} (s : Shape) (p : ISA α β) (flags : Flags) (m : Memory α)
    (g r f h : Nat) : Nat → Frag α → Frag α
  | 0,carry => carry
  | w+1,carry => channelLoop s p flags m g r f h w (blocks s.cg 32)
      (widthLoop s p flags m g r f h w carry)
def heightLoop {α β} (s : Shape) (p : ISA α β) (flags : Flags) (m : Memory α)
    (g r f : Nat) : Nat → Frag α
  | 0 => fun _ _ => p.zero
  | h+1 => widthLoop s p flags m g r f h s.kw (heightLoop s p flags m g r f h)
def sourceValue {α β} (s : Shape) (p : ISA α β) (flags : Flags) (m : Memory α) (x : Cell) :=
  heightLoop s p flags m x.group (x.spatial/64*64) (x.feature/32*32) s.kh
    ⟨x.spatial%64,Nat.mod_lt _ (by decide)⟩ ⟨x.feature%32,Nat.mod_lt _ (by decide)⟩
def store {α} (m : Memory α) (address : Nat) (value : α) : Memory α :=
  fun r a => if r=.output ∧ a=address then value else m r a
def commit {α} (writes : List (Nat×α)) (m : Memory α) :=
  writes.foldl (fun state w => store state w.1 w.2) m
def programWrites {α β} (s : Shape) (p : ISA α β) (flags : Flags)
    (pid : Program) (m : Memory α) : List (Nat×α) :=
  (lanes.map (cellOf pid)).filterMap fun x =>
    if s.Active x then some (outputPointer s x,sourceValue s p flags m x) else none
def sourceProgram {α β} (s : Shape) (p : ISA α β) (flags : Flags)
    (pid : Program) (m : Memory α) := commit (programWrites s p flags pid m) m
def sourcePrograms {α β} (s : Shape) (p : ISA α β) (flags : Flags) : List Program → Memory α → Memory α
  | [],m => m
  | pid::rest,m => sourcePrograms s p flags rest (sourceProgram s p flags pid m)
def source {α β} (s : Shape) (p : ISA α β) (flags : Flags) (m : Memory α) :=
  sourcePrograms s p flags (programs s) m

end VeriTile.Bench.Repairs.Conv2d

namespace VeriTile.Bench.Repairs.Conv2d.KernelHalf
open VeriTile.Triton FP.Structural
noncomputable section
/-- Readable source quotation for the FP16 constexpr specialization.
The repaired masked loads explicitly supply zero; both TF32 flag values and
all three source loops are preserved. Real and native proofs share this source. -/
def typed (dtype : FloatDType)
    (Input Weight Output : RegionName)
    (batch_dim in_feat_dim in_height in_width out_feat_dim out_height out_width
      input_batch_stride input_in_feat_stride input_height_stride input_width_stride
      weight_out_feat_stride weight_in_feat_stride weight_height_stride weight_width_stride
      output_batch_stride output_out_feat_stride output_height_stride output_width_stride
      kernel_height kernel_width stride_height stride_width padding_height padding_width groups : Nat)
    (_fp16 _tf32 : Bool)
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
          input_block = tl.load(curr_input_pointer, mask=input_mask, other=0.0, dtype=DTYPE)
          weight_block = tl.load(curr_weight_pointer, mask=weight_mask, other=0.0, dtype=DTYPE)
        if _fp16 {
          input_block = (input_block).to(tl.float16)
          weight_block = (weight_block).to(tl.float16)
        }
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


structure Parameters where
  (Input Weight Output : RegionName)
  (batch_dim in_feat_dim in_height in_width out_feat_dim out_height out_width input_batch_stride input_in_feat_stride input_height_stride input_width_stride weight_out_feat_stride weight_in_feat_stride weight_height_stride weight_width_stride output_batch_stride output_out_feat_stride output_height_stride output_width_stride kernel_height kernel_width stride_height stride_width padding_height padding_width groups BLOCK_SIZE_BATCH_HEIGHT_WIDTH BLOCK_SIZE_IN_FEAT BLOCK_SIZE_OUT_FEAT : Nat)
  tf32 : Bool

def source (p : Parameters) : ComputeKernel :=
  typed .fp32 p.Input p.Weight p.Output
    p.batch_dim p.in_feat_dim p.in_height p.in_width p.out_feat_dim p.out_height p.out_width p.input_batch_stride p.input_in_feat_stride p.input_height_stride p.input_width_stride p.weight_out_feat_stride p.weight_in_feat_stride p.weight_height_stride p.weight_width_stride p.output_batch_stride p.output_out_feat_stride p.output_height_stride p.output_width_stride p.kernel_height p.kernel_width p.stride_height p.stride_width p.padding_height p.padding_width p.groups Bool.true p.tf32
    p.BLOCK_SIZE_BATCH_HEIGHT_WIDTH p.BLOCK_SIZE_IN_FEAT p.BLOCK_SIZE_OUT_FEAT


end
end VeriTile.Bench.Repairs.Conv2d.KernelHalf

namespace VeriTile.Bench.Repairs.Conv2d.KernelDefault
open VeriTile.Triton FP.Structural
noncomputable section
/-- The same Python source specialized at fp16=false, with FP32 dot operands.
Masked loads explicitly supply zero; both TF32 values and every source loop
are preserved. This specialization introduces no half conversion. -/
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
          input_block = tl.load(curr_input_pointer, mask=input_mask, other=0.0, dtype=DTYPE)
          weight_block = tl.load(curr_weight_pointer, mask=weight_mask, other=0.0, dtype=DTYPE)
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


abbrev Parameters := KernelHalf.Parameters

def source (p : Parameters) : ComputeKernel :=
  typed .fp32 p.Input p.Weight p.Output
    p.batch_dim p.in_feat_dim p.in_height p.in_width p.out_feat_dim p.out_height p.out_width p.input_batch_stride p.input_in_feat_stride p.input_height_stride p.input_width_stride p.weight_out_feat_stride p.weight_in_feat_stride p.weight_height_stride p.weight_width_stride p.output_batch_stride p.output_out_feat_stride p.output_height_stride p.output_width_stride p.kernel_height p.kernel_width p.stride_height p.stride_width p.padding_height p.padding_width p.groups p.tf32
    p.BLOCK_SIZE_BATCH_HEIGHT_WIDTH p.BLOCK_SIZE_IN_FEAT p.BLOCK_SIZE_OUT_FEAT


end
end VeriTile.Bench.Repairs.Conv2d.KernelDefault

namespace VeriTile.Bench.Repairs.Conv2d.KernelSource
open VeriTile.Triton
abbrev Parameters := KernelHalf.Parameters
/-- One selected readable source for both arithmetic interpretations. The
original constexpr FP16 branch determines the operand type before dot. -/
def kernel (fp16 : Bool) (p : Parameters) : ComputeKernel :=
  if fp16 then KernelHalf.source p else KernelDefault.source p
end VeriTile.Bench.Repairs.Conv2d.KernelSource
