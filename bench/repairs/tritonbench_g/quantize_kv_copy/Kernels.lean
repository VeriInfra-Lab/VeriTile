import VeriTile.Triton.DSL
import VeriTile.Triton.DSL.CallSignature
import bench.repairs.tritonbench_g.rowwise_quantization_triton.Kernels

namespace VeriTile.Bench.Repairs.QuantizeKvCopy
set_option autoImplicit false
set_option maxHeartbeats 2000000
namespace RW
export VeriTile.Bench.Repairs.RowwiseQuantizationTriton (Byte Heap Tree Write writeAll InRange)
end RW
abbrev Byte := RW.Byte
abbrev Heap := RW.Heap
abbrev Tree := RW.Tree
abbrev Write := RW.Write
abbrev writeAll := RW.writeAll

/-- Native logical geometry; each head is split into fixed groups of eight. -/
structure Request where
  tokens : Nat
  sourceRows : Nat
  headCount : Nat
  groups : Nat
  capacity : Nat
  destStride : Nat
  slot : Nat → Nat

def groupWidth : Nat := 8
def Request.programs (p : Request) : Nat := p.tokens * p.headCount
def Request.sourcePrograms (p : Request) : Nat := p.sourceRows * p.headCount
def Request.outputPrograms (p : Request) : Nat := p.capacity * p.headCount
def Request.destination (p : Request) (program : Nat) : Nat :=
  p.slot (program / p.headCount) * p.headCount + program % p.headCount

def groupBlock (p : Request) : Nat := 2 ^ Nat.clog 2 p.groups

structure Domain (p : Request) : Prop where
  heads_positive : 0 < p.headCount
  groups_positive : 0 < p.groups
  sources : p.tokens ≤ p.sourceRows
  destination_bound : ∀ s, s < p.tokens → p.slot s < p.capacity
  destination_int32 : ∀ s, s < p.tokens → p.slot s < 2^31
  destination_unique : ∀ s t, s < p.tokens → t < p.tokens → p.slot s = p.slot t → s = t

structure Plan (p : Request) where
  tree : Tree
  covers : tree.leaves.Perm (List.range groupWidth)

/-- Native storage allocation and byte offset of the Tensor data pointer. -/
structure Tensor where
  object : Nat
  base : Nat
  offset : Nat
  capacity : Nat

def Tensor.data (t : Tensor) : Nat := t.base + t.offset
def Tensor.Contains (t : Tensor) (a : Nat) : Prop := RW.InRange t.base t.capacity a

inductive Role where
  | input | destination | output | scale
  deriving DecidableEq

structure Layout where
  tensor : Role → Tensor

def elements (p : Request) : Role → Nat
  | .input => 4 * (p.sourcePrograms * p.groups * groupWidth)
  | .destination => 4 * (if p.tokens = 0 then 0 else (p.tokens - 1) * p.destStride + 1)
  | .output => p.outputPrograms * p.groups * groupWidth
  | .scale => 4 * (p.outputPrograms * p.groups)

structure NativeAllocations (p : Request) (v : Layout) : Prop where
  capacity : ∀ r, (v.tensor r).offset + elements p r ≤ (v.tensor r).capacity
  separate : ∀ r s, r ≠ s → ∀ a, (v.tensor r).Contains a → ¬ (v.tensor s).Contains a

def inputAddress (p : Request) (v : Layout) (s h c : Nat) : Nat :=
  (v.tensor .input).data + 4 * ((s * p.groups + h) * groupWidth + c)
def destAddress (p : Request) (v : Layout) (s : Nat) : Nat :=
  (v.tensor .destination).data + 4 * ((s / p.headCount) * p.destStride)
def outputAddress (p : Request) (v : Layout) (slot h c : Nat) : Nat :=
  (v.tensor .output).data + ((slot * p.groups + h) * groupWidth + c)
def scaleAddress (p : Request) (v : Layout) (slot h : Nat) (b : Fin 4) : Nat :=
  (v.tensor .scale).data + (4 * (slot * p.groups + h) + b.val)

def loadDest (heap : Heap) (address : Nat) : Nat :=
  ((List.range 4).map (fun b => (heap (address + b)).val * 256 ^ b)).sum

/-- Decode the INT32 token destination and retain program_id(1). -/
def loadTarget (p : Request) (heap : Heap) (address program : Nat) : Nat :=
  loadDest heap address * p.headCount + program % p.headCount

def loadInput {α : Type} (decode : (Fin 4 → Byte) → α) (p : Request) (v : Layout)
    (heap : Heap) (s h c : Nat) : α :=
  decode (fun b => heap (inputAddress p v s h c + b.val))

def Metadata (p : Request) (v : Layout) (heap : Heap) : Prop :=
  ∀ s, s < p.programs → loadTarget p heap (destAddress p v s) s = p.destination s

structure Operations (α η : Type) where
  one : α
  levels : α
  abs : α → α
  maximum : α → α → α
  divide : α → α → α
  isZero : α → Bool
  toScale : α → η
  fromScale : η → α
  int8 : α → Byte

def rowMaximum {α η : Type} (ops : Operations α η) (plan : Tree) (x : Nat → α) : α :=
  plan.evaluate ops.maximum (fun c => ops.abs (x c))
def rowScale {α η : Type} (ops : Operations α η) (maximum : α) : η :=
  ops.toScale (ops.divide maximum ops.levels)
def denominator {α η : Type} (ops : Operations α η) (maximum : α) : α :=
  if ops.isZero maximum then ops.one else ops.fromScale (rowScale ops maximum)
def rowCode {α η : Type} (ops : Operations α η) (x maximum : α) : Byte :=
  ops.int8 (ops.divide x (denominator ops maximum))

/-- Actual SSA assignment order: abs, max, FP32 divide, same-dtype scale cast,
select, divide, signed-int8 conversion. The stored scale precedes selection. -/
def sourceRegisters {α η : Type} (ops : Operations α η) (tree : Tree) (x : Nat → α) : (Nat → Byte) × η :=
  let abs_data := fun c => ops.abs (x c)
  let maximum := tree.evaluate ops.maximum abs_data
  let data_scale := ops.toScale (ops.divide maximum ops.levels)
  let divisor := if ops.isZero maximum then ops.one else ops.fromScale data_scale
  let q_src_data := fun c => ops.int8 (ops.divide (x c) divisor)
  (q_src_data, data_scale)

def codes {p : Request} {α η : Type} (ops : Operations α η) (plan : Plan p)
    (decode : (Fin 4 → Byte) → α) (v : Layout) (heap : Heap) (s h c : Nat) : Byte :=
  (sourceRegisters ops plan.tree (loadInput decode p v heap s h)).1 c
def scales {p : Request} {α η : Type} (ops : Operations α η) (plan : Plan p)
    (decode : (Fin 4 → Byte) → α) (encode : η → Fin 4 → Byte)
    (v : Layout) (heap : Heap) (s h : Nat) (b : Fin 4) : Byte :=
  encode (sourceRegisters ops plan.tree (loadInput decode p v heap s h)).2 b

def programEvents {α η : Type} (ops : Operations α η) (p : Request) (plan : Plan p)
    (decode : (Fin 4 → Byte) → α) (encode : η → Fin 4 → Byte)
    (v : Layout) (heap : Heap) (s : Nat) : List Write :=
  let dest := loadTarget p heap (destAddress p v s) s
  ((List.range (groupBlock p)).flatMap (fun h => if h < p.groups then
    (List.range groupWidth).map (fun c => ⟨outputAddress p v dest h c, codes ops plan decode v heap s h c⟩) else [])) ++
  ((List.range (groupBlock p)).flatMap (fun h => if h < p.groups then
    (List.finRange 4).map (fun b => ⟨scaleAddress p v dest h b, scales ops plan decode encode v heap s h b⟩) else []))

def runProgram {α η : Type} (ops : Operations α η) (p : Request) (plan : Plan p)
    (decode : (Fin 4 → Byte) → α) (encode : η → Fin 4 → Byte) (v : Layout) (heap : Heap) (s : Nat) : Heap :=
  writeAll (programEvents ops p plan decode encode v heap s) heap
def runGrid {α η : Type} (ops : Operations α η) (p : Request) (plan : Plan p)
    (decode : (Fin 4 → Byte) → α) (encode : η → Fin 4 → Byte) (v : Layout) (order : List Nat) (heap : Heap) : Heap :=
  order.foldl (runProgram ops p plan decode encode v) heap

structure Schedule (p : Request) where
  order : List Nat
  complete : order.Perm (List.range p.programs)

def Touched (p : Request) (v : Layout) (a : Nat) : Prop :=
  (∃ s h c, s < p.programs ∧ h < p.groups ∧ c < groupWidth ∧ a = outputAddress p v (p.destination s) h c) ∨
  (∃ s h, ∃ b : Fin 4, s < p.programs ∧ h < p.groups ∧ a = scaleAddress p v (p.destination s) h b)

/-- The Python return value is None. Both output objects are caller arguments. -/
def call {α η : Type} (ops : Operations α η) (p : Request) (plan : Plan p)
    (decode : (Fin 4 → Byte) → α) (encode : η → Fin 4 → Byte) (v : Layout) (schedule : Schedule p) (heap : Heap) : Heap × Option Unit :=
  (runGrid ops p plan decode encode v schedule.order heap, none)

/-- Native token metadata is four little-endian bytes of a nonnegative INT32. -/
def RawMetadata (p : Request) (v : Layout) (heap : Heap) : Prop :=
  ∀ token, token < p.tokens →
    loadDest heap ((v.tensor .destination).data + 4*(token*p.destStride)) = p.slot token

def inputGroup {α : Type} (decode : (Fin 4 → Byte) → α) (p : Request) (v : Layout)
    (heap : Heap) (token head group : Nat) : Nat → α :=
  loadInput decode p v heap (token*p.headCount+head) group
def logicalOutputAddress (p : Request) (v : Layout) (slot head group lane : Nat) : Nat :=
  outputAddress p v (slot*p.headCount+head) group lane
def logicalScaleAddress (p : Request) (v : Layout) (slot head group : Nat) (b : Fin 4) : Nat :=
  scaleAddress p v (slot*p.headCount+head) group b

def LogicalTouched (p : Request) (v : Layout) (a : Nat) : Prop :=
  (∃ token head group lane, token < p.tokens ∧ head < p.headCount ∧ group < p.groups ∧
    lane < groupWidth ∧ a = logicalOutputAddress p v (p.slot token) head group lane) ∨
  (∃ token head group, ∃ b : Fin 4, token < p.tokens ∧ head < p.headCount ∧ group < p.groups ∧
    a = logicalScaleAddress p v (p.slot token) head group b)

namespace Native
def signed (x : Nat) : BitVec 64 := BitVec.ofNat 64 x

def inputIndex (p : Request) (s h c : Nat) : BitVec 64 :=
  signed (s/p.headCount)*signed (p.headCount*p.groups*groupWidth)+
    signed (s%p.headCount)*signed (p.groups*groupWidth)+signed h*signed groupWidth+signed c

def outputIndex (p : Request) (slot h c : Nat) : BitVec 64 :=
  signed slot*signed (p.groups*groupWidth)+signed h*signed groupWidth+signed c

def scaleIndex (p : Request) (slot h : Nat) : BitVec 64 := signed slot*signed p.groups+signed h

def groupMask (p : Request) (a : Nat) : Bool := (signed a).slt (signed p.groups)
def pointer (base : Nat) (index : BitVec 64) (bytes : Nat) : BitVec 64 :=
  signed base+index*signed bytes


def load {α : Type} (decode : (Fin 4 → Byte) → α) (p : Request) (v : Layout) (heap : Heap) (s h c : Nat) : α :=
  decode (fun b => heap ((pointer (v.tensor .input).data (inputIndex p s h c) 4).toNat+b.val))

def destination (p : Request) (v : Layout) (heap : Heap) (s : Nat) : Nat :=
  (BitVec.ofNat 32 (loadDest heap (pointer (v.tensor .destination).data
    (signed ((s/p.headCount)*p.destStride)) 4).toNat)).toInt.toNat*p.headCount+s%p.headCount

def programEvents {α η : Type} (ops : Operations α η) (p : Request) (plan : Plan p)
    (decode : (Fin 4 → Byte) → α) (encode : η → Fin 4 → Byte)
    (v : Layout) (heap : Heap) (s : Nat) : List Write :=
  let dest := destination p v heap s
  ((List.range (groupBlock p)).flatMap (fun h => if groupMask p h then
    (List.range groupWidth).map (fun c =>
      ⟨(pointer (v.tensor .output).data (outputIndex p dest h c) 1).toNat,
       (sourceRegisters ops plan.tree (load decode p v heap s h)).1 c⟩) else [])) ++
  ((List.range (groupBlock p)).flatMap (fun h => if groupMask p h then
    (List.finRange 4).map (fun b =>
      ⟨(pointer (v.tensor .scale).data (scaleIndex p dest h) 4).toNat+b.val,
       encode (sourceRegisters ops plan.tree (load decode p v heap s h)).2 b⟩) else []))

def runProgram {α η : Type} (ops : Operations α η) (p : Request) (plan : Plan p)
    (decode : (Fin 4 → Byte) → α) (encode : η → Fin 4 → Byte)
    (v : Layout) (heap : Heap) (s : Nat) : Heap :=
  writeAll (programEvents ops p plan decode encode v heap s) heap

def runGrid {α η : Type} (ops : Operations α η) (p : Request) (plan : Plan p)
    (decode : (Fin 4 → Byte) → α) (encode : η → Fin 4 → Byte)
    (v : Layout) (order : List Nat) (heap : Heap) : Heap :=
  order.foldl (runProgram ops p plan decode encode v) heap
end Native

end VeriTile.Bench.Repairs.QuantizeKvCopy

namespace VeriTile.Bench.Repairs.QuantizeKvCopy.KernelSyntax
open VeriTile.Triton

/-- Complete canonical JIT. The integer widths remain visible in the quote;
fixed-width address interpretation is proved separately from scalar arithmetic. -/
def callSignature : CallSignature where
  pointers := [("K",.fp32),("Dest_loc",.i32),("Out",.i8),("Out_scale",.fp32)]
  scalars := [("stride_k_bs",.integer),("stride_k_h",.integer),("stride_k_g",.integer),("stride_k_d",.integer),("stride_o_bs",.integer),("stride_o_h",.integer),("stride_o_g",.integer),("stride_o_d",.integer),("stride_os_bs",.integer),("stride_os_h",.integer),("stride_os_g",.integer),("stride_dest",.integer),("group_size",.integer)]
  constexpr := ["BLOCK_GROUP_NUM","BLOCK_GROUP_DIM"]

def source (groups block heads : Nat) : ComputeKernel :=
  let K := ("K" : RegionName)
  let Dest_loc := ("D" : RegionName)
  let Out := ("O" : RegionName)
  let Out_scale := ("S" : RegionName)
  let BLOCK_GROUP_NUM := block
  let BLOCK_GROUP_DIM := (8 : Nat)
  let rowStride := heads * groups * 8
  let headStride := groups * 8
  let scaleStride := heads * groups
  with_triton_call callSignature call_args [
    stride_k_bs := rowStride, stride_k_h := headStride, stride_k_g := 8, stride_k_d := 1, stride_o_bs := rowStride, stride_o_h := headStride, stride_o_g := 8, stride_o_d := 1, stride_os_bs := scaleStride, stride_os_h := groups, stride_os_g := 1, stride_dest := 1, group_size := groups] in triton {
    cur_index = tl.program_id(0).to(tl.int64)
    cur_head = tl.program_id(1).to(tl.int64)
    offs_g = tl.arange(0, BLOCK_GROUP_NUM).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_GROUP_DIM).to(tl.int64)
    dest_index = tl.load(Dest_loc + cur_index * stride_dest).to(tl.int64)
    src_data = tl.load(K + cur_index * stride_k_bs + cur_head * stride_k_h + offs_g[:, None] * stride_k_g + offs_d[None, :] * stride_k_d, mask=offs_g[:, None] < group_size, other=0.0)
    abs_data = tl.abs(src_data)
    maximum = tl.max(abs_data, axis=1)
    data_scale = (maximum / 127.0).to(Out_scale.dtype.element_ty)
    denominator = tl.where(maximum == 0.0, 1.0, data_scale)
    q_src_data = (src_data / denominator[:, None]).to(tl.int8)
    o_ptrs = Out + dest_index * stride_o_bs + cur_head * stride_o_h + offs_g[:, None] * stride_o_g + offs_d[None, :] * stride_o_d
    os_ptrs = Out_scale + dest_index * stride_os_bs + cur_head * stride_os_h + offs_g * stride_os_g
    tl.store(o_ptrs, q_src_data, mask=offs_g[:, None] < group_size)
    tl.store(os_ptrs, data_scale, mask=offs_g < group_size)
  }

def addressEarly (groups block heads : Nat) : ComputeKernel :=
  let K := ("K" : RegionName)
  let Dest_loc := ("D" : RegionName)
  let Out := ("O" : RegionName)
  let Out_scale := ("S" : RegionName)
  let BLOCK_GROUP_NUM := block
  let BLOCK_GROUP_DIM := (8 : Nat)
  let rowStride := heads * groups * 8
  let headStride := groups * 8
  let scaleStride := heads * groups
  with_triton_call callSignature call_args [
    stride_k_bs := rowStride, stride_k_h := headStride, stride_k_g := 8, stride_k_d := 1, stride_o_bs := rowStride, stride_o_h := headStride, stride_o_g := 8, stride_o_d := 1, stride_os_bs := scaleStride, stride_os_h := groups, stride_os_g := 1, stride_dest := 1, group_size := groups] in triton {
    cur_index = tl.program_id(0).to(tl.int64)
    cur_head = tl.program_id(1).to(tl.int64)
    offs_g = tl.arange(0, BLOCK_GROUP_NUM).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_GROUP_DIM).to(tl.int64)
    dest_index = tl.load(Dest_loc + cur_index * stride_dest).to(tl.int64)
    src_data = tl.load(K + cur_index * stride_k_bs + cur_head * stride_k_h + offs_g[:, None] * stride_k_g + offs_d[None, :] * stride_k_d, mask=offs_g[:, None] < group_size, other=0.0)
    o_ptrs = Out + dest_index * stride_o_bs + cur_head * stride_o_h + offs_g[:, None] * stride_o_g + offs_d[None, :] * stride_o_d
    os_ptrs = Out_scale + dest_index * stride_os_bs + cur_head * stride_os_h + offs_g * stride_os_g
    abs_data = tl.abs(src_data)
    maximum = tl.max(abs_data, axis=1)
    data_scale = (maximum / 127.0).to(Out_scale.dtype.element_ty)
    denominator = tl.where(maximum == 0.0, 1.0, data_scale)
    q_src_data = (src_data / denominator[:, None]).to(tl.int8)
    tl.store(o_ptrs, q_src_data, mask=offs_g[:, None] < group_size)
    tl.store(os_ptrs, data_scale, mask=offs_g < group_size)
  }


end VeriTile.Bench.Repairs.QuantizeKvCopy.KernelSyntax
