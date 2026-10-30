import bench.repairs.tritonbench_g.quantize_copy_kv.Kernels
import bench.repairs.tritonbench_g.rowwise_quantization_triton.Kernels

namespace VeriTile.Bench.Repairs.QuantizeCopyKv
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

structure Request where
  tokens : Nat
  sourceRows : Nat
  heads : Nat
  columns : Nat
  capacity : Nat
  destStride : Nat
  destination : Nat → Nat

def headBlock (p : Request) : Nat := 2 ^ Nat.clog 2 p.heads

structure Domain (p : Request) : Prop where
  heads_positive : 0 < p.heads
  dimension : ∃ exponent, p.columns = 2 ^ exponent
  sources : p.tokens ≤ p.sourceRows
  destination_bound : ∀ s, s < p.tokens → p.destination s < p.capacity
  destination_int64 : ∀ s, s < p.tokens → p.destination s < 2^63
  destination_unique : ∀ s t, s < p.tokens → t < p.tokens → p.destination s = p.destination t → s = t

structure Plan (p : Request) where
  tree : Tree
  covers : tree.leaves.Perm (List.range p.columns)

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
  | .input => 4 * (p.sourceRows * p.heads * p.columns)
  | .destination => 8 * (if p.tokens = 0 then 0 else (p.tokens - 1) * p.destStride + 1)
  | .output => p.capacity * p.heads * p.columns
  | .scale => 2 * (p.capacity * p.heads)

structure NativeAllocations (p : Request) (v : Layout) : Prop where
  capacity : ∀ r, (v.tensor r).offset + elements p r ≤ (v.tensor r).capacity
  separate : ∀ r s, r ≠ s → ∀ a, (v.tensor r).Contains a → ¬ (v.tensor s).Contains a

def inputAddress (p : Request) (v : Layout) (s h c : Nat) : Nat :=
  (v.tensor .input).data + 4 * ((s * p.heads + h) * p.columns + c)
def destAddress (p : Request) (v : Layout) (s : Nat) : Nat :=
  (v.tensor .destination).data + 8 * (s * p.destStride)
def outputAddress (p : Request) (v : Layout) (slot h c : Nat) : Nat :=
  (v.tensor .output).data + ((slot * p.heads + h) * p.columns + c)
def scaleAddress (p : Request) (v : Layout) (slot h : Nat) (b : Fin 2) : Nat :=
  (v.tensor .scale).data + (2 * (slot * p.heads + h) + b.val)

def loadDest (heap : Heap) (address : Nat) : Nat :=
  ((List.range 8).map (fun b => (heap (address + b)).val * 256 ^ b)).sum

def loadInput {α : Type} (decode : (Fin 4 → Byte) → α) (p : Request) (v : Layout)
    (heap : Heap) (s h c : Nat) : α :=
  decode (fun b => heap (inputAddress p v s h c + b.val))

def Metadata (p : Request) (v : Layout) (heap : Heap) : Prop :=
  ∀ s, s < p.tokens → loadDest heap (destAddress p v s) = p.destination s

structure Operations (α η : Type) where
  one : α
  levels : α
  abs : α → α
  maximum : α → α → α
  divide : α → α → α
  isZero : α → Bool
  toHalf : α → η
  fromHalf : η → α
  int8 : α → Byte

def rowMaximum {α η : Type} (ops : Operations α η) (plan : Tree) (x : Nat → α) : α :=
  plan.evaluate ops.maximum (fun c => ops.abs (x c))
def rowScale {α η : Type} (ops : Operations α η) (maximum : α) : η :=
  ops.toHalf (ops.divide maximum ops.levels)
def denominator {α η : Type} (ops : Operations α η) (maximum : α) : α :=
  if ops.isZero maximum then ops.one else ops.fromHalf (rowScale ops maximum)
def rowCode {α η : Type} (ops : Operations α η) (x maximum : α) : Byte :=
  ops.int8 (ops.divide x (denominator ops maximum))

/-- Actual SSA assignment order: abs, max, divide, cast-half, select, divide,
signed-int8 conversion. The scale stored is the rounded half before selection. -/
def sourceRegisters {α η : Type} (ops : Operations α η) (tree : Tree) (x : Nat → α) : (Nat → Byte) × η :=
  let abs_data := fun c => ops.abs (x c)
  let maximum := tree.evaluate ops.maximum abs_data
  let data_scale := ops.toHalf (ops.divide maximum ops.levels)
  let divisor := if ops.isZero maximum then ops.one else ops.fromHalf data_scale
  let q_src_data := fun c => ops.int8 (ops.divide (x c) divisor)
  (q_src_data, data_scale)

def codes {p : Request} {α η : Type} (ops : Operations α η) (plan : Plan p)
    (decode : (Fin 4 → Byte) → α) (v : Layout) (heap : Heap) (s h c : Nat) : Byte :=
  (sourceRegisters ops plan.tree (loadInput decode p v heap s h)).1 c
def scales {p : Request} {α η : Type} (ops : Operations α η) (plan : Plan p)
    (decode : (Fin 4 → Byte) → α) (encode : η → Fin 2 → Byte)
    (v : Layout) (heap : Heap) (s h : Nat) (b : Fin 2) : Byte :=
  encode (sourceRegisters ops plan.tree (loadInput decode p v heap s h)).2 b

def programEvents {α η : Type} (ops : Operations α η) (p : Request) (plan : Plan p)
    (decode : (Fin 4 → Byte) → α) (encode : η → Fin 2 → Byte)
    (v : Layout) (heap : Heap) (s : Nat) : List Write :=
  let dest := loadDest heap (destAddress p v s)
  ((List.range (headBlock p)).flatMap (fun h => if h < p.heads then
    (List.range p.columns).map (fun c => ⟨outputAddress p v dest h c, codes ops plan decode v heap s h c⟩) else [])) ++
  ((List.range (headBlock p)).flatMap (fun h => if h < p.heads then
    (List.finRange 2).map (fun b => ⟨scaleAddress p v dest h b, scales ops plan decode encode v heap s h b⟩) else []))

def runProgram {α η : Type} (ops : Operations α η) (p : Request) (plan : Plan p)
    (decode : (Fin 4 → Byte) → α) (encode : η → Fin 2 → Byte) (v : Layout) (heap : Heap) (s : Nat) : Heap :=
  writeAll (programEvents ops p plan decode encode v heap s) heap
def runGrid {α η : Type} (ops : Operations α η) (p : Request) (plan : Plan p)
    (decode : (Fin 4 → Byte) → α) (encode : η → Fin 2 → Byte) (v : Layout) (order : List Nat) (heap : Heap) : Heap :=
  order.foldl (runProgram ops p plan decode encode v) heap

structure Schedule (p : Request) where
  order : List Nat
  complete : order.Perm (List.range p.tokens)

def Touched (p : Request) (v : Layout) (a : Nat) : Prop :=
  (∃ s h c, s < p.tokens ∧ h < p.heads ∧ c < p.columns ∧ a = outputAddress p v (p.destination s) h c) ∨
  (∃ s h, ∃ b : Fin 2, s < p.tokens ∧ h < p.heads ∧ a = scaleAddress p v (p.destination s) h b)

/-- The Python return value is None. Both output objects are caller arguments. -/
def call {α η : Type} (ops : Operations α η) (p : Request) (plan : Plan p)
    (decode : (Fin 4 → Byte) → α) (encode : η → Fin 2 → Byte) (v : Layout) (schedule : Schedule p) (heap : Heap) : Heap × Option Unit :=
  (runGrid ops p plan decode encode v schedule.order heap, none)

namespace Native
def signed (x : Nat) : BitVec 64 := BitVec.ofNat 64 x

def inputIndex (p : Request) (s h c : Nat) : BitVec 64 :=
  signed s*signed (p.heads*p.columns)+signed h*signed p.columns+signed c

def outputIndex (p : Request) (slot h c : Nat) : BitVec 64 :=
  signed slot*signed (p.heads*p.columns)+signed h*signed p.columns+signed c

def scaleIndex (p : Request) (slot h : Nat) : BitVec 64 := signed slot*signed p.heads+signed h

def headMask (p : Request) (a : Nat) : Bool := (signed a).slt (signed p.heads)
def pointer (base : Nat) (index : BitVec 64) (bytes : Nat) : BitVec 64 :=
  signed base+index*signed bytes


def load {α : Type} (decode : (Fin 4 → Byte) → α) (p : Request) (v : Layout) (heap : Heap) (s h c : Nat) : α :=
  decode (fun b => heap ((pointer (v.tensor .input).data (inputIndex p s h c) 4).toNat+b.val))

def destination (p : Request) (v : Layout) (heap : Heap) (s : Nat) : Nat :=
  (signed (loadDest heap (pointer (v.tensor .destination).data (signed (s*p.destStride)) 8).toNat)).toInt.toNat

def programEvents {α η : Type} (ops : Operations α η) (p : Request) (plan : Plan p)
    (decode : (Fin 4 → Byte) → α) (encode : η → Fin 2 → Byte)
    (v : Layout) (heap : Heap) (s : Nat) : List Write :=
  let dest := destination p v heap s
  ((List.range (headBlock p)).flatMap (fun h => if headMask p h then
    (List.range p.columns).map (fun c =>
      ⟨(pointer (v.tensor .output).data (outputIndex p dest h c) 1).toNat,
       (sourceRegisters ops plan.tree (load decode p v heap s h)).1 c⟩) else [])) ++
  ((List.range (headBlock p)).flatMap (fun h => if headMask p h then
    (List.finRange 2).map (fun b =>
      ⟨(pointer (v.tensor .scale).data (scaleIndex p dest h) 2).toNat+b.val,
       encode (sourceRegisters ops plan.tree (load decode p v heap s h)).2 b⟩) else []))

def runProgram {α η : Type} (ops : Operations α η) (p : Request) (plan : Plan p)
    (decode : (Fin 4 → Byte) → α) (encode : η → Fin 2 → Byte)
    (v : Layout) (heap : Heap) (s : Nat) : Heap :=
  writeAll (programEvents ops p plan decode encode v heap s) heap

def runGrid {α η : Type} (ops : Operations α η) (p : Request) (plan : Plan p)
    (decode : (Fin 4 → Byte) → α) (encode : η → Fin 2 → Byte)
    (v : Layout) (order : List Nat) (heap : Heap) : Heap :=
  order.foldl (runProgram ops p plan decode encode v) heap
end Native

end VeriTile.Bench.Repairs.QuantizeCopyKv
