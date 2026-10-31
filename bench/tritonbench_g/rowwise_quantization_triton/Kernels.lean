import Mathlib
import VeriTile.Triton.Float.ScalarOps
import VeriTile.Triton.DSL
import VeriTile.Triton.DSL.CallSignature

/-! Source-order scalar/register and byte-store semantics for the selected
FP32 contiguous rowwise quantizer. `llrint` and direct binary16 conversion are
local primitive ports; neither port supplies a row or kernel result. The max
tree contains exactly the launched lanes. Its native schedule is an ordinary
compiler boundary; no associativity of native FP maxima is assumed. -/
namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton
open VeriTile.Triton
set_option autoImplicit false
set_option maxHeartbeats 2000000
abbrev Byte := Fin 256
abbrev Half := Fin 65536
abbrev Word := VeriTile.Triton.FP.Value .fp32
abbrev Heap := Nat → Byte

def callSignature : CallSignature where
  pointers := [("x_ptr",.fp32),("output_ptr",.i8),("output_maxs",.fp16)]
  scalars := [("n_elements",.integer)]
  constexpr := ["BLOCK_SIZE","P2"]

/-- The unchanged complete original Python body. Storage and invocation types
are outside the quotation; the unused runtime argument emits no instruction. -/
def source (x_ptr output_ptr output_maxs : RegionName)
    (columns width : Nat) : ComputeKernel :=
  let BLOCK_SIZE := columns
  let P2 := width
  with_triton_signature callSignature in triton {
      pid = tl.program_id(axis=0).to(tl.int64)
      block_start = pid * tl.full((), BLOCK_SIZE, tl.int64)
      arange = tl.arange(0, P2)
      offsets = block_start + arange
      row_mask = arange < BLOCK_SIZE
      x = tl.load(x_ptr + offsets, mask=row_mask)

      abs_x = tl.abs(x)
      max_val = tl.max(tl.where(row_mask, abs_x, 0), axis=0)
      safe_max = tl.where(max_val == 0.0, 1.0, max_val)
      output = tl.extra.cuda.libdevice.llrint(127.0 * (x / safe_max))
      tl.store(output_ptr + offsets, output, mask=row_mask)
      tl.store(output_maxs + pid, max_val)
  }

/-- Existing early-maximum-store candidate, with all source operations retained. -/
def earlyMaximumSource (x_ptr output_ptr output_maxs : RegionName)
    (columns width : Nat) : ComputeKernel :=
  let BLOCK_SIZE := columns
  let P2 := width
  with_triton_signature callSignature in triton {
      pid = tl.program_id(axis=0).to(tl.int64)
      block_start = pid * tl.full((), BLOCK_SIZE, tl.int64)
      arange = tl.arange(0, P2)
      offsets = block_start + arange
      row_mask = arange < BLOCK_SIZE
      x = tl.load(x_ptr + offsets, mask=row_mask)

      abs_x = tl.abs(x)
      max_val = tl.max(tl.where(row_mask, abs_x, 0), axis=0)
      tl.store(output_maxs + pid, max_val)
      safe_max = tl.where(max_val == 0.0, 1.0, max_val)
      output = tl.extra.cuda.libdevice.llrint(127.0 * (x / safe_max))
      tl.store(output_ptr + offsets, output, mask=row_mask)
  }

def load32 (heap : Heap) (address : Nat) : Word :=
  VeriTile.Triton.FP.Value.ofBits .fp32
    ((heap address).val + 256 * (heap (address + 1)).val +
      65536 * (heap (address + 2)).val + 16777216 * (heap (address + 3)).val)

def narrow8 (z : Int) : Byte := ⟨(z % 256).toNat % 256, Nat.mod_lt _ (by decide)⟩
def halfByte (word : Half) (byte : Fin 2) : Byte :=
  ⟨word.val / 256 ^ byte.val % 256, Nat.mod_lt _ (by decide)⟩

structure Operations (α : Type) where
  zero : α
  one : α
  scale : α
  abs : α → α
  maximum : α → α → α
  isZero : α → Bool
  divide : α → α → α
  multiply : α → α → α
  llrint : α → Int
  toHalf : α → Half

inductive Tree where
  | leaf (lane : Nat)
  | node (left right : Tree)
  deriving DecidableEq

def Tree.leaves : Tree → List Nat
  | .leaf lane => [lane]
  | .node left right => left.leaves ++ right.leaves

def Tree.evaluate {α : Type} (combine : α → α → α) (input : Nat → α) : Tree → α
  | .leaf lane => input lane
  | .node left right => combine (left.evaluate combine input) (right.evaluate combine input)

structure Launch where
  rows : Nat
  columns : Nat
  width : Nat
  positive : 0 < columns
  covers : columns ≤ width
  tree : Tree
  leafCover : tree.leaves.Perm (List.range width)

structure Layout where
  input : Nat
  output : Nat
  maximum : Nat

def inputAddress (p : Launch) (v : Layout) (row column : Nat) : Nat :=
  v.input + 4 * (row * p.columns + column)
def outputAddress (p : Launch) (v : Layout) (row column : Nat) : Nat :=
  v.output + row * p.columns + column
def maximumAddress (v : Layout) (row : Nat) (byte : Fin 2) : Nat :=
  v.maximum + 2 * row + byte.val

def rowMaximum {α : Type} (ops : Operations α) (p : Launch) (values : Nat → α) : α :=
  p.tree.evaluate ops.maximum (fun lane => if lane < p.columns then ops.abs (values lane) else ops.zero)

def rowScaled {α : Type} (ops : Operations α) (value maximum : α) : α :=
  ops.multiply ops.scale (ops.divide value (if ops.isZero maximum then ops.one else maximum))

def rowQuantized {α : Type} (ops : Operations α) (value maximum : α) : Byte :=
  narrow8 (ops.llrint (rowScaled ops value maximum))

structure Write where
  address : Nat
  value : Byte
  deriving DecidableEq

def write (heap : Heap) (event : Write) : Heap :=
  fun address => if address = event.address then event.value else heap address

def writeAll (events : List Write) (heap : Heap) : Heap := events.foldl write heap

def sourceRowValues {α : Type} (ops : Operations α) (p : Launch) (v : Layout)
    (values : Nat → α) (row : Nat) : List Write :=
  let maximum := rowMaximum ops p values
  (List.range p.width).filterMap (fun column =>
    if column < p.columns then some ⟨outputAddress p v row column,
      rowQuantized ops (values column) maximum⟩ else none) ++
    (List.finRange 2).map (fun byte => ⟨maximumAddress v row byte, halfByte (ops.toHalf maximum) byte⟩)

def sourceRow (ops : Operations Word) (p : Launch) (v : Layout)
    (heap : Heap) (row : Nat) : List Write :=
  sourceRowValues ops p v (fun column => load32 heap (inputAddress p v row column)) row

def runRow (ops : Operations Word) (p : Launch) (v : Layout) (heap : Heap) (row : Nat) : Heap :=
  writeAll (sourceRow ops p v heap row) heap

def runGrid (ops : Operations Word) (p : Launch) (v : Layout)
    (order : List Nat) (heap : Heap) : Heap :=
  order.foldl (runRow ops p v) heap

def InRange (base length address : Nat) : Prop := base ≤ address ∧ address < base + length

def OutputCell (p : Launch) (v : Layout) (address : Nat) : Prop :=
  InRange v.output (p.rows * p.columns) address ∨ InRange v.maximum (2 * p.rows) address

structure Separation (p : Launch) (v : Layout) : Prop where
  input : ∀ address, InRange v.input (4 * (p.rows * p.columns)) address → ¬ OutputCell p v address
  outputs : ∀ address, InRange v.output (p.rows * p.columns) address →
    ¬ InRange v.maximum (2 * p.rows) address

structure Schedule (p : Launch) where
  order : List Nat
  complete : order.Perm (List.range p.rows)

end VeriTile.Bench.Repairs.RowwiseQuantizationTriton
