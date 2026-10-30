import bench.tritonbench_g.bgmv_expand_slice.Proofs.Geometry

namespace VeriTile.Bench.Repairs.BgmvExpandSlice.Storage
open VeriTile.Triton

def signedLimit : Nat := 9223372036854775808
structure Buffer where
  base : Nat
  extent : Nat
  bytes : Nat
  bytesPositive : 0<bytes
  signedSize : extent*bytes<signedLimit
  physicalWindow : base+extent*bytes≤18446744073709551616

def Buffer.address (b : Buffer) (off : Nat) : Nat := b.base+off*b.bytes

def Buffer.Contains (b : Buffer) (addr : Nat) : Prop :=
  b.base≤addr ∧ addr<b.base+b.extent*b.bytes

def Disjoint (a b : Buffer) : Prop :=
  ∀ addr, a.Contains addr → ¬b.Contains addr

/-- These are the actual fresh x, weights, result and compact-index allocations
from the same wrapper call. Width equalities specialize the production route;
they are ordinary tensor storage facts, not numerical or execution premises. -/
structure Placement (h : Call) where
  input : Buffer
  weights : Buffer
  output : Buffer
  indices : Buffer
  inputSize : input.extent=h.batches*h.BK
  weightSize : weights.extent=h.adapters*((h.N+255)*h.BK)
  outputSize : output.extent=h.batches*(h.columns+255)
  indexSize : indices.extent=h.batches
  inputWidth : input.bytes=4
  weightWidth : weights.bytes=2
  outputWidth : output.bytes=4
  indexWidth : indices.bytes=4
  separate : List.Pairwise Disjoint [input,weights,output,indices]

def buffer (h : Call) (p : Placement h) (region : RegionName) : Buffer :=
  if region="X" then p.input else if region="W" then p.weights
  else if region="Indices" then p.indices else p.output

abbrev Heap (β : Type) := Nat → β

def lift (h : Call) (p : Placement h) {β : Type} (heap : Heap β) (region : RegionName) (off : Nat) : β :=
  heap ((buffer h p region).address off)

theorem address_injective (b : Buffer) (i j : Nat) (same : b.address i=b.address j) : i=j := by
  unfold Buffer.address at same
  exact Nat.eq_of_mul_eq_mul_right b.bytesPositive (Nat.add_left_cancel same)

theorem address_inside (b : Buffer) (off : Nat) (inside : off<b.extent) : b.Contains (b.address off) := by
  have bound := Nat.mul_lt_mul_of_pos_right inside b.bytesPositive
  unfold Buffer.Contains Buffer.address
  omega

theorem byte_span (b : Buffer) (off : Nat) (inside : off<b.extent) :
    b.address off+b.bytes≤18446744073709551616 := by
  have bound := Nat.mul_le_mul_right b.bytes (Nat.succ_le_of_lt inside)
  have cap := b.physicalWindow
  unfold Buffer.address
  rw [Nat.succ_mul] at bound
  omega

theorem all_source_offsets (h : Call) (p : Placement h) (program : Geometry.Program h)
    (slot : Nat) (selected : slot<h.adapters) (n : Fin 256) (k : Fin h.BK) :
    Geometry.inputOffset h program.batch k.val<p.input.extent ∧
    Geometry.weightOffset h slot program.row n.val k.val<p.weights.extent ∧
    Geometry.outputOffset h program.batch program.row n.val<p.output.extent ∧
    program.batch<p.indices.extent := by
  rw [p.inputSize,p.weightSize,p.outputSize,p.indexSize]
  exact ⟨Geometry.input_inside h program k,Geometry.weight_inside h program slot selected n k,
    Geometry.output_inside h program n,program.batchInside⟩

#axiomsClean all_source_offsets
#axiomsClean byte_span
end VeriTile.Bench.Repairs.BgmvExpandSlice.Storage
