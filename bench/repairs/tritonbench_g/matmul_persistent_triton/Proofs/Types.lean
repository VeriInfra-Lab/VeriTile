import bench.repairs.tritonbench_g.matmul_persistent_triton.Kernels
import bench.tritonbench_g.matmul_persistent_triton.Proofs.ScheduleCoverage
import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealMath

namespace VeriTile.Bench.Repairs.MatmulPersistentTriton
open VeriTile.Bench.TritonBenchG.MatmulPersistentTriton
abbrev Storage := TypedSource.Storage

def Storage.bytes : Storage → Nat
  | .fp16 => 2
  | .f8e4 => 1

def Storage.blockK : Storage → Nat
  | .fp16 => 64
  | .f8e4 => 128

def Storage.stages : Storage → Nat
  | .fp16 => 3
  | .f8e4 => 4

structure Problem where
  M : Nat
  N : Nat
  K : Nat
  SMS : Nat
  profile : Storage

def Problem.rows (p : Problem) : Nat := RealMath.blocks p.M 128
def Problem.columns (p : Problem) : Nat := RealMath.blocks p.N 256
def Problem.kTiles (p : Problem) : Nat := RealMath.blocks p.K p.profile.blockK
def Problem.tiles (p : Problem) : Nat := p.rows*p.columns
def Problem.group (p : Problem) : Nat := min 8 p.rows
def Problem.grid (p : Problem) : Nat := min p.SMS p.tiles
def Problem.work (p : Problem) (pid : Nat) : Nat := Schedule.tilesFor p.tiles p.SMS pid
def Problem.tile (p : Problem) (pid ordinal : Nat) : Nat := Schedule.tileId p.SMS pid ordinal
def Problem.tileRow (p : Problem) (tile : Nat) : Nat := Schedule.row p.rows p.columns p.group tile
def Problem.tileColumn (p : Problem) (tile : Nat) : Nat := Schedule.column p.rows p.columns p.group tile

/-- Raw/resolved ordinary strided tensor metadata. Storage identity and object
identity are distinct. Native offsets/strides are nonnegative in this domain. -/
structure Tensor where
  object : Nat
  storage : Nat
  offset : Nat
  rows : Nat
  columns : Nat
  rowStride : Nat
  columnStride : Nat
  dtype : Storage
  device : Nat
  isNeg : Bool
  isConj : Bool
  allocationBytes : Nat
  allocationBase : BitVec 64
  pointer : BitVec 64
  deriving DecidableEq

abbrev Address := Nat × Nat
abbrev Memory (α : Type) := Address → α

def Tensor.address (t : Tensor) (row column : Nat) : Address :=
  (t.storage,t.offset+row*t.rowStride+column*t.columnStride)

def Tensor.span (t : Tensor) : Nat :=
  if t.rows=0 ∨ t.columns=0 then t.offset
  else t.offset+(t.rows-1)*t.rowStride+(t.columns-1)*t.columnStride+1

/-- Ordinary representable Tensor ABI and backing-storage bounds. These do not
mention tiles, dot, output values, execution or any task numerical law. -/
structure Tensor.Bounded (t : Tensor) : Prop where
  rows : t.rows<2^63
  columns : t.columns<2^63
  rowStride : t.rowStride<2^63
  columnStride : t.columnStride<2^63
  offset : t.offset<2^63
  span : if t.rows=0 ∨ t.columns=0 then True else t.dtype.bytes*t.span≤t.allocationBytes
  capacity : t.allocationBytes<2^63
  physical : t.allocationBase.toNat+t.allocationBytes≤2^64
  pointer : if t.rows=0 ∨ t.columns=0 then True else
    t.pointer.toNat=t.allocationBase.toNat+t.dtype.bytes*t.offset

structure Buffers where
  A : Tensor
  B : Tensor
  C : Tensor

structure Layout (p : Problem) (b : Buffers) : Prop where
  aShape : b.A.rows=p.M ∧ b.A.columns=p.K
  bShape : b.B.rows=p.K ∧ b.B.columns=p.N
  cShape : b.C.rows=p.M ∧ b.C.columns=p.N
  aType : b.A.dtype=p.profile
  bType : b.B.dtype=p.profile
  cType : b.C.dtype=p.profile
  aBound : b.A.Bounded
  bBound : b.B.Bounded
  cBound : b.C.Bounded
  outputStrides : b.C.rowStride=max p.N 1 ∧ b.C.columnStride=1
  aSeparate : b.C.storage≠b.A.storage
  bSeparate : b.C.storage≠b.B.storage

def clamp (length coordinate : Nat) : Nat := if coordinate<length then coordinate else 0
def Problem.row (p : Problem) (tile lane : Nat) : Nat := p.tileRow tile*128+lane
def Problem.column (p : Problem) (tile lane : Nat) : Nat := p.tileColumn tile*256+lane
def Problem.k (p : Problem) (iteration lane : Nat) : Nat := iteration*p.profile.blockK+lane

end VeriTile.Bench.Repairs.MatmulPersistentTriton
