import bench.tritonbench_g.triton_matmul.Proofs.SourceDecomposition
import bench.tritonbench_g.triton_matmul.Proofs.Mathematical_SourceScope
import bench.tritonbench_g.matmul_triton_autotune.Sources.Mathematical_Host
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair
open VeriTile.Triton FP.Structural FPValue
open ParallelAttention.FPStages (foldRange)
noncomputable section
structure Request where
  (A B C : RegionName)
  (storage storageB : Storage)
  (rows columns inner innerB strideAM strideAK strideBK strideBN : Nat)

def Request.parameters (r : Request) := SourceScope.parameters r.A r.B r.C r.storage
  r.rows r.columns r.inner r.strideAM r.strideAK r.strideBK r.strideBN
def kernel (r : Request) := Source.typed r.parameters r.storage
def tileRows (r : Request) := (r.rows+127)/128
def tileColumns (r : Request) := (r.columns+255)/256
def programs (r : Request) := tileRows r*tileColumns r
def instanceState {α : Type} (r : Request) (pid : Nat) (s : State α) : State α :=
  {s with pids := (fun ax => if ax=0 then pid else 0)
          numPids := (fun ax => if ax=0 then programs r else 1)}
def runPrograms {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (r : Request) : List Nat → State α → Option (State α)
  | [], s => some s
  | pid::rest, s => (FP.Structural.exec (backend M D) (kernel r) (instanceState r pid s)).bind
      (runPrograms M D r rest)
inductive HostError where | incompatibleDimensions | incompatibleDtypes | executionFailure
  deriving DecidableEq
def host {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (r : Request) (s : State α) : Except HostError (RegionName × State α) :=
  if r.inner≠r.innerB then .error .incompatibleDimensions
  else if r.storage≠r.storageB then .error .incompatibleDtypes
  else match runPrograms M D r (List.range (programs r)) s with
    | none => .error .executionFailure
    | some t => .ok (r.C,t)

structure Numerics (α : Type) where
  zero : α
  loadZero : α
  dotAcc : (k : Nat) → (Fin k → α) → (Fin k → α) → α → α
  castOutput : α → α
structure PrimitiveLaws {α : Type} (M : Algebra α) (D : DotAcc α)
    (Q : Numerics α) (st : Storage) : Prop where
  initialZero : M.literal (some .fp32) .real 0=Q.zero
  loadZero : M.cast none .real (dtype st) (M.literal none .real 0.0)=Q.loadZero
  dotAcc : ∀ (m k n : Nat) (a : TileIndex [m,k] → α) (b : TileIndex [k,n] → α)
      (acc : TileIndex [m,n] → α) (i : TileIndex [m,n]),
    D.apply (some .fp32) (dtype st) (dtype st) .real {} (batch := []) a b acc i=
      Q.dotAcc k (fun j => a (i.1,j,PUnit.unit)) (fun j => b (j,i.2.1,PUnit.unit)) (acc i)
  outputCast : ∀ x, M.cast none (dtype st) (dtype st) (M.cast none .real (dtype st) x)=Q.castOutput x

def inputA {α : Type} [Inhabited α] (r : Request) (s : State α) (row k : Nat) :=
  FPValue.read r.storage (s.mem r.A (row*r.strideAM+k*r.strideAK))
def inputB {α : Type} [Inhabited α] (r : Request) (s : State α) (k col : Nat) :=
  FPValue.read r.storage (s.mem r.B (k*r.strideBK+col*r.strideBN))
def blockStep {α : Type} [Inhabited α] (Q : Numerics α) (r : Request)
    (s : State α) (row col block : Nat) (acc : α) : α :=
  Q.dotAcc (SourceScope.blockK r.storage)
    (fun k => if block*SourceScope.blockK r.storage+k.val<r.inner then
      inputA r s row (block*SourceScope.blockK r.storage+k.val) else Q.loadZero)
    (fun k => if block*SourceScope.blockK r.storage+k.val<r.inner then
      inputB r s (block*SourceScope.blockK r.storage+k.val) col else Q.loadZero) acc
def accumulator {α : Type} [Inhabited α] (Q : Numerics α) (r : Request)
    (s : State α) (row col : Nat) :=
  foldRange (fun block x => blockStep Q r s row col block x) 0 r.parameters.numKBlocks 1 Q.zero
def reference {α : Type} [Inhabited α] (Q : Numerics α) (r : Request)
    (s : State α) (row col : Nat) := Q.castOutput (accumulator Q r s row col)
def outputOffset (r : Request) (row col : Nat) := max r.columns 1*row+col
def Footprint (r : Request) (region : RegionName) (offset : Nat) : Prop :=
  region=r.C ∧ ∃ row<r.rows, ∃ col<r.columns, offset=outputOffset r row col
def Result {α : Type} [Inhabited α] (Q : Numerics α) (r : Request)
    (initial final : State α) : Prop :=
  (∀ row<r.rows, ∀ col<r.columns,
    final.mem r.C (outputOffset r row col)=stored r.storage (reference Q r initial row col)) ∧
  (∀ region offset, ¬Footprint r region offset → final.mem region offset=initial.mem region offset)

def elementBytes : Storage → Nat | .half => 2 | .e4m3 => 1

def safeCoordinate (limit x : Nat) := if x<limit then x else 0
def safeAOffset (r : Request) (row k : Nat) := safeCoordinate r.rows row*r.strideAM+safeCoordinate r.inner k*r.strideAK
def safeBOffset (r : Request) (k col : Nat) := safeCoordinate r.inner k*r.strideBK+safeCoordinate r.columns col*r.strideBN
def safeCOffset (r : Request) (row col : Nat) := max r.columns 1*safeCoordinate r.rows row+safeCoordinate r.columns col

/-- Ordinary valid tensor metadata and physical storage bounds, including zero
strides and arbitrary non-contiguous A/B. No padded-envelope bound is assumed. -/
structure TensorAPI {α : Type} (r : Request) (s : State α) where
  extentA : Nat
  extentB : Nat
  baseA : Nat
  baseB : Nat
  baseC : Nat
  metadataA : MatmulTritonAutotune.Whole.TensorMetadata
  metadataB : MatmulTritonAutotune.Whole.TensorMetadata
  metadataC : MatmulTritonAutotune.Whole.TensorMetadata
  shapeA : metadataA.shape=[r.rows,r.inner]
  shapeB : metadataB.shape=[r.innerB,r.columns]
  shapeC : metadataC.shape=[r.rows,r.columns]
  stridesA : metadataA.strides=[r.strideAM,r.strideAK]
  stridesB : metadataB.strides=[r.strideBK,r.strideBN]
  stridesC : metadataC.strides=[max r.columns 1,1]
  dtypeA : metadataA.dtype=dtype r.storage
  dtypeB : metadataB.dtype=dtype r.storage
  dtypeC : metadataC.dtype=dtype r.storage
  deviceB : metadataB.device=metadataA.device
  deviceC : metadataC.device=metadataA.device
  dimensions : r.inner=r.innerB
  storage : r.storage=r.storageB
  dimM : r.rows<2^63
  dimN : r.columns<2^63
  dimK : r.inner<2^63
  freshA : r.C≠r.A
  freshB : r.C≠r.B
  physicalFreshA : MatmulTritonAutotune.Whole.byteSpansDisjoint baseC (elementBytes r.storage*(r.rows*r.columns)) baseA (elementBytes r.storage*extentA)
  physicalFreshB : MatmulTritonAutotune.Whole.byteSpansDisjoint baseC (elementBytes r.storage*(r.rows*r.columns)) baseB (elementBytes r.storage*extentB)
  alignmentA : baseA%elementBytes r.storage=0
  alignmentB : baseB%elementBytes r.storage=0
  alignmentC : baseC%elementBytes r.storage=0
  spanA : baseA+elementBytes r.storage*extentA≤2^64
  spanB : baseB+elementBytes r.storage*extentB≤2^64
  spanC : baseC+elementBytes r.storage*(r.rows*r.columns)≤2^64
  allocationA : elementBytes r.storage*extentA<2^63
  allocationB : elementBytes r.storage*extentB<2^63
  allocationC : elementBytes r.storage*(r.rows*r.columns)<2^63
  strideAM : r.strideAM<2^63
  strideAK : r.strideAK<2^63
  strideBK : r.strideBK<2^63
  strideBN : r.strideBN<2^63
  inputA : ∀ row<r.rows, ∀ k<r.inner, row*r.strideAM+k*r.strideAK<extentA
  inputB : ∀ k<r.inner, ∀ col<r.columns, k*r.strideBK+col*r.strideBN<extentB
  tagA : ∀ row<r.rows, ∀ k<r.inner, ∃ x, s.mem r.A (row*r.strideAM+k*r.strideAK)=stored r.storage x
  tagB : ∀ k<r.inner, ∀ col<r.columns, ∃ x, s.mem r.B (k*r.strideBK+col*r.strideBN)=stored r.storage x
  sameViewOrigin : r.A=r.B → baseA=baseB
  aliasConsistency : ∀ row<r.rows, ∀ k<r.inner, ∀ k'<r.inner, ∀ col<r.columns,
    baseA+elementBytes r.storage*(row*r.strideAM+k*r.strideAK)=baseB+elementBytes r.storage*(k'*r.strideBK+col*r.strideBN) →
    s.mem r.A (row*r.strideAM+k*r.strideAK)=s.mem r.B (k'*r.strideBK+col*r.strideBN)

/-- Only newly allocated resolve_neg buffers and the fresh output may change during preparation.
The output is freshly allocated, never a supplied output view; its identity
is exactly r.C. The caller's A/B allocations are not modified. -/
def CopyFootprint {α : Type} {r : Request} {s : State α} (t : TensorAPI r s)
    (copyA copyB : Bool) (region : RegionName) (offset : Nat) : Prop :=
  (copyA=Bool.true ∧ region=r.A ∧ offset<t.extentA) ∨
  (copyB=Bool.true ∧ region=r.B ∧ offset<t.extentB)

/-- Standard coordinate-preserving resolve_neg contract. Logical arrays are
independent caller inputs, not raw-memory definitions. No false-negative-bit
premise is imposed, and the materialized strides may differ from the inputs. -/
structure ResolveNegContract {α : Type} [Inhabited α] (r : Request)
    (before prepared : State α) (t : TensorAPI r prepared)
    (A : Fin r.rows → Fin r.inner → α) (B : Fin r.inner → Fin r.columns → α) where
  copyA : Bool
  copyB : Bool
  originalA : RegionName
  originalB : RegionName
  originalBaseA : Nat
  originalBaseB : Nat
  originalExtentA : Nat
  originalExtentB : Nat
  outputByteFreshA : MatmulTritonAutotune.Whole.byteSpansDisjoint t.baseC (elementBytes r.storage*(r.rows*r.columns)) originalBaseA (elementBytes r.storage*originalExtentA)
  outputByteFreshB : MatmulTritonAutotune.Whole.byteSpansDisjoint t.baseC (elementBytes r.storage*(r.rows*r.columns)) originalBaseB (elementBytes r.storage*originalExtentB)
  copyAByteFreshA : copyA=Bool.true → MatmulTritonAutotune.Whole.byteSpansDisjoint t.baseA (elementBytes r.storage*t.extentA) originalBaseA (elementBytes r.storage*originalExtentA)
  copyAByteFreshB : copyA=Bool.true → MatmulTritonAutotune.Whole.byteSpansDisjoint t.baseA (elementBytes r.storage*t.extentA) originalBaseB (elementBytes r.storage*originalExtentB)
  copyBByteFreshA : copyB=Bool.true → MatmulTritonAutotune.Whole.byteSpansDisjoint t.baseB (elementBytes r.storage*t.extentB) originalBaseA (elementBytes r.storage*originalExtentA)
  copyBByteFreshB : copyB=Bool.true → MatmulTritonAutotune.Whole.byteSpansDisjoint t.baseB (elementBytes r.storage*t.extentB) originalBaseB (elementBytes r.storage*originalExtentB)
  outputFreshA : r.C≠originalA
  outputFreshB : r.C≠originalB
  copyAFreshA : copyA=Bool.true → r.A≠originalA
  copyAFreshB : copyA=Bool.true → r.A≠originalB
  copyBFreshA : copyB=Bool.true → r.B≠originalA
  copyBFreshB : copyB=Bool.true → r.B≠originalB
  logicalA : ∀ i k, inputA r prepared i.val k.val=A i k
  logicalB : ∀ k j, inputB r prepared k.val j.val=B k j
  frame : ∀ region offset, ¬Footprint r region offset →
    ¬CopyFootprint t copyA copyB region offset →
    prepared.mem region offset=before.mem region offset

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair
