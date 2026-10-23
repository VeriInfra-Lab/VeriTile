import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.TypedRealGrid
import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.BoundDimension
import VeriTile.Triton.Float.ProgramIO
import VeriTile.Triton.Float.LogicalProgramIO

/-! Typed public windows and the actual first/second/dimension source schedule.
Requested precision is selected by the existing binding, including the whole
integer conversion macro. Physical codecs and byte transport stay connected in
BoundSource and the retained physical results, independently of this IO layout.
-/
namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.IO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
noncomputable section

/-- Real semantics erase the requested integer conversion macro as before. -/
def projectedKind : Request.Kind → TypedScratch.Kind
  | .floating k => k
  | .integer _ => .fp32

/-- Install only axis zero; all other launch/template fields are preserved. -/
def atProgram {α : Type} (backend : Algebra α) (kernel : ComputeKernel)
    (pid : Nat) : Invocation α where
  kernel := kernel
  backend := backend
  pids := fun _ => pid
  setPids := fun ax => decide (ax = 0)

/-- Stage two reads no PID and keeps the preceding launch context unchanged. -/
def unchanged {α : Type} (backend : Algebra α) (kernel : ComputeKernel) : Invocation α where
  kernel := kernel
  backend := backend
  pids := fun _ => 0
  setPids := fun _ => Bool.false

@[simp] theorem atProgram_initial {α : Type} (backend : Algebra α)
    (kernel : ComputeKernel) (pid : Nat) (s : State α) :
    (atProgram backend kernel pid).initial s =
      { s with pids := fun ax => if ax = 0 then pid else s.pids ax } := by
  simp [atProgram, Invocation.initial, bindRegisters]

@[simp] theorem unchanged_initial {α : Type} (backend : Algebra α)
    (kernel : ComputeKernel) (s : State α) : (unchanged backend kernel).initial s = s := by
  simp [unchanged, Invocation.initial, bindRegisters]

/-- Complete real first grid, then the actual second quotation. -/
def realSchedule (kind : TypedScratch.Kind) (rest : Algebra Math.Value)
    (X V : RegionName) (I O : Region .int) (c : GlobalConfig) (index64 : Bool) :
    List (Invocation Math.Value) :=
  (List.range c.blocks).map (atProgram (TypedReal.backend rest)
    (TypedScratch.firstKernel kind X V I c.size c.block 0 index64)) ++
  [unchanged (TypedReal.backend rest) (TypedScratch.secondKernel kind V I O c.blocks c.midBlock)]

/-- Active input, scalar INT64 output, and both complete first-grid scratch
windows. Original arbitrary region names and permitted aliases are retained. -/
@[reducible] def realGlobal (kind : TypedScratch.Kind) (rest : Algebra Math.Value)
    (X V : RegionName) (I O : Region .int) (c : GlobalConfig) (index64 : Bool) :
    ProgramIO ℝ Math.Value where
  inputs := [⟨.real, [c.size], X⟩]
  outputs := [⟨.int, [1], Region.cast O⟩]
  scratch := [⟨(TypedScratch.channel kind).toTileDType, [c.blocks], V⟩,
    ⟨.int, [c.blocks], Region.cast I⟩]
  buffers := [X, V, Region.cast I, Region.cast O]
  read := Fin.cases (fun _ j => j.1.val) (fun i => Fin.elim0 i)
  inputMask := fun _ _ _ => True
  write := fun _ _ _ => 0
  outputMask := fun _ _ _ => True
  scratchWrite := Fin.cases (fun _ j => j.1.val)
    (Fin.cases (fun _ j => j.1.val) (fun i => Fin.elim0 i))
  scratchMask := fun _ _ _ => True
  schedule := fun _ => realSchedule kind rest X V I O c index64
  encode := fun x => (x : Math.Value)
  observe := fun _ _ _ => 0

/-- All thirteen requested formats use their actual source pair and existing
scalar model. Integer scratch remains an integer channel. -/
def nativeSchedule (casts : Request.Casts) (kind : Request.Kind)
    (binding : BoundSource.Binding casts kind) (rest : Algebra Word)
    (c : GlobalConfig) (index64 : Bool) : List (Invocation Word) :=
  (List.range c.blocks).map (atProgram (binding.ops rest) (binding.first c index64)) ++
  [unchanged (binding.ops rest) (binding.second c)]

def scratchType {casts kind} : BoundSource.Binding casts kind → TileDType
  | .floating k _ _ => (TypedScratch.channel k).toTileDType
  | .integer _ _ => .int

/-- INT64 scratch is an initialized input over the whole gather window. Only
active first-grid entries are writable scratch; tail cells remain framed. -/
@[reducible] def nativeGlobal (casts : Request.Casts) (kind : Request.Kind)
    (binding : BoundSource.Binding casts kind) (rest : Algebra Word)
    (c : GlobalConfig) (index64 : Bool) : ProgramIO Word Word where
  inputs := [⟨.real, [c.size], BoundSource.X⟩,
    ⟨.int, [c.midBlock], Region.cast BoundSource.I⟩]
  outputs := [⟨.int, [1], Region.cast BoundSource.O⟩]
  scratch := [⟨scratchType binding, [c.blocks], BoundSource.V⟩,
    ⟨.int, [c.blocks], Region.cast BoundSource.I⟩]
  buffers := [BoundSource.X, BoundSource.V, Region.cast BoundSource.I, Region.cast BoundSource.O]
  read := Fin.cases (fun _ j => j.1.val)
    (Fin.cases (fun _ j => j.1.val) (fun i => Fin.elim0 i))
  inputMask := fun _ _ _ => True
  write := fun _ _ _ => 0
  outputMask := fun _ _ _ => True
  scratchWrite := Fin.cases (fun _ j => j.1.val)
    (Fin.cases (fun _ j => j.1.val) (fun i => Fin.elim0 i))
  scratchMask := fun _ _ _ => True
  schedule := fun _ => nativeSchedule casts kind binding rest c index64
  encode := id
  observe := fun _ _ => id

def globalInputs (kind : TypedScratch.Kind) (rest : Algebra Math.Value)
    (X V : RegionName) (I O : Region .int) (c : GlobalConfig) (index64 : Bool) :
    (realGlobal kind rest X V I O c index64).Inputs ≃ (Fin c.size → ℝ) where
  toFun xs i := xs 0 (i,())
  invFun xs := Fin.cases (fun i => xs i.1) (fun i => Fin.elim0 i)
  left_inv := by intro xs; funext o i; fin_cases o; rcases i with ⟨j,u⟩; cases u; rfl
  right_inv := by intro xs; rfl

def globalOutputs (kind : TypedScratch.Kind) (rest : Algebra Math.Value)
    (X V : RegionName) (I O : Region .int) (c : GlobalConfig) (index64 : Bool) :
    (realGlobal kind rest X V I O c index64).Outputs ≃ Int where
  toFun xs := xs 0 (0,())
  invFun xs := Fin.cases (fun _ => xs) (fun i => Fin.elim0 i)
  left_inv := by
    intro xs; funext o i; fin_cases o; rcases i with ⟨j,u⟩; cases u
    have h : j=0 := Fin.eq_zero j; subst j; rfl
  right_inv := by intro xs; rfl

def logicalGlobal (kind : TypedScratch.Kind) (rest : Algebra Math.Value)
    (X V : RegionName) (I O : Region .int) (c : GlobalConfig) (index64 : Bool) :
    LogicalProgramIO ℝ Math.Value (Fin c.size → ℝ) Int where
  core := realGlobal kind rest X V I O c index64
  inputs := globalInputs kind rest X V I O c index64
  outputs := globalOutputs kind rest X V I O c index64

end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.IO

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.NativeDimensionIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
noncomputable section

def launch (rest : Algebra Word) (c : DimConfig) (index64 : Bool) (job : Nat × Nat) : Invocation Word :=
  IO.atProgram (Typed.backend rest)
    (TypedSource.dimension BoundSource.X BoundSource.O c.outer c.extent c.inner 128 128 0 index64)
    (job.1*c.inner+job.2)

end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.NativeDimensionIO

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.IO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
noncomputable section

/-- Original arbitrary-size mathematical dimension program. Its source loops
over the entire reduced axis; only active output rows are writable. -/
@[reducible] def realDimension (rest : Algebra Math.Value) (X : RegionName) (O : Region .int)
    (M N K BM BN base pid : Nat) (index64 : Bool) : ProgramIO ℝ Math.Value where
  inputs := [⟨.real,[M,N,K],X⟩]
  outputs := [⟨.int,[BM],Region.cast O⟩]
  scratch := []
  buffers := [X,Region.cast O]
  read := Fin.cases (fun _ j => j.1.val*N*K+j.2.1.val*K+j.2.2.1.val) (fun i => Fin.elim0 i)
  inputMask := fun _ _ _ => True
  write := Fin.cases (fun _ j => (((pid+base)/K)*BM+j.1.val)*K+(pid+base)%K) (fun i => Fin.elim0 i)
  outputMask := Fin.cases (fun _ j => ((pid+base)/K)*BM+j.1.val<M) (fun i => Fin.elim0 i)
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i
  schedule := fun _ => [atProgram (TypedReal.backend rest)
    (TypedSource.dimension X O M N K BM BN base index64) pid]
  encode := fun x => (x : Math.Value)
  observe := fun _ _ _ => 0

/-- Every physical dimension CTA is represented by its complete quoted source,
with the same 128-row/128-column tiles and two-dimensional launch enumeration. -/
def nativeDimensionSchedule (rest : Algebra Word) (c : DimConfig) (index64 : Bool) :
    List (Invocation Word) :=
  (BoundSource.groups c).map (NativeDimensionIO.launch rest c index64)

@[reducible] def nativeDimension (rest : Algebra Word) (c : DimConfig) (index64 : Bool) :
    ProgramIO Word Word where
  inputs := [⟨.real,[c.outer,c.extent,c.inner],BoundSource.X⟩]
  outputs := [⟨.int,[c.outer,c.inner],Region.cast BoundSource.O⟩]
  scratch := []
  buffers := [BoundSource.X,Region.cast BoundSource.O]
  read := Fin.cases (fun _ j => j.1.val*c.extent*c.inner+j.2.1.val*c.inner+j.2.2.1.val) (fun i => Fin.elim0 i)
  inputMask := fun _ _ _ => True
  write := Fin.cases (fun _ j => j.1.val*c.inner+j.2.1.val) (fun i => Fin.elim0 i)
  outputMask := fun _ _ _ => True
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i
  schedule := fun _ => nativeDimensionSchedule rest c index64
  encode := id
  observe := fun _ _ => id

def dimensionInputs (rest : Algebra Math.Value) (X : RegionName) (O : Region .int)
    (M N K BM BN base pid : Nat) (index64 : Bool) :
    (realDimension rest X O M N K BM BN base pid index64).Inputs ≃ (Fin M → Fin N → Fin K → ℝ) where
  toFun xs i j k := xs 0 (i,j,k,())
  invFun xs := Fin.cases (fun i => xs i.1 i.2.1 i.2.2.1) (fun i => Fin.elim0 i)
  left_inv := by intro xs; funext o i; fin_cases o; rcases i with ⟨i,j,k,u⟩; cases u; rfl
  right_inv := by intro xs; rfl

def dimensionOutputs (rest : Algebra Math.Value) (X : RegionName) (O : Region .int)
    (M N K BM BN base pid : Nat) (index64 : Bool) :
    (realDimension rest X O M N K BM BN base pid index64).Outputs ≃ (Fin BM → Int) where
  toFun xs j := xs 0 (j,())
  invFun xs := Fin.cases (fun i => xs i.1) (fun i => Fin.elim0 i)
  left_inv := by intro xs; funext o i; fin_cases o; rcases i with ⟨j,u⟩; cases u; rfl
  right_inv := by intro xs; rfl

def logicalDimension (rest : Algebra Math.Value) (X : RegionName) (O : Region .int)
    (M N K BM BN base pid : Nat) (index64 : Bool) :
    LogicalProgramIO ℝ Math.Value (Fin M → Fin N → Fin K → ℝ) (Fin BM → Int) where
  core := realDimension rest X O M N K BM BN base pid index64
  inputs := dimensionInputs rest X O M N K BM BN base pid index64
  outputs := dimensionOutputs rest X O M N K BM BN base pid index64

/-- The full reduction row selected by this dimension CTA; out-of-range rows
are totalized only for the existing inactive output lanes. -/
def dimensionRow (M N K BM base pid : Nat) (xs : Fin M → Fin N → Fin K → ℝ)
    (j : Fin BM) (col : Fin N) : ℝ :=
  if hr : ((pid+base)/K)*BM+j.val < M then
    if hk : (pid+base)%K < K then xs ⟨_,hr⟩ col ⟨_,hk⟩ else 0
  else 0

end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.IO
