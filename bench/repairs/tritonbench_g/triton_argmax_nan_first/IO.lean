import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.TypedRealGrid
import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.JITInteger
import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.JITDimensionStep
import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.BoundDimension
import VeriTile.Triton.Float.ProgramIO
import VeriTile.Triton.Float.LogicalProgramIO

/-! Typed public windows and the actual first/second/dimension source schedule.
Requested precision is selected by the existing binding, including explicit scalar
integer conversion instructions. Physical codecs and byte transport stay connected in
BoundSource and the retained physical results, independently of this IO layout.
-/

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITBinding
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
noncomputable section

def dtype {casts kind} : BoundSource.Binding casts kind → SourceDType
  | .floating k _ _ => JITStage1.floatingType k
  | .integer k _ => JITInteger.sourceType k

def ops {casts kind} : BoundSource.Binding casts kind → Algebra Word → Algebra Word
  | .floating _ round _ => TypedScratch.backend round
  | .integer k _ => JITInteger.backend k (casts.integer k)

def cell {casts kind} : BoundSource.Binding casts kind → MidWord kind.format → Cell Word
  | .floating k _ codec => fun word => .mk (TypedScratch.channel k).toTileDType
      (ofFloat (TypedScratch.channel k) (codec.decode word))
  | .integer k _ => fun word => JITInteger.cell k (word.cast (Formats.format_width k).symm)

def first {casts kind} (binding : BoundSource.Binding casts kind) (c : GlobalConfig) (index64 : Bool) :=
  JITSource.stage1 BoundSource.X BoundSource.V BoundSource.I (dtype binding) c.block index64

def second {casts kind} (binding : BoundSource.Binding casts kind) (c : GlobalConfig) :=
  JITSource.stage2 BoundSource.V BoundSource.I BoundSource.O (dtype binding) c.midBlock
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITBinding

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
noncomputable section

/-- The Real shadow specializes the idealized conversion projection; it does not
evaluate finite integer clamps as Real identities. The native refinement is
proved instruction by instruction in JITIntegerStage1. -/
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


/-- Host scalar bindings are data supplied at the launch boundary. -/
def firstLaunch {α : Type} (backend : Algebra α) (kernel : ComputeKernel)
    (size base pid : Nat) : Invocation α :=
  { atProgram backend kernel pid with arguments :=
    [⟨"M", .int, [], fun _ => (size : Int)⟩,
     ⟨"program_base", .int, [], fun _ => (base : Int)⟩] }

def secondLaunch {α : Type} (backend : Algebra α) (kernel : ComputeKernel)
    (size : Nat) : Invocation α :=
  { unchanged backend kernel with arguments := [⟨"mid_size", .int, [], fun _ => (size : Int)⟩] }

def dimensionLaunch {α : Type} (backend : Algebra α) (kernel : ComputeKernel)
    (M N K base pid : Nat) : Invocation α :=
  { atProgram backend kernel pid with arguments :=
    [⟨"M", .int, [], fun _ => (M : Int)⟩, ⟨"N", .int, [], fun _ => (N : Int)⟩,
     ⟨"K", .int, [], fun _ => (K : Int)⟩, ⟨"program_base", .int, [], fun _ => (base : Int)⟩] }

@[simp] theorem firstLaunch_initial {α : Type} (backend : Algebra α) (kernel : ComputeKernel)
    (size base pid : Nat) (s : State α) :
    (firstLaunch backend kernel size base pid).initial s =
      JITStage1.arguments ((atProgram backend kernel pid).initial s) size base := rfl

@[simp] theorem secondLaunch_initial {α : Type} (backend : Algebra α) (kernel : ComputeKernel)
    (size : Nat) (s : State α) :
    (secondLaunch backend kernel size).initial s =
      s.setReg "mid_size" .int [] (fun _ => (size : Int)) := by
  simp [secondLaunch,unchanged,Invocation.initial,bindRegisters]

@[simp] theorem dimensionLaunch_initial {α : Type} (backend : Algebra α) (kernel : ComputeKernel)
    (M N K base pid : Nat) (s : State α) :
    (dimensionLaunch backend kernel M N K base pid).initial s =
      JITDimension.arguments ((atProgram backend kernel pid).initial s) M N K base := rfl

/-- Complete real first grid, then the actual second quotation. -/
def realSchedule (kind : TypedScratch.Kind) (rest : Algebra Math.Value)
    (X V : RegionName) (I O : Region .int) (c : GlobalConfig) (index64 : Bool) :
    List (Invocation Math.Value) :=
  (List.range c.blocks).map (firstLaunch (TypedReal.backend rest)
    (JITSource.stage1 X V I (JITStage1.floatingType kind) c.block index64) c.size 0) ++
  [secondLaunch (TypedReal.backend rest) (JITSource.stage2 V I O (JITStage1.floatingType kind) c.midBlock) c.blocks]

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
scalar model. Signed, unsigned and boolean scratch retain their respective channels. -/
def nativeSchedule (casts : Request.Casts) (kind : Request.Kind)
    (binding : BoundSource.Binding casts kind) (rest : Algebra Word)
    (c : GlobalConfig) (index64 : Bool) : List (Invocation Word) :=
  (List.range c.blocks).map (firstLaunch (JITBinding.ops binding rest) (JITBinding.first binding c index64) c.size 0) ++
  [secondLaunch (JITBinding.ops binding rest) (JITBinding.second binding c) c.blocks]

def scratchType {casts kind} : BoundSource.Binding casts kind → TileDType
  | .floating k _ _ => (TypedScratch.channel k).toTileDType
  | .integer k _ => JITInteger.channel k

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
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITIO

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITNativeDimensionIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
noncomputable section

def launch (rest : Algebra Word) (c : DimConfig) (index64 : Bool) (job : Nat × Nat) : Invocation Word :=
  JITIO.dimensionLaunch (Typed.backend rest)
    (JITSource.dimension BoundSource.X BoundSource.O 128 128 index64)
    c.outer c.extent c.inner 0 (job.1*c.inner+job.2)

end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITNativeDimensionIO

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITIO
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
  schedule := fun _ => [dimensionLaunch (TypedReal.backend rest)
    (JITSource.dimension X O BM BN index64) M N K base pid]
  encode := fun x => (x : Math.Value)
  observe := fun _ _ _ => 0

/-- Every physical dimension CTA is represented by its complete quoted source,
with the same 128-row/128-column tiles and two-dimensional launch enumeration. -/
def nativeDimensionSchedule (rest : Algebra Word) (c : DimConfig) (index64 : Bool) :
    List (Invocation Word) :=
  (BoundSource.groups c).map (JITNativeDimensionIO.launch rest c index64)

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
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITIO
