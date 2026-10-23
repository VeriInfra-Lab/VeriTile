import bench.repairs.tritonbench_g.context_attn_fwd.Kernels
import bench.tritonbench_g.context_attn_fwd.Proofs.RealIO
import VeriTile.Triton.Memory.AlgorithmProgramIO
import VeriTile.Triton.Memory.LogicalIO

/-! Finite typed windows for the original complete ordinary Real grid.
All CTAs read the same initial arrays, as required by the established ordinary
grid semantics. Output channels enumerate every grid/lane pair. -/
namespace VeriTile.Bench.Repairs.ContextAttnFwd.RealInterface
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnFwd
noncomputable section
open Classical

def inputRegion (r : Regions) (i : Fin 6) : RegionName :=
  [r.query,r.key,r.value,r.start,r.sequence,r.prompt][i]

def inputType (i : Fin 6) : ChanTy := if i.val < 3 then .float else .int

abbrev Values (r : Regions) (extent : RegionBounds) :=
  (i : Fin 6) → Fin (extent (inputRegion r i)) → (inputType i).carrier

def arrays (r : Regions) (extent : RegionBounds) (v : Values r extent) : Formula.Inputs where
  q := fun off => if h : off < extent r.query then v 0 ⟨off,h⟩ else 0
  k := fun off => if h : off < extent r.key then v 1 ⟨off,h⟩ else 0
  v := fun off => if h : off < extent r.value then v 2 ⟨off,h⟩ else 0
  start := fun off => if h : off < extent r.start then v 3 ⟨off,h⟩ else 0
  sequence := fun off => if h : off < extent r.sequence then v 4 ⟨off,h⟩ else 0
  prompt := fun off => if h : off < extent r.prompt then v 5 ⟨off,h⟩ else 0

abbrev Event (p : Layout) (g : Grid) := GridIndex g × TileIndex [p.BM,p.D]

def decode (p : Layout) (g : Grid) : Fin (Fintype.card (Event p g)) → Event p g :=
  (Fintype.equivFin (Event p g)).symm

def encode (p : Layout) (g : Grid) : Event p g → Fin (Fintype.card (Event p g)) :=
  Fintype.equivFin (Event p g)

def launch {g : Grid} (idx : GridIndex g) : Formula.Launch := ⟨idx.toPids 0,idx.toPids 1⟩

def active (p : Layout) (g : Grid) (a : Formula.Inputs) (e : Event p g) : Prop :=
  RealIO.active p (launch e.1) a e.2

def address (p : Layout) (g : Grid) (a : Formula.Inputs) (e : Event p g) : Nat :=
  RealIO.write p (launch e.1) a e.2

/-- Original nonnegative formed-address conditions and complete-grid unique
active output ownership. No execution or output equality is a premise. -/
def Domain (p : Layout) (r : Regions) (extent : RegionBounds) (g : Grid) (v : Values r extent) : Prop :=
  (∀ idx : GridIndex g,AddressBounds.Valid p r (launch idx) (arrays r extent v) extent) ∧
  (∀ e f : Event p g,active p g (arrays r extent v) e → active p g (arrays r extent v) f →
    address p g (arrays r extent v) e=address p g (arrays r extent v) f → e=f)

def source (p : Layout) (r : Regions) (scale : ℝ) : AlgorithmSource where
  kernel := (kernel p r .fp32 scale).eraseDType
  projection := RealIO.projection p r .fp32 Bool.false scale

def io (p : Layout) (r : Regions) (extent : RegionBounds) (g : Grid) (scale : ℝ) : AlgorithmProgramIO where
  nIn := 6
  nOut := 1
  nScr := 0
  bufs := [r.query,r.key,r.value,r.start,r.sequence,r.prompt,r.out]
  ity := inputType
  iarity := fun i => extent (inputRegion r i)
  ibuf := inputRegion r
  oty := fun _ => .float
  oarity := fun _ => Fintype.card (Event p g)
  obuf := fun _ => r.out
  obuf_mem := by intro o; simp
  sarity := Fin.elim0
  sbuf := Fin.elim0
  iwin := fun _ _ _ _ _ j => j.val
  imask := fun _ _ _ _ _ _ => True
  owin := fun _ v _ _ _ j => address p g (arrays r extent v) (decode p g j)
  omask := fun _ v _ _ _ j => active p g (arrays r extent v) (decode p g j)
  swin := fun k => Fin.elim0 k
  smask := fun k => Fin.elim0 k
  stages := fun _ => [.ordinary (source p r scale) g]

/-- Independent base-2 weighted sum, preserving the finite sentinel and the
padded denominator contributions of the original mathematical specification. -/
def formula (p : Layout) (r : Regions) (extent : RegionBounds) (g : Grid) (scale : ℝ)
    (v : Values r extent) (_ : Fin 1) (j : Fin (Fintype.card (Event p g))) : ℝ :=
  let e:=decode p g j
  RealIO.closedExpected p scale (launch e.1) (arrays r extent v) e.2

#specNonCircular formula avoiding [exec,Kernel.GridLaunchedOrdinary,Program.inputs,kernel]
#specNonCircular Domain avoiding [exec,Kernel.GridLaunchedOrdinary,Program.inputs,kernel]
end
end VeriTile.Bench.Repairs.ContextAttnFwd.RealInterface

namespace VeriTile.Bench.Repairs.ContextAttnFwd.RealInterface.Logical
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnFwd
noncomputable section

/-- Complete finite backing, including all signed metadata, aliases and
padding. Logical arrays below do not replace or restrict this storage. -/
structure Backing (r : Regions) (extent : RegionBounds) where
  storage : Values r extent

/-- Actual mathematical axes: source launch, query/key row and feature.
Total zero/mask extensions and signed metadata follow the original formula. -/
structure Arrays where
  query : Formula.Launch → Nat → Nat → ℝ
  key : Formula.Launch → Nat → Nat → ℝ
  value : Formula.Launch → Nat → Nat → ℝ
  blocks : Formula.Launch → Nat
  start : Nat → Int
  sequence : Nat → Int
  prompt : Nat → Int

def mathematicalArrays (p : Formula.Layout) (a : Formula.Inputs) : Arrays where
  query := fun launch row feature => Formula.query p launch a row feature
  key := fun launch row feature => Formula.key p launch a row feature
  value := fun launch row feature => Formula.value p launch a row feature
  blocks := fun launch => Formula.count p launch a
  start := a.start
  sequence := a.sequence
  prompt := a.prompt

def arraysOfBacking (p : Layout) (r : Regions) (extent : RegionBounds) (backing : Backing r extent) : Arrays :=
  mathematicalArrays (p) (RealInterface.arrays r extent backing.storage)

/-- A graph, rather than an additional consistency precondition: for every
old backing there is exactly one connected mathematical array record. -/
structure Inputs (p : Layout) (r : Regions) (extent : RegionBounds) where
  arrays : Arrays
  backing : Backing r extent
  connection : arrays = arraysOfBacking p r extent backing

def inputs (p : Layout) (r : Regions) (extent : RegionBounds) : Values r extent ≃ Inputs p r extent where
  toFun := fun raw => ⟨arraysOfBacking p r extent ⟨raw⟩,⟨raw⟩,rfl⟩
  invFun := fun values => values.backing.storage
  left_inv := fun _ => rfl
  right_inv := by
    rintro ⟨arrays,⟨storage⟩,connection⟩
    cases connection
    rfl

/-- Logical output coordinates enumerate precisely the original output cells. -/
def outputs (p : Layout) (r : Regions) (extent : RegionBounds) (g : Grid) (scale : ℝ) :
    (Fin 1 → Fin (Fintype.card (Event p g)) → ℝ) ≃ (Event p g → ℝ) where
  toFun := fun values coordinate => values 0 ((Fintype.equivFin (Event p g)) coordinate)
  invFun := fun values _ j => values ((Fintype.equivFin (Event p g)).symm j)
  left_inv := by
    intro values
    funext channel j
    have channel_zero : channel = 0 := Fin.eq_zero channel
    subst channel
    exact congrArg (values 0) ((Fintype.equivFin (Event p g)).apply_symm_apply j)
  right_inv := by
    intro values
    funext coordinate
    exact congrArg values ((Fintype.equivFin (Event p g)).symm_apply_apply coordinate)

def domain (p : Layout) (r : Regions) (extent : RegionBounds) (g : Grid) (scale : ℝ)
    (values : Inputs p r extent) : Prop := Domain p r extent g values.backing.storage

def io (p : Layout) (r : Regions) (extent : RegionBounds) (g : Grid) (scale : ℝ) :
    LogicalDomainIO (Values r extent) (Fin 1 → Fin (Fintype.card (Event p g)) → ℝ)
      (Inputs p r extent) (Event p g → ℝ) (RealInterface.io p r extent g scale).ImplementsOn where
  inputs := inputs p r extent
  outputs := outputs p r extent g scale

end
end VeriTile.Bench.Repairs.ContextAttnFwd.RealInterface.Logical
