import bench.tritonbench_g.context_attn_llama.Proofs.Mathematical_HostReadback
import VeriTile.Triton.Memory.AlgorithmProgramIO
import bench.tritonbench_g.context_attn_llama.Kernels
import VeriTile.Triton.Memory.LogicalIO

/-! Public regional IO of the complete original ordinary grid. The existing
flat-allocation transport remains a separate physical-memory theorem. -/
namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.RealInterface
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.ContextAttnLlama
noncomputable section
open scoped Classical

/-- Original layout, capacity and output-ownership conditions, without any
operational state or execution premise. Input metadata is bound separately. -/
structure Geometry (c : Host.Call) (capacity : RegionBounds) : Prop where
  prompt_bound : c.batches ≤ capacity c.regions.prompt
  request_bound : c.batches ≤ capacity c.regions.request
  page_bound : ∀ b, b < c.batches → ∀ j, j < c.length b+c.prompt b →
    c.layout.reqRow*c.request b+c.layout.reqToken*j < capacity c.regions.pages
  start_bound : c.batches ≤ capacity c.regions.start
  length_bound : c.batches ≤ capacity c.regions.sequence
  group_pos : 0 < c.layout.kvGroup
  grouped : c.layout.heads = c.kvHeads*c.layout.kvGroup
  query : ∀ b, b < c.batches → ∀ h, h < c.layout.heads → ∀ row, row < c.length b →
    ∀ d, d < c.layout.D →
      (c.start b+row)*c.layout.qRow+h*c.layout.qHead+d*c.layout.qDim < capacity c.regions.query
  key : ∀ b, b < c.batches → ∀ h, h < c.layout.heads → ∀ row, row < c.length b+c.prompt b →
    ∀ d, d < c.layout.D →
      c.page b row*c.layout.kRow+(h/c.layout.kvGroup)*c.layout.kHead+d*c.layout.kDim < capacity c.regions.key
  value : ∀ b, b < c.batches → ∀ h, h < c.layout.heads → ∀ row, row < c.length b+c.prompt b →
    ∀ d, d < c.layout.D →
      c.page b row*c.layout.vRow+(h/c.layout.kvGroup)*c.layout.vHead+d*c.layout.vDim < capacity c.regions.value
  out : ∀ b, b < c.batches → ∀ h, h < c.layout.heads → ∀ row, row < c.length b →
    ∀ d, d < c.layout.D → c.output b h row d < capacity c.regions.out
  packed : ∀ b b' row row', b < c.batches → b' < c.batches →
    row < c.length b → row' < c.length b' →
    c.start b+row = c.start b'+row' → b=b' ∧ row=row'
  output_view : ∀ b b' h h' row row' d d', b < c.batches → b' < c.batches →
    h < c.layout.heads → h' < c.layout.heads → row < c.length b → row' < c.length b' →
    d < c.layout.D → d' < c.layout.D →
    c.output b h row d = c.output b' h' row' d' →
    c.start b+row = c.start b'+row' ∧ h=h' ∧ d=d'
  input_separate : ∀ rg, rg ∈ [c.regions.query,c.regions.key,c.regions.value,
    c.regions.start,c.regions.sequence,c.regions.prompt,c.regions.request,c.regions.pages] → c.regions.out ≠ rg

def inputRegion (c : Host.Call) (i : Fin 8) : RegionName :=
  match i with
  | ⟨0,_⟩ => c.regions.query | ⟨1,_⟩ => c.regions.key | ⟨2,_⟩ => c.regions.value
  | ⟨3,_⟩ => c.regions.start | ⟨4,_⟩ => c.regions.sequence
  | ⟨5,_⟩ => c.regions.prompt | ⟨6,_⟩ => c.regions.request | ⟨7,_⟩ => c.regions.pages

def inputType (i : Fin 8) : ChanTy := if i.val<3 then .float else .int
abbrev Inputs (c : Host.Call) (capacity : RegionBounds) :=
  (i : Fin 8) → Fin (capacity (inputRegion c i)) → (inputType i).carrier
abbrev Lane (c : Host.Call) := GridIndex c.grid × RealIO.Lane c.layout
def laneCount (c : Host.Call) := Fintype.card (Lane c)
def lanes (c : Host.Call) : Lane c ≃ Fin (laneCount c) := Fintype.equivFin _

def array {α : Type} [Zero α] (xs : Fin n → α) (i : Nat) : α :=
  if h : i<n then xs ⟨i,h⟩ else 0

def logicalInputs (c : Host.Call) (capacity : RegionBounds) (values : Inputs c capacity) : Formula.Inputs where
  q := array (values 0)
  k := array (values 1)
  v := array (values 2)
  start := array (values 3)
  sequence := array (values 4)
  prompt := array (values 5)
  request := array (values 6)
  pages := array (values 7)

/-- The original metadata values, on the admitted finite arrays only. -/
structure Metadata (c : Host.Call) (capacity : RegionBounds) (values : Inputs c capacity) : Prop where
  start : ∀ b,b<c.batches → (logicalInputs c capacity values).start b=(c.start b : Int)
  sequence : ∀ b,b<c.batches → (logicalInputs c capacity values).sequence b=((c.length b+c.prompt b : Nat) : Int)
  prompt : ∀ b,b<c.batches → (logicalInputs c capacity values).prompt b=(c.prompt b : Int)
  request : ∀ b,b<c.batches → (logicalInputs c capacity values).request b=(c.request b : Int)
  pages : ∀ b,b<c.batches → ∀ j,j<c.length b+c.prompt b →
    (logicalInputs c capacity values).pages (c.layout.reqRow*c.request b+c.layout.reqToken*j)=(c.page b j : Int)

def source (c : Host.Call) : AlgorithmSource where
  kernel := (kernel c.layout c.regions .fp32 (Host.scale c.layout.D)).eraseDType
  projection := RealIO.projection c.layout c.regions .fp32 Bool.false (Host.scale c.layout.D)

def programIO (c : Host.Call) (capacity : RegionBounds) : AlgorithmProgramIO where
  nIn := 8
  nOut := 1
  nScr := 0
  bufs := c.regions.out :: c.inputs
  ity := inputType
  iarity := fun i=>capacity (inputRegion c i)
  ibuf := inputRegion c
  oty := fun _=>.float
  oarity := fun _=>laneCount c
  obuf := fun _=>c.regions.out
  obuf_mem := fun _=>List.mem_cons_self
  sarity := fun i=>Fin.elim0 i
  sbuf := fun i=>Fin.elim0 i
  iwin := fun _ _ _ _ _ j=>j.val
  imask := fun _ _ _ _ _ _=>True
  owin := fun _ _ _ _ _ j=>let k:=(lanes c).symm j; c.offset k.1 k.2
  omask := fun _ _ _ _ _ j=>let k:=(lanes c).symm j; c.active k.1 k.2
  swin := fun i=>Fin.elim0 i
  smask := fun i=>Fin.elim0 i
  initialUndef := .preserve
  stages := fun _=>[.ordinary (source c) c.grid]

/-- Independent gathered finite-sentinel weighted sum, with every padded
key contribution, original base-2 exponential and per-block range retained. -/
def formula (c : Host.Call) (capacity : RegionBounds) (values : Inputs c capacity) :
    Fin 1 → Fin (laneCount c) → ℝ := fun _ j=>
  let k:=(lanes c).symm j
  Formula.expected c.layout (c.launchAt k.1) (logicalInputs c capacity values)
    (Host.scale c.layout.D) k.2.1.val k.2.2.1.val

#specNonCircular formula avoiding [exec,execR,AlgorithmExecution,Kernel.GridLaunchedOrdinary]
#specNonCircular Geometry avoiding [exec,execR,AlgorithmExecution,Kernel.GridLaunchedOrdinary]
#specNonCircular Metadata avoiding [exec,execR,AlgorithmExecution,Kernel.GridLaunchedOrdinary]
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.RealInterface

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.RealInterface.Logical
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnLlama
noncomputable section

/-- Complete finite backing, including all signed metadata, aliases and
padding. Logical arrays below do not replace or restrict this storage. -/
structure Backing (c : Host.Call) (capacity : RegionBounds) where
  storage : RealInterface.Inputs c capacity

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
  request : Nat → Int
  pages : Nat → Int

def mathematicalArrays (p : Formula.Layout) (a : Formula.Inputs) : Arrays where
  query := fun launch row feature => Formula.query p launch a row feature
  key := fun launch row feature => Formula.key p launch a row feature
  value := fun launch row feature => Formula.value p launch a row feature
  blocks := fun launch => Formula.count p launch a
  start := a.start
  sequence := a.sequence
  prompt := a.prompt
  request := a.request
  pages := a.pages

def arraysOfBacking (c : Host.Call) (capacity : RegionBounds) (backing : Backing c capacity) : Arrays :=
  mathematicalArrays (c.layout) (logicalInputs c capacity backing.storage)

/-- A graph, rather than an additional consistency precondition: for every
old backing there is exactly one connected mathematical array record. -/
structure Inputs (c : Host.Call) (capacity : RegionBounds) where
  arrays : Arrays
  backing : Backing c capacity
  connection : arrays = arraysOfBacking c capacity backing

def inputs (c : Host.Call) (capacity : RegionBounds) : RealInterface.Inputs c capacity ≃ Inputs c capacity where
  toFun := fun raw => ⟨arraysOfBacking c capacity ⟨raw⟩,⟨raw⟩,rfl⟩
  invFun := fun values => values.backing.storage
  left_inv := fun _ => rfl
  right_inv := by
    rintro ⟨arrays,⟨storage⟩,connection⟩
    cases connection
    rfl

/-- Logical output coordinates enumerate precisely the original output cells. -/
def outputs (c : Host.Call) (capacity : RegionBounds) :
    (Fin 1 → Fin (laneCount c) → ℝ) ≃ (Lane c → ℝ) where
  toFun := fun values coordinate => values 0 ((lanes c) coordinate)
  invFun := fun values _ j => values ((lanes c).symm j)
  left_inv := by
    intro values
    funext channel j
    have channel_zero : channel = 0 := Fin.eq_zero channel
    subst channel
    exact congrArg (values 0) ((lanes c).apply_symm_apply j)
  right_inv := by
    intro values
    funext coordinate
    exact congrArg values ((lanes c).symm_apply_apply coordinate)

def domain (c : Host.Call) (capacity : RegionBounds) (values : Inputs c capacity) : Prop :=
  Metadata c capacity values.backing.storage

def io (c : Host.Call) (capacity : RegionBounds) :
    LogicalDomainIO (RealInterface.Inputs c capacity) (Fin 1 → Fin (laneCount c) → ℝ)
      (Inputs c capacity) (Lane c → ℝ) (programIO c capacity).ImplementsOn where
  inputs := inputs c capacity
  outputs := outputs c capacity

end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.RealInterface.Logical
