import bench.repairs.tritonbench_g.context_attn_mistral.KernelOnly.Kernels
import bench.tritonbench_g.context_attn_mistral.Proofs.Program
import bench.tritonbench_g.context_attn_mistral.Proofs.HostReadback
import VeriTile.Triton.Memory.AlgorithmProgramIO
import VeriTile.Triton.Memory.LogicalIO

namespace VeriTile.Bench.Repairs.ContextAttnMistralKernel.KernelSource
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnMistral Formula

/-- Actual source parameters selected by the retained layout and region ABI. -/
def original (p : Layout) (r : Program.Regions) (scale : ℝ) : ComputeKernel :=
  kernel r.query r.key r.value scale r.start r.sequence r.out
    p.qRow p.qHead p.qDim p.kRow p.kHead p.kDim
    p.vRow p.vHead p.vDim p.outRow p.outHead p.outDim p.kvGroup p.window p.BM p.D p.BN

def candidate (p : Layout) (r : Program.Regions) (scale : ℝ) : ComputeKernel :=
  blockStartReused r.query r.key r.value scale r.start r.sequence r.out
    p.qRow p.qHead p.qDim p.kRow p.kHead p.kDim
    p.vRow p.vHead p.vDim p.outRow p.outHead p.outDim p.kvGroup p.window p.BM p.D p.BN

def source (reused : Bool) (p : Layout) (r : Program.Regions) (scale : ℝ) : ComputeKernel :=
  if reused then candidate p r scale else original p r scale

end VeriTile.Bench.Repairs.ContextAttnMistralKernel.KernelSource

/-! Public regional IO of the complete original ordinary grid. The existing
flat-allocation transport remains a separate physical-memory theorem. -/
namespace VeriTile.Bench.Repairs.ContextAttnMistralKernel.RealInterface
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
noncomputable section
open scoped Classical

/-- Original layout, capacity and output-ownership conditions, without any
operational state or execution premise. Input metadata is bound separately. -/
structure Geometry (c : Host.Call) (capacity : RegionBounds) : Prop where
  start_bound : c.batches ≤ capacity c.regions.start
  length_bound : c.batches ≤ capacity c.regions.sequence
  group_pos : 0 < c.layout.kvGroup
  grouped : c.heads = c.kvHeads*c.layout.kvGroup
  query : ∀ b, b < c.batches → ∀ h, h < c.heads → ∀ row, row < c.length b →
    ∀ d, d < c.layout.D →
      (c.start b+row)*c.layout.qRow+h*c.layout.qHead+d*c.layout.qDim < capacity c.regions.query
  key : ∀ b, b < c.batches → ∀ h, h < c.heads → ∀ row, row < c.length b →
    ∀ d, d < c.layout.D →
      (c.start b+row)*c.layout.kRow+(h/c.layout.kvGroup)*c.layout.kHead+d*c.layout.kDim < capacity c.regions.key
  value : ∀ b, b < c.batches → ∀ h, h < c.heads → ∀ row, row < c.length b →
    ∀ d, d < c.layout.D →
      (c.start b+row)*c.layout.vRow+(h/c.layout.kvGroup)*c.layout.vHead+d*c.layout.vDim < capacity c.regions.value
  out : ∀ b, b < c.batches → ∀ h, h < c.heads → ∀ row, row < c.length b →
    ∀ d, d < c.layout.D → c.output b h row d < capacity c.regions.out
  packed : ∀ b b' row row', b < c.batches → b' < c.batches →
    row < c.length b → row' < c.length b' →
    c.start b+row = c.start b'+row' → b=b' ∧ row=row'
  output_view : ∀ b b' h h' row row' d d', b < c.batches → b' < c.batches →
    h < c.heads → h' < c.heads → row < c.length b → row' < c.length b' →
    d < c.layout.D → d' < c.layout.D →
    c.output b h row d = c.output b' h' row' d' →
    c.start b+row = c.start b'+row' ∧ h=h' ∧ d=d'
  input_separate : ∀ rg, rg ∈ [c.regions.query,c.regions.key,c.regions.value,
    c.regions.start,c.regions.sequence] → c.regions.out ≠ rg

def inputRegion (c : Host.Call) (i : Fin 5) : RegionName :=
  match i with
  | ⟨0,_⟩ => c.regions.query | ⟨1,_⟩ => c.regions.key | ⟨2,_⟩ => c.regions.value
  | ⟨3,_⟩ => c.regions.start | ⟨4,_⟩ => c.regions.sequence

def inputType (i : Fin 5) : ChanTy := if i.val<3 then .float else .int
abbrev Inputs (c : Host.Call) (capacity : RegionBounds) :=
  (i : Fin 5) → Fin (capacity (inputRegion c i)) → (inputType i).carrier
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

/-- The original metadata values, on the admitted finite arrays only. -/
structure Metadata (c : Host.Call) (capacity : RegionBounds) (values : Inputs c capacity) : Prop where
  start : ∀ b,b<c.batches → (logicalInputs c capacity values).start b=(c.start b : Int)
  sequence : ∀ b,b<c.batches → (logicalInputs c capacity values).sequence b=((c.length b : Nat) : Int)

def source (c : Host.Call) : AlgorithmSource where
  kernel := (KernelSource.original c.layout c.regions (Host.scale c.layout.D)).eraseDType
  projection := RealIO.projection c.layout c.regions .fp32 Bool.false (Host.scale c.layout.D)

def programIO (c : Host.Call) (capacity : RegionBounds) : AlgorithmProgramIO where
  nIn := 5
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

end
end VeriTile.Bench.Repairs.ContextAttnMistralKernel.RealInterface

namespace VeriTile.Bench.Repairs.ContextAttnMistralKernel.LogicalData
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
noncomputable section

/-- Complete storage is retained, including aliases, padding and unused cells. -/
structure Backing (c : Host.Call) (capacity : RegionBounds) where
  query : Fin (capacity c.regions.query) → ℝ
  key : Fin (capacity c.regions.key) → ℝ
  value : Fin (capacity c.regions.value) → ℝ
  starts : Fin (capacity c.regions.start) → Int
  lengths : Fin (capacity c.regions.sequence) → Int

def storageEquiv (c : Host.Call) (capacity : RegionBounds) :
    RealInterface.Inputs c capacity ≃ Backing c capacity where
  toFun := fun xs => ⟨xs 0,xs 1,xs 2,xs 3,xs 4⟩
  invFun := fun xs i => match i with
    | ⟨0,_⟩ => xs.query | ⟨1,_⟩ => xs.key | ⟨2,_⟩ => xs.value
    | ⟨3,_⟩ => xs.starts | ⟨4,_⟩ => xs.lengths
  left_inv := by intro xs; funext i; fin_cases i <;> rfl
  right_inv := by intro xs; cases xs; rfl

def storageInputs (xs : Backing c capacity) : Formula.Inputs where
  q := RealInterface.array xs.query
  k := RealInterface.array xs.key
  v := RealInterface.array xs.value
  start := RealInterface.array xs.starts
  sequence := RealInterface.array xs.lengths

/-- Mathematical axes are batch, query/KV head, sequence time and dimension.
The exact signed metadata, masks and stride aliases determine the tensor view. -/
def queryView (c : Host.Call) (xs : Backing c capacity) (batch head time dim : Nat) : ℝ :=
  if (time : Int) < RealInterface.array xs.lengths batch then
    RealInterface.array xs.query
      ((RealInterface.array xs.starts batch+time)*c.layout.qRow+head*c.layout.qHead+dim*c.layout.qDim).toNat
  else 0

def keyView (c : Host.Call) (xs : Backing c capacity) (batch kvHead time dim : Nat) : ℝ :=
  if (time : Int) < RealInterface.array xs.lengths batch then
    RealInterface.array xs.key
      ((RealInterface.array xs.starts batch+time)*c.layout.kRow+kvHead*c.layout.kHead+dim*c.layout.kDim).toNat
  else 0

def valueView (c : Host.Call) (xs : Backing c capacity) (batch kvHead time dim : Nat) : ℝ :=
  if (time : Int) < RealInterface.array xs.lengths batch then
    RealInterface.array xs.value
      ((RealInterface.array xs.starts batch+time)*c.layout.vRow+kvHead*c.layout.vHead+dim*c.layout.vDim).toNat
  else 0

/-- The logical tensors are uniquely determined by complete storage. Consistency
is part of this equivalent representation, not an additional input-domain guard. -/
structure Inputs (c : Host.Call) (capacity : RegionBounds) where
  backing : Backing c capacity
  query : Nat → Nat → Nat → Nat → ℝ
  key : Nat → Nat → Nat → Nat → ℝ
  value : Nat → Nat → Nat → Nat → ℝ
  query_eq : query = queryView c backing
  key_eq : key = keyView c backing
  value_eq : value = valueView c backing

def inputEquiv (c : Host.Call) (capacity : RegionBounds) :
    RealInterface.Inputs c capacity ≃ Inputs c capacity where
  toFun := fun xs =>
    let b := storageEquiv c capacity xs
    ⟨b,queryView c b,keyView c b,valueView c b,rfl,rfl,rfl⟩
  invFun := fun xs => (storageEquiv c capacity).symm xs.backing
  left_inv := by intro xs; exact (storageEquiv c capacity).left_inv xs
  right_inv := by
    rintro ⟨backing,query,key,value,hq,hk,hv⟩
    cases hq
    cases hk
    cases hv
    simp only [Equiv.apply_symm_apply]

def sequenceLength (xs : Inputs c capacity) (batch : Nat) : Int :=
  RealInterface.array xs.backing.lengths batch

/-- Every emitted grid/lane cell, including inactive and underlaunched rows. -/
structure Coordinate (c : Host.Call) where
  batch : Fin c.batches
  head : Fin c.heads
  block : Fin c.blocks
  row : Fin c.layout.BM
  dim : Fin c.layout.D

def coordinateEquiv (c : Host.Call) : RealInterface.Lane c ≃ Coordinate c where
  toFun := fun lane =>
    ⟨⟨(c.launchAt lane.1).batch,Host.batch_bound c lane.1⟩,
     ⟨(c.launchAt lane.1).head,Host.head_bound c lane.1⟩,
     ⟨(c.launchAt lane.1).block,Host.block_bound c lane.1⟩,lane.2.1,lane.2.2.1⟩
  invFun := fun j => (c.indexOf j.batch j.head j.block,j.row,j.dim,())
  left_inv := by
    rintro ⟨idx,row,dim,u⟩
    cases u
    dsimp only
    apply Prod.ext
    · apply Host.launchAt_injective c
      all_goals simp only [Host.launch_indexOf]
    · rfl
  right_inv := by
    rintro ⟨batch,head,block,row,dim⟩
    simp only [Host.launch_indexOf]

def outputEquiv (c : Host.Call) : (Fin 1 → Fin (RealInterface.laneCount c) → ℝ) ≃
    (Coordinate c → ℝ) where
  toFun := fun ys j => ys 0 (RealInterface.lanes c ((coordinateEquiv c).symm j))
  invFun := fun ys _ j => ys (coordinateEquiv c ((RealInterface.lanes c).symm j))
  left_inv := by
    intro ys
    funext i j
    fin_cases i
    simp
  right_inv := by intro ys; funext j; simp

def gatheredInputs (c : Host.Call) (capacity : RegionBounds) (xs : Inputs c capacity) :
    Formula.Inputs := storageInputs xs.backing

def Metadata (c : Host.Call) (capacity : RegionBounds) (xs : Inputs c capacity) : Prop :=
  RealInterface.Metadata c capacity ((inputEquiv c capacity).symm xs)

def attentionIO (c : Host.Call) (capacity : RegionBounds) :
    LogicalDomainIO (RealInterface.Inputs c capacity)
      (Fin 1 → Fin (RealInterface.laneCount c) → ℝ)
      (Inputs c capacity) (Coordinate c → ℝ)
      (RealInterface.programIO c capacity).ImplementsOn where
  inputs := inputEquiv c capacity
  outputs := outputEquiv c

end
end VeriTile.Bench.Repairs.ContextAttnMistralKernel.LogicalData
