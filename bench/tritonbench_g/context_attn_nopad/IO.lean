import bench.tritonbench_g.context_attn_nopad.FPReferenceGrid
import VeriTile.Triton.Float.ObservedProgramIO
import VeriTile.Triton.Memory.AlgorithmProgramIO
import bench.tritonbench_g.context_attn_nopad.Proofs.Mathematical_HostCoverage
import VeriTile.Triton.Memory.LogicalIO

/-! Canonical typed/native and flattened Real IO for the selected kernel.
The different carriers and observers are intentional; source addresses and
active coordinates are shared. Connection and safety proofs stay in Proofs. -/

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.IOData
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
open Classical
noncomputable section
variable {α : Type}

abbrev timeExtent (c : Call) := c.blocks*128
abbrev Coordinate (c : Call) := TileIndex [c.batches,c.q.heads,timeExtent c,c.q.dims]
def active (c : Call) (j : Coordinate c) : Prop := j.2.2.1.val<c.length j.1.val

def address (c : Call) (t : Tensor3) (j : Coordinate c) : Nat :=
  t.offset (c.start j.1.val+j.2.2.1.val) j.2.1.val j.2.2.2.1.val

abbrev inputs (c : Call) (r : Regions) : List IOChannel :=
  [⟨.real,[c.batches,c.q.heads,timeExtent c,c.q.dims],r.query⟩,
   ⟨.real,[c.batches,c.q.heads,timeExtent c,c.q.dims],r.key⟩,
   ⟨.real,[c.batches,c.q.heads,timeExtent c,c.q.dims],r.value⟩,
   ⟨.int,[c.batches],r.start⟩,⟨.int,[c.batches],r.sequence⟩]
abbrev outputs (c : Call) (r : Regions) : List IOChannel :=
  [⟨.real,[c.batches,c.q.heads,timeExtent c,c.q.dims],r.out⟩]

def read (c : Call) (r : Regions) (i : Fin (inputs c r).length) :
    TileIndex (inputs c r)[i].shape → Nat :=
  match i with
  | ⟨0,_⟩ => address c c.q
  | ⟨1,_⟩ => address c c.k
  | ⟨2,_⟩ => address c c.v
  | ⟨3,_⟩ | ⟨4,_⟩ => fun j => j.1.val

def inputActive (c : Call) (r : Regions) (i : Fin (inputs c r).length) :
    TileIndex (inputs c r)[i].shape → Prop :=
  match i with
  | ⟨0,_⟩ | ⟨1,_⟩ | ⟨2,_⟩ => active c
  | ⟨3,_⟩ | ⟨4,_⟩ => fun _ => True

def job (c : Call) (r : Regions) (o : FPBackend.Ops α) (idx : GridIndex c.grid) : Invocation α where
  kernel := kernel c r
  backend := FP.ProfiledRewrite.engine o.algebra
  pids := idx.toPids
  numPids := some c.grid.toNumPids

abbrev io (c : Call) (r : Regions) (o : FPBackend.Ops α) : ProgramIO α α where
  inputs := inputs c r
  outputs := outputs c r
  scratch := []
  buffers := [r.query,r.key,r.value,r.start,r.sequence,r.out]
  read := fun i _ => read c r i
  inputMask := fun i _ => inputActive c r i
  write := fun i _ => match i with | ⟨0,_⟩ => address c c.o
  outputMask := fun i _ => match i with | ⟨0,_⟩ => active c
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i
  schedule := fun _ => (FPHost.jobs c).map (job c r o)
  encode := id
  observe := fun _ _ => id

/-- The original fixed signed metadata observations, with no source-result
condition. This leaves incomplete launch grids and floating-buffer aliases free. -/
def domain (c : Call) (r : Regions) (xs : ChannelValues α (inputs c r)) : Prop :=
  (∀ b : Fin c.batches,xs 3 (b,PUnit.unit)=(c.start b.val : Int)) ∧
  (∀ b : Fin c.batches,xs 4 (b,PUnit.unit)=(c.length b.val : Int))

/-- Only actual live Q/K/V coordinates in the selected finite grid. -/
def ReadHit (c : Call) (r : Regions) (region : RegionName) (offset : Nat) : Prop :=
  ∃ j : Coordinate c,active c j ∧
    ((region=r.query ∧ offset=address c c.q j) ∨
     (region=r.key ∧ offset=address c c.k j) ∨
     (region=r.value ∧ offset=address c c.v j))

/-- Logical finite input lookup. Simultaneous input binding ensures aliases
supply the same payload regardless of which matching channel is selected. -/
def initialMemory [Inhabited α] (c : Call) (r : Regions)
    (xs : ChannelValues α (inputs c r)) (region : RegionName) (offset : Nat) : α :=
  if h : ReadHit c r region offset then
    let j := Classical.choose h
    if region=r.query ∧ offset=address c c.q j then xs 0 j
    else if region=r.key ∧ offset=address c c.k j then xs 1 j else xs 2 j
  else default
end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.IOData

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.RealData
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
open Classical
noncomputable section

abbrev Coordinate (c : Call) := IOData.Coordinate c
abbrev laneCount (c : Call) := Fintype.card (Coordinate c)
def decode (c : Call) : Fin (laneCount c) → Coordinate c := (Fintype.equivFin (Coordinate c)).symm
def encode (c : Call) : Coordinate c → Fin (laneCount c) := Fintype.equivFin (Coordinate c)

abbrev ity : Fin 5 → ChanTy
  | ⟨0,_⟩ | ⟨1,_⟩ | ⟨2,_⟩ => .float
  | ⟨3,_⟩ | ⟨4,_⟩ => .int
abbrev arity (c : Call) : Fin 5 → Nat
  | ⟨0,_⟩ | ⟨1,_⟩ | ⟨2,_⟩ => laneCount c
  | ⟨3,_⟩ | ⟨4,_⟩ => c.batches
abbrev Values (c : Call) := (i : Fin 5) → Fin (arity c i) → (ity i).carrier

def inputRegion (r : Regions) : Fin 5 → RegionName
  | ⟨0,_⟩ => r.query | ⟨1,_⟩ => r.key | ⟨2,_⟩ => r.value
  | ⟨3,_⟩ => r.start | ⟨4,_⟩ => r.sequence

def read (c : Call) (i : Fin 5) : Fin (arity c i) → Nat :=
  match i with
  | ⟨0,_⟩ => fun j => IOData.address c c.q (decode c j)
  | ⟨1,_⟩ => fun j => IOData.address c c.k (decode c j)
  | ⟨2,_⟩ => fun j => IOData.address c c.v (decode c j)
  | ⟨3,_⟩ | ⟨4,_⟩ => fun j => j.val

def inputActive (c : Call) (i : Fin 5) : Fin (arity c i) → Prop :=
  match i with
  | ⟨0,_⟩ | ⟨1,_⟩ | ⟨2,_⟩ => fun j => IOData.active c (decode c j)
  | ⟨3,_⟩ | ⟨4,_⟩ => fun _ => True

abbrev io (c : Call) (r : Regions) : AlgorithmProgramIO where
  nIn := 5
  nOut := 1
  nScr := 0
  bufs := [r.query,r.key,r.value,r.start,r.sequence,r.out]
  ity := ity
  iarity := arity c
  ibuf := inputRegion r
  oty := fun _ => .float
  oarity := fun _ => laneCount c
  obuf := fun _ => r.out
  obuf_mem := by intro o; simp
  sarity := fun i => Fin.elim0 i
  sbuf := fun i => Fin.elim0 i
  iwin := fun i _ _ _ _ => read c i
  imask := fun i _ _ _ _ => inputActive c i
  owin := fun _ _ _ _ _ j => IOData.address c c.o (decode c j)
  omask := fun _ _ _ _ _ j => IOData.active c (decode c j)
  swin := fun i => Fin.elim0 i
  smask := fun i => Fin.elim0 i
  stages := fun _ => [.ordinary
    ⟨(kernel c r).eraseDType,RealIO.projection c.layout r .fp32 Bool.false c.scale⟩ c.grid]

def domain (c : Call) (xs : Values c) : Prop :=
  (∀ b : Fin c.batches,xs 3 b=(c.start b.val : Int)) ∧
  (∀ b : Fin c.batches,xs 4 b=(c.length b.val : Int))

def Labels (r : Regions) : Prop :=
  r.out≠r.query ∧ r.out≠r.key ∧ r.out≠r.value ∧ r.out≠r.start ∧ r.out≠r.sequence
end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.RealData

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.LogicalData
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
noncomputable section

structure Inputs (c : Call) where
  query : RealData.Coordinate c → ℝ
  key : RealData.Coordinate c → ℝ
  value : RealData.Coordinate c → ℝ
  starts : Fin c.batches → Int
  lengths : Fin c.batches → Int

def inputEquiv (c : Call) : RealData.Values c ≃ Inputs c where
  toFun := fun xs => ⟨fun j => xs 0 (RealData.encode c j),
    fun j => xs 1 (RealData.encode c j), fun j => xs 2 (RealData.encode c j), xs 3, xs 4⟩
  invFun := fun xs i => match i with
    | ⟨0,_⟩ => fun j => xs.query (RealData.decode c j)
    | ⟨1,_⟩ => fun j => xs.key (RealData.decode c j)
    | ⟨2,_⟩ => fun j => xs.value (RealData.decode c j)
    | ⟨3,_⟩ => xs.starts
    | ⟨4,_⟩ => xs.lengths
  left_inv := by
    intro xs
    funext i j
    fin_cases i <;> simp [RealData.encode,RealData.decode]
  right_inv := by
    intro xs
    cases xs
    simp [RealData.encode,RealData.decode]

def outputEquiv (c : Call) : (Fin 1 → Fin (RealData.laneCount c) → ℝ) ≃
    (RealData.Coordinate c → ℝ) where
  toFun := fun ys j => ys 0 (RealData.encode c j)
  invFun := fun ys _ j => ys (RealData.decode c j)
  left_inv := by
    intro ys
    funext i j
    fin_cases i
    simp [RealData.encode,RealData.decode]
  right_inv := by intro ys; funext j; simp [RealData.encode,RealData.decode]

def Metadata (c : Call) (xs : Inputs c) : Prop :=
  (∀ b, xs.starts b = (c.start b.val : Int)) ∧
  (∀ b, xs.lengths b = (c.length b.val : Int))

/-- The full original algorithm judgment, including bounds, masks and frame. -/
def attentionIO (c : Call) (r : Regions) :
    LogicalDomainIO (RealData.Values c) (Fin 1 → Fin (RealData.laneCount c) → ℝ)
      (Inputs c) (RealData.Coordinate c → ℝ) (RealData.io c r).ImplementsOn where
  inputs := inputEquiv c
  outputs := outputEquiv c

end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.LogicalData
