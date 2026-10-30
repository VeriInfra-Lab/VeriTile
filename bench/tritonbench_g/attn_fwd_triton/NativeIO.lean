import bench.tritonbench_g.attn_fwd_triton.Proofs.Execution
import bench.tritonbench_g.attn_fwd_triton.Proofs.HostBlock
import VeriTile.Triton.Float.ProgramIO

/-! Complete selected attention grid. All three tensor inputs and both scale
vectors are FP32 in the selected source; Structural compute loads use its
erased Real cell channel. The score dot retains TF32, the value dot FP16,
and the returned channel BF16. Read roles may alias. -/
namespace VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.NativeInterface
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open AttnFwdTriton.TypedExecution Graph
noncomputable section
open scoped Classical

def tensorShape (p : Config) := [p.batches,p.heads,p.context,p.headDim]
def queryScales (p : Config) := p.batches*p.heads*((p.context+127)/128)
def keyScales (p : Config) := p.batches*p.heads*((p.context+63)/64)

def inputs (p : Config) : List IOChannel :=
  [⟨.real,tensorShape p,p.Q⟩,⟨.real,tensorShape p,p.K⟩,
    ⟨.real,tensorShape p,p.V⟩,⟨.real,[queryScales p],p.QS⟩,
    ⟨.real,[keyScales p],p.KS⟩]
def outputs (p : Config) : List IOChannel := [⟨.bf16,tensorShape p,p.Out⟩]
abbrev Inputs (p : Config) := ChannelValues Expr (inputs p)
abbrev Outputs (p : Config) := ChannelValues Expr (outputs p)

def coordinate (p : Config) (i : TileIndex (tensorShape p)) : Coordinate p :=
  (i.1,i.2.1,i.2.2.1,i.2.2.2.1)
def index (p : Config) (c : Coordinate p) : TileIndex (tensorShape p) :=
  (c.1,c.2.1,c.2.2.1,c.2.2.2,PUnit.unit)

def read (p : Config) (i : Fin (inputs p).length) : TileIndex (inputs p)[i].shape → Nat :=
  match i with
  | ⟨0,_⟩ => fun j => j.1.val*p.queryBatch+j.2.1.val*p.queryHead+
      j.2.2.1.val*p.queryRow+j.2.2.2.1.val*p.queryDim
  | ⟨1,_⟩ => fun j => j.1.val*p.keyBatch+j.2.1.val*p.keyHead+
      j.2.2.1.val*p.keyRow+j.2.2.2.1.val*p.keyDim
  | ⟨2,_⟩ => fun j => j.1.val*p.valueBatch+j.2.1.val*p.valueHead+
      j.2.2.1.val*p.valueRow+j.2.2.2.1.val*p.valueDim
  | ⟨3,_⟩ => fun j => j.1.val
  | ⟨4,_⟩ => fun j => j.1.val
  | ⟨n+5,h⟩ => False.elim (by simp [inputs] at h)
def write (p : Config) (o : Fin (outputs p).length) : TileIndex (outputs p)[o].shape → Nat :=
  match o with
  | ⟨0,_⟩ => fun j => logicalAddress p (coordinate p j)
  | ⟨n+1,h⟩ => False.elim (by simp [outputs] at h)

def memory (p : Config) : MemoryIO Expr where
  inputs := inputs p
  outputs := outputs p
  scratch := []
  buffers := [p.Q,p.K,p.V,p.QS,p.KS,p.Out]
  read := fun i _ => read p i
  inputMask := fun _ _ _ => True
  write := fun o _ => write p o
  outputMask := fun _ _ _ => True
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i

def invocation (p : Config) (idx : ProgramIndex p) : Invocation Expr where
  kernel := source p (AttnFwdTriton.HostShape.blockD p.headDim) Bool.false
  backend := engine backend
  pids := fun ax => if ax=0 then idx.1.val else if ax=1 then idx.2.val else 0
  numPids := some (fun ax => if ax=0 then rowCount p else if ax=1 then p.batches*p.heads else 1)

def fullGrid (p : Config) : List (ProgramIndex p) := Finset.univ.toList

def programIO (p : Config) : ProgramIO Expr Expr :=
  {memory p with
    schedule := fun _ => (fullGrid p).map (invocation p)
    encode := id
    observe := fun _ _ => id}

/-- A logical input location. Multiple aliases are bound simultaneously by IO;
the proof below therefore may choose any matching logical input coordinate. -/
def InputLocation (p : Config) (region : RegionName) (offset : Nat) :=
  {i : Fin (inputs p).length // ∃ j : TileIndex (inputs p)[i].shape,
    (inputs p)[i].buf=region ∧ read p i j=offset}

def payload (p : Config) (values : Inputs p)
    (i : Fin (inputs p).length) (j : TileIndex (inputs p)[i].shape) : Expr :=
  match i with
  | ⟨0,_⟩ | ⟨1,_⟩ | ⟨2,_⟩ | ⟨3,_⟩ | ⟨4,_⟩ => values _ j
  | ⟨n+5,h⟩ => False.elim (by simp [inputs] at h)

/-- Pure lookup in the declared logical input arrays. It neither executes the
source nor reads a runtime State; unused addresses receive a neutral leaf. -/
def logicalMemory (p : Config) (values : Inputs p) (region : RegionName) (offset : Nat) : Cell Expr :=
  .mk .real (if h : Nonempty (InputLocation p region offset) then
    let i := Classical.choice h
    payload p values i.val (Classical.choose i.property)
  else .zero)

/-- Existing independent ordered two-loop recurrence on logical arrays. -/
def formula (p : Config) (values : Inputs p) : Outputs p := fun o =>
  match o with
  | ⟨0,_⟩ => fun j =>
      let c := coordinate p j
      Model.output (Model.result p (AttnFwdTriton.HostShape.blockD p.headDim)
        (logicalMemory p values) (owner p c).1.val (owner p c).2.val)
        (lane p _ (AttnFwdTriton.HostShape.blockD_covers _) c)
  | ⟨n+1,h⟩ => False.elim (by simp [outputs] at h)

#specNonCircular formula avoiding [FP.Structural.exec,FP.Structural.run,runSchedule,
  AttnFwdTriton.TypedExecution.reference]
end
end VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.NativeInterface
