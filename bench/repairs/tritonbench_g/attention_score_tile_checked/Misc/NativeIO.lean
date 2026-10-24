import bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.NativeReference
import VeriTile.Triton.Float.ObservedInputCellsIO

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.NativeInterface
open VeriTile Triton FP.Structural Reused Geometry NativeInstructions NativeStaging
noncomputable section

def inputPorts (c : Shape) : List IOChannel :=
  [⟨.real,[c.querySize],"q"⟩,⟨.real,[c.keySize],"k"⟩,
   ⟨.real,[c.normalizerSize],"m"⟩,⟨.real,[c.outputSize],"out"⟩]
def outputPorts (c : Shape) : List IOChannel := [⟨.real,[c.outputSize],"out"⟩]

abbrev Inputs (c : Shape) := ChannelValues Graph (inputPorts c)

def flatInputs (c : Shape) (xs : Inputs c) : RealInterface.Inputs Graph c
  | ⟨0,_⟩,j => xs (0 : Fin 4) (j,())
  | ⟨1,_⟩,j => xs (1 : Fin 4) (j,())
  | ⟨2,_⟩,j => xs (2 : Fin 4) (j,())
  | ⟨3,_⟩,j => xs (3 : Fin 4) (j,())

/-- The original empty allocator supplies literal zero before any source runs.
Nonempty source inputs and outputs retain the staging model's ready graphs. -/
def encode (c : Shape) (g : Graph) : NativeStaging.Expr :=
  if RealInterface.empty c then
    match g with
    | .literal32 value => .literal value
    | _ => .ready g
  else .ready g

theorem lower_encode (c : Shape) (g : Graph) : lower (encode c g)=g := by
  unfold encode
  split_ifs
  · cases g <;> rfl
  · rfl

def invocation (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool)
    (job : WholeGrid.Job c) : Invocation NativeStaging.Expr where
  kernel := kernel c offset size scale window complement
  backend := FP.ProfiledRewrite.engine (FPEngine.withDot algebra dotModel)
  pids := (RealInterface.launch c job).pids

def programIO (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool) :
    ProgramIO Graph NativeStaging.Expr where
  inputs := inputPorts c
  outputs := outputPorts c
  scratch := []
  buffers := ["q","k","m","out"]
  read := fun i _ => match i with
    | ⟨0,_⟩ => fun j => j.1.val
    | ⟨1,_⟩ => fun j => j.1.val
    | ⟨2,_⟩ => fun j => j.1.val
    | ⟨3,_⟩ => fun j => j.1.val
  inputMask := fun i _ _ => i.val≠3 ∨ RealInterface.empty c
  write := fun i _ => match i with | ⟨0,_⟩ => fun j => j.1.val
  outputMask := fun _ _ _ => True
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i
  schedule := fun _ => if RealInterface.empty c then []
    else (WholeGrid.jobs c).map (invocation c offset size scale window complement)
  encode := encode c
  observe := fun _ _ => lower

def initialOutput (c : Shape) (xs : Inputs c) : Prop :=
  RealInterface.empty c → ∀ j, xs (3 : Fin 4) (j,())=NativeFormula.zero32

def formula (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool)
    (xs : Inputs c) : ChannelValues Graph (outputPorts c) := by
  intro o
  change Fin 1 at o
  have ho : o=(0 : Fin 1) := Subsingleton.elim _ _
  subst o
  exact fun j =>
    let coord := RealInterface.outputCoordinates c j.1
    NativeReference.column c (RealInterface.logicalInputs c (flatInputs c xs)) coord.1 coord.2.1
      offset size scale window complement coord.2.2

def observe (I : ISA (BitVec 32)) : Graph → BitVec 32 := Graph.eval { I with bits:=id }

#specNonCircular formula avoiding [kernel,nativeExecute,FP.Structural.State,FP.Structural.exec,FPGrid.run]
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.NativeInterface
