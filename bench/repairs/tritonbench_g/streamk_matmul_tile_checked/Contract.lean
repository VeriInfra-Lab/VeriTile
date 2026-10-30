import bench.repairs.tritonbench_g.streamk_matmul.Contract

/-! Independent candidate: repair the static source domain without changing any
kernel. Empty outputs instantiate no kernel. Runtime K=0 still instantiates the
dot body, so it does not remove the BLOCK_K lower bound. -/
namespace VeriTile.Bench.Repairs.StreamK.TileChecked.Contract
open VeriTile.Torch
abbrev Config := Checked.Contract.Config

/-- The three actual source tensor shapes: A offsets/loads, B offsets/loads,
and accumulator/output/scratch. These are frontend limits, not GPU resources. -/
def StaticTiles (c : Config) : Prop :=
  16 ≤ c.blockK ∧ c.blockM*c.blockK ≤ 2^20 ∧
  c.blockK*c.blockN ≤ 2^20 ∧ c.blockM*c.blockN ≤ 2^20
instance (c : Config) : Decidable (StaticTiles c) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _))

structure InputContract (a b : TensorMetadata) (c : Config) : Prop where
  prior : Checked.Contract.InputContract a b c
  static : Checked.Contract.Empty a b ∨ StaticTiles c

inductive Error where
  | prior (error : Checked.Contract.Error)
  | staticTiles
  deriving DecidableEq, Repr

def checkTensorInputs (a b : TensorMetadata) (c : Config) : Except Error Unit := do
  match Checked.Contract.checkTensorInputs a b c with
  | .error e => .error (.prior e)
  | .ok _ =>
    if Checked.Contract.Empty a b ∨ StaticTiles c then .ok () else .error .staticTiles

def checkInputs (a b : ArgumentMetadata) (c : Config) : Except Error Unit :=
  match a,b with
  | .tensor a,.tensor b => checkTensorInputs a b c
  | _,_ => .error (.prior .tensor)

def AdmittedArguments (a b : ArgumentMetadata) (c : Config) : Prop :=
  match a,b with
  | .tensor a,.tensor b => InputContract a b c
  | _,_ => False
end VeriTile.Bench.Repairs.StreamK.TileChecked.Contract
