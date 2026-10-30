import VeriTile.Triton.Float.LogicalProgramIO
import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.KernelInterface.GridIO
import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.Proofs.NativeProfile
import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed.Proofs.RealForwardFlat
import VeriTile.Triton.Float.AddressedInputsIO

/-! Finite whole-tensor ports and actual source schedules. The canonical sources
retain FP32 operations and storage; their Structural logical cells use `.real`.
The mathematical forward carrier includes finite values and negative infinity.
Native inputs are fixed stored FP32 words, labelled by their read addresses;
output observation selects the actual consumer's scalar reduction schedule. -/
namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Interface
open VeriTile Triton FP.Structural
open SoftmaxFlaggemsFiniteSeed
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
noncomputable section

def count (d : Dimensions) : Nat := d.M*d.N*d.K

def channels (regions : List RegionName) (n : Nat) : List IOChannel :=
  regions.map (fun r => ⟨.real,[n],r⟩)

def index (regions : List RegionName) (n : Nat) :
    (i : Fin (channels regions n).length) → TileIndex (channels regions n)[i].shape → Fin n :=
  match regions with
  | [] => fun i => i.elim0
  | _::rest => Fin.cases (fun j => j.1) (index rest n)

def layout (α : Type) (regions : List RegionName) (Y : RegionName) (n : Nat) : MemoryIO α where
  inputs := channels regions n
  outputs := [⟨.real,[n],Y⟩]
  scratch := []
  buffers := regions++[Y]
  read := fun i _ j => (index regions n i j).val
  inputMask := fun _ _ _ => True
  write := Fin.cases (fun _ j => j.1.val) (fun i => i.elim0)
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k

def forwardIO (X Y : RegionName) (d : Dimensions) (c : LaunchConfig) : ProgramIO (WithBot ℝ) (WithBot ℝ) where
  toMemoryIO := layout (WithBot ℝ) [X] Y (count d)
  schedule := fun _ => forwardJobs Ideal.algebra X Y d c
  encode := id
  observe := fun _ _ => id

def backwardIO (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig) : ProgramIO ℝ (WithBot ℝ) where
  toMemoryIO := layout ℝ [P,G] Y (count d)
  schedule := fun _ => backwardJobs Ideal.algebra P G Y d c
  encode := fun x => (x : WithBot ℝ)
  observe := fun _ _ x => x.unbotD 0

/-- Total mathematical extension only; no backing outside the finite tensor is
requested. The formulas use it solely at the tensor's valid coordinates. -/
def extend {α : Type} [Inhabited α] {n : Nat} (xs : Fin n → α) (k : Nat) : α :=
  if h : k<n then xs ⟨k,h⟩ else default

def forwardInput (X Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (v : (forwardIO X Y d c).Inputs) (k : Fin (count d)) : WithBot ℝ :=
  v (0 : Fin 1) (k,PUnit.unit)

def backwardInput (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (v : (backwardIO P G Y d c).Inputs) (role : Fin 2) (k : Fin (count d)) : ℝ :=
  if role.val=0 then v (0 : Fin 2) (k,PUnit.unit) else v (1 : Fin 2) (k,PUnit.unit)

/-- Each normalized column contains a finite value. Other entries may be -∞;
this mathematical condition is not a new Python guard or numerical hypothesis. -/
def ForwardDomain (X Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (v : (forwardIO X Y d c).Inputs) : Prop :=
  RealForward.Defined (extend (forwardInput X Y d c v)) d

def forwardFormula (X Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (v : (forwardIO X Y d c).Inputs) : (forwardIO X Y d c).Outputs :=
  Fin.cases (fun k => (Target.flatForward (extend (forwardInput X Y d c v)) d.N d.K k.1.val : WithBot ℝ))
    (fun i => i.elim0)

def backwardFormula (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (v : (backwardIO P G Y d c).Inputs) : (backwardIO P G Y d c).Outputs :=
  Fin.cases (fun k => Target.flatBackward (extend (backwardInput P G Y d c v 0))
    (extend (backwardInput P G Y d c v 1)) d.N d.K k.1.val) (fun i => i.elim0)

/-- Logical M × N × K coordinates, including empty dimensions. -/
abbrev Tensor (α : Type) (d : Dimensions) := Fin d.M → Fin d.N → Fin d.K → α

def tensorIndex (d : Dimensions) : ((Fin d.M × Fin d.N) × Fin d.K) ≃ Fin (count d) :=
  (Equiv.prodCongr finProdFinEquiv (Equiv.refl _)).trans finProdFinEquiv

/-- Exact coordinate equivalence for a single tensor channel. No values,
including empty/tail cases, are removed by this view. -/
def tensorValues (α : Type) (d : Dimensions) (region : RegionName) :
    ChannelValues α [⟨.real,[count d],region⟩] ≃ Tensor α d where
  toFun := fun values row n col => values (0 : Fin 1) (tensorIndex d ((row,n),col),())
  invFun := fun xs => Fin.cases (fun j =>
    let coord := (tensorIndex d).symm j.1
    xs coord.1.1 coord.1.2 coord.2) (fun i => i.elim0)
  left_inv := by
    intro values
    funext channel j
    have zero : channel=0 := Fin.eq_zero channel
    subst channel
    change values 0 (tensorIndex d ((tensorIndex d).symm j.1),()) = values 0 j
    rw [(tensorIndex d).apply_symm_apply]
    exact congrArg (values 0) (Prod.ext rfl (Subsingleton.elim _ _))
  right_inv := by
    intro xs
    funext row n col
    change xs ((tensorIndex d).symm (tensorIndex d ((row,n),col))).1.1
      ((tensorIndex d).symm (tensorIndex d ((row,n),col))).1.2
      ((tensorIndex d).symm (tensorIndex d ((row,n),col))).2 = _
    rw [(tensorIndex d).symm_apply_apply]

/-- Saved output and incoming gradient are distinct logical tensors. -/
structure BackwardInputs (d : Dimensions) where
  saved : Tensor ℝ d
  gradient : Tensor ℝ d

def backwardValues (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig) :
    (backwardIO P G Y d c).Inputs ≃ BackwardInputs d where
  toFun := fun values => {
    saved := fun row n col => values (0 : Fin 2) (tensorIndex d ((row,n),col),())
    gradient := fun row n col => values (1 : Fin 2) (tensorIndex d ((row,n),col),()) }
  invFun := fun xs => Fin.cases
    (fun j => let coord := (tensorIndex d).symm j.1
      xs.saved coord.1.1 coord.1.2 coord.2)
    (Fin.cases (fun j => let coord := (tensorIndex d).symm j.1
      xs.gradient coord.1.1 coord.1.2 coord.2) (fun i => i.elim0))
  left_inv := by
    intro values
    funext channel j
    fin_cases channel <;>
      change values _ (tensorIndex d ((tensorIndex d).symm j.1),()) = values _ j <;>
      rw [(tensorIndex d).apply_symm_apply] <;>
      exact congrArg (values _) (Prod.ext rfl (Subsingleton.elim _ _))
  right_inv := by
    intro xs
    cases xs with
    | mk saved gradient =>
      apply congrArg₂ BackwardInputs.mk
      · funext row n col
        exact congrArg (fun coord => saved coord.1.1 coord.1.2 coord.2)
          ((tensorIndex d).symm_apply_apply ((row,n),col))
      · funext row n col
        exact congrArg (fun coord => gradient coord.1.1 coord.1.2 coord.2)
          ((tensorIndex d).symm_apply_apply ((row,n),col))


def logicalForwardIO (X Y : RegionName) (d : Dimensions) (c : LaunchConfig) :
    LogicalProgramIO (WithBot ℝ) (WithBot ℝ) (Tensor (WithBot ℝ) d) (Tensor (WithBot ℝ) d) where
  core := forwardIO X Y d c
  inputs := tensorValues (WithBot ℝ) d X
  outputs := tensorValues (WithBot ℝ) d Y

def logicalBackwardIO (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig) :
    LogicalProgramIO ℝ (WithBot ℝ) (BackwardInputs d) (Tensor ℝ d) where
  core := backwardIO P G Y d c
  inputs := backwardValues P G Y d c
  outputs := tensorValues ℝ d Y

/-- The unchanged original domain, bijectively expressed on the logical tensor:
each N-column has one finite logit, and the other logits may be negative infinity. -/
def LogicalForwardDomain (X Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (xs : Tensor (WithBot ℝ) d) : Prop :=
  ForwardDomain X Y d c ((tensorValues (WithBot ℝ) d X).symm xs)

end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Interface
