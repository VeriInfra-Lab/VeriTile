import bench.repairs.tritonbench_g.log_softmax.Proofs.KernelGrid
import VeriTile.Triton.Float.ProgramEquivalence
import bench.repairs.tritonbench_g.log_softmax.Proofs.Coordinates
import VeriTile.Triton.Float.LogicalProgramIO


/-! Dense regional channels for the complete canonical source grids. The
source fixes FP32 precision; `.real` is its Structural interpreter cell type.
All dimensions, masks, full launch order and the index-only variant are explicit.
-/
namespace VeriTile.Bench.Repairs.LogSoftmax.IO
open VeriTile.Triton FP.Structural
noncomputable section

def launch (direction : Direction) (algebra : Algebra β) (candidate : Bool)
    (M N K BM C : Nat) (p : Nat × Nat) : Invocation β where
  kernel := KernelGrid.program direction candidate M N K BM C
  backend := algebra
  pids := fun ax => if ax=0 then p.1 else if ax=1 then p.2 else 0

def schedule (direction : Direction) (algebra : Algebra β) (candidate : Bool)
    (M N K BM C : Nat) : List (Invocation β) :=
  (KernelGrid.programs M N K BM).map (launch direction algebra candidate M N K BM C)

def forward (encode : α → β) (observe : β → α) (algebra : Algebra β)
    (candidate : Bool) (M N K BM C : Nat) : ProgramIO α β where
  inputs := [⟨.real,[M*N*K],"input"⟩]
  outputs := [⟨.real,[M*N*K],"output"⟩]
  scratch := []
  buffers := ["input","output"]
  read := fun i => Fin.cases (fun _ j => j.1.val) (fun i => Fin.elim0 i) i
  inputMask := fun _ _ _ => True
  write := fun o => Fin.cases (fun _ j => j.1.val) (fun i => Fin.elim0 i) o
  outputMask := fun _ _ _ => True
  scratchWrite := (fun i => Fin.elim0 i)
  scratchMask := (fun i => Fin.elim0 i)
  schedule := fun _ => schedule .forward algebra candidate M N K BM C
  encode := encode
  observe := fun _ _ => observe

def backward (encode : α → β) (observe : β → α) (algebra : Algebra β)
    (candidate : Bool) (M N K BM C : Nat) : ProgramIO α β where
  inputs := [⟨.real,[M*N*K],"input"⟩,⟨.real,[M*N*K],"gradient"⟩]
  outputs := [⟨.real,[M*N*K],"output"⟩]
  scratch := []
  buffers := ["input","gradient","output"]
  read := fun i => Fin.cases (fun _ j => j.1.val)
    (Fin.cases (fun _ j => j.1.val) (fun i => Fin.elim0 i)) i
  inputMask := fun _ _ _ => True
  write := fun o => Fin.cases (fun _ j => j.1.val) (fun i => Fin.elim0 i) o
  outputMask := fun _ _ _ => True
  scratchWrite := (fun i => Fin.elim0 i)
  scratchMask := (fun i => Fin.elim0 i)
  schedule := fun _ => schedule .backward algebra candidate M N K BM C
  encode := encode
  observe := fun _ _ => observe

def grid (direction : Direction) (encode : α → β) (observe : β → α)
    (algebra : Algebra β) (candidate : Bool) (M N K BM C : Nat) : ProgramIO α β :=
  match direction with
  | .forward => forward encode observe algebra candidate M N K BM C
  | .backward => backward encode observe algebra candidate M N K BM C

/-- The retained candidate changes only integer index multiplication. -/
def commuted (direction : Direction) (encode : α → β) (observe : β → α)
    (algebra : Algebra β) (M N K BM C : Nat) :
    ProgramIO.Variant (grid direction encode observe algebra Bool.false M N K BM C) where
  schedule := fun _ => schedule direction algebra Bool.true M N K BM C
  observe := fun _ _ => observe

/-- Extend a logical flat input only to state the independent row formulas.
Every queried row element in the theorem is within the channel extent. -/
def vector (fallback : α) (xs : Values α .real [count]) (i : Nat) : α :=
  if h : i<count then xs (⟨i,h⟩,PUnit.unit) else fallback

def realForward (M N K BM : Nat) (plan : Plan N (Grid.blockColumns N)) :
    ProgramIO ℝ (WithBot ℝ) :=
  forward some (fun v => v.getD 0) (KernelMath.forwardEngine RealProof.ops plan).algebra
    Bool.false M N K BM (Grid.blockColumns N)

def realBackward (M N K BM : Nat) (plan : Plan N (Grid.blockColumns N)) :
    ProgramIO ℝ (WithBot ℝ) :=
  backward some (fun v => v.getD 0) (KernelMath.backwardEngine RealProof.ops plan).algebra
    Bool.false M N K BM (Grid.blockColumns N)

end
end VeriTile.Bench.Repairs.LogSoftmax.IO

namespace VeriTile.Bench.Repairs.LogSoftmax.ReaderIO
open VeriTile.Triton FP.Structural Address
noncomputable section
abbrev Tensor (M N K : Nat) := Fin M → Fin N → Fin K → ℝ

/-- Every dense flat input cell corresponds to exactly one logical tensor cell. -/
def tensor (M N K : Nat) : Values ℝ .real [M*N*K] ≃ Tensor M N K where
  toFun := fun xs m n k => xs (⟨linear (m,n,k),linear_bound _⟩,())
  invFun := fun xs j => let q := Coordinates.decode j.1; xs q.1 q.2.1 q.2.2
  left_inv := by
    intro xs; funext j
    apply congrArg xs
    apply Prod.ext
    · apply Fin.ext; exact Coordinates.decode_linear j.1
    · exact Subsingleton.elim _ _
  right_inv := by
    intro xs; funext m n k
    have same : Coordinates.decode (⟨linear (m,n,k),linear_bound _⟩ : Fin (M*N*K)) =
        (m,n,k) := linear_injective (Coordinates.decode_linear _)
    change xs _ _ _ = xs m n k
    rw [same]

structure BackwardInputs (M N K : Nat) where
  saved : Tensor M N K
  upstream : Tensor M N K

def forwardInputs (M N K : Nat) : ChannelValues ℝ [⟨.real,[M*N*K],"input"⟩] ≃ Tensor M N K :=
  (ChannelValues.singleEquiv ℝ _).trans (tensor M N K)
def outputs (M N K : Nat) : ChannelValues ℝ [⟨.real,[M*N*K],"output"⟩] ≃ Tensor M N K :=
  (ChannelValues.singleEquiv ℝ _).trans (tensor M N K)

def backwardInputs (M N K : Nat) :
    ChannelValues ℝ [⟨.real,[M*N*K],"input"⟩,⟨.real,[M*N*K],"gradient"⟩] ≃ BackwardInputs M N K where
  toFun := fun xs => ⟨tensor M N K (xs 0), tensor M N K (xs 1)⟩
  invFun := fun xs => Fin.cases ((tensor M N K).symm xs.saved)
    (Fin.cases ((tensor M N K).symm xs.upstream) (fun i => i.elim0))
  left_inv := by
    intro xs; funext i; fin_cases i <;> exact (tensor M N K).symm_apply_apply _
  right_inv := by
    intro xs
    change (⟨(tensor M N K) ((tensor M N K).symm xs.saved),
      (tensor M N K) ((tensor M N K).symm xs.upstream)⟩ : BackwardInputs M N K) = xs
    simp only [Equiv.apply_symm_apply]

def forward (M N K BM : Nat) (plan : Plan N (Grid.blockColumns N)) :
    LogicalProgramIO ℝ (WithBot ℝ) (Tensor M N K) (Tensor M N K) where
  core := IO.realForward M N K BM plan
  inputs := forwardInputs M N K
  outputs := outputs M N K

def backward (M N K BM : Nat) (plan : Plan N (Grid.blockColumns N)) :
    LogicalProgramIO ℝ (WithBot ℝ) (BackwardInputs M N K) (Tensor M N K) where
  core := IO.realBackward M N K BM plan
  inputs := backwardInputs M N K
  outputs := outputs M N K
end
end VeriTile.Bench.Repairs.LogSoftmax.ReaderIO

namespace VeriTile.Bench.Repairs.LogSoftmax.Reference
open VeriTile.Triton FP.Structural Address
noncomputable section

/-- The usual unshifted mathematical log-softmax on a complete logical row. -/
def forward (M N K : Nat) (xs : Values ℝ .real [M*N*K])
    (i : TileIndex [M*N*K]) : ℝ :=
  let q := Coordinates.decode i.1
  RealProof.targetForward
    (fun j : Fin N => IO.vector 0 xs (offset N K q.1.val j.val q.2.2.val)) q.2.1

/-- VJP from the supplied saved values and upstream gradient; it does not
require that the saved values came from a preceding forward invocation. -/
def backward (M N K : Nat) (ys gs : Values ℝ .real [M*N*K])
    (i : TileIndex [M*N*K]) : ℝ :=
  let q := Coordinates.decode i.1
  RealProof.targetBackward N
    (fun j => IO.vector 0 ys (offset N K q.1.val j q.2.2.val))
    (fun j => IO.vector 0 gs (offset N K q.1.val j q.2.2.val)) q.2.1.val

def forwardFormula (M N K BM : Nat) (plan : Plan N (Grid.blockColumns N)) :
    (IO.realForward M N K BM plan).Inputs → (IO.realForward M N K BM plan).Outputs :=
  fun vals o => Fin.cases (forward M N K (vals (0 : Fin 1))) (fun i => Fin.elim0 i) o

def backwardFormula (M N K BM : Nat) (plan : Plan N (Grid.blockColumns N)) :
    (IO.realBackward M N K BM plan).Inputs → (IO.realBackward M N K BM plan).Outputs :=
  fun vals o => Fin.cases (backward M N K (vals (0 : Fin 2)) (vals (1 : Fin 2))) (fun i => Fin.elim0 i) o

end
end VeriTile.Bench.Repairs.LogSoftmax.Reference
