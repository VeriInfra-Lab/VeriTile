import VeriTile.Triton.Float.LogicalProgramIO
import bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Proofs.RawKernel
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelExtended
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelLogBackward
import VeriTile.Triton.Float.AddressedInputsIO
import VeriTile.Triton.Float.ProgramEquivalence

/-! Full B×Q row-grid IO. D is the selected padded reduction width; only
lanes below K are outputs. Causal reads and absent/QK/BK mask addressing are
explicit, including repeated mask addresses across rows. The quoted source
fixes FP32 loads, arithmetic and stores; `.real` is its erased cell carrier. -/
namespace VeriTile.Bench.Repairs.KsoftmaxGridYChecked.Interface
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.Ksoftmax
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
noncomputable section

def inputs (B Q D : Nat) : List IOChannel :=
  [⟨.real, [B,Q,D], "left"⟩, ⟨.real, [B,Q,D], "right"⟩]
def outputs (B Q D : Nat) : List IOChannel := [⟨.real, [B,Q,D], "dst"⟩]
abbrev Inputs (A : Type) (B Q D : Nat) := ChannelValues A (inputs B Q D)
abbrev Outputs (A : Type) (B Q D : Nat) := ChannelValues A (outputs B Q D)
abbrev Coord (B Q D : Nat) := TileIndex [B,Q,D]

def inputCoord (B Q D : Nat) (p : Fin (inputs B Q D).length) :
    TileIndex (inputs B Q D)[p].shape → Coord B Q D :=
  match p with
  | ⟨0, _⟩ | ⟨1, _⟩ => id
  | ⟨n+2, h⟩ => False.elim (by simp [inputs] at h)
def outputCoord (B Q D : Nat) (p : Fin (outputs B Q D).length) :
    TileIndex (outputs B Q D)[p].shape → Coord B Q D :=
  match p with
  | ⟨0, _⟩ => id
  | ⟨n+1, h⟩ => False.elim (by simp [outputs] at h)

def memory (A : Type) (op : Operation) (B Q K D : Nat) : MemoryIO A where
  inputs := inputs B Q D
  outputs := outputs B Q D
  scratch := []
  buffers := ["left", "right", "dst"]
  read := fun p _ j =>
    let c := inputCoord B Q D p j
    if p.val = 0 then Native.flat Q K c.1 c.2.1 c.2.2.1
    else rightAddress op Q K c.1 c.2.1 c.2.2.1
  inputMask := fun p _ j =>
    let c := inputCoord B Q D p j
    (p.val = 0 ∨ op.readsRight = Bool.true) ∧
      Forward.readActive K c.2.1 op.causal c.2.2.1 = Bool.true
  write := fun p _ j =>
    let c := outputCoord B Q D p j
    Native.flat Q K c.1 c.2.1 c.2.2.1
  outputMask := fun p _ j => (outputCoord B Q D p j).2.2.1.val < K
  scratchWrite := fun p => Fin.elim0 p
  scratchMask := fun p => Fin.elim0 p

def invocation (M : Algebra A) (op : Operation) (Q K D b r : Nat) : Invocation A where
  kernel := KernelExecution.source op Q K D
  backend := FP.ProfiledRewrite.engine M
  pids := fun ax => if ax = 0 then b else r
  setPids := fun ax => decide (ax = 0 ∨ ax = 1)

/-- The selected checked launch skips all work when any logical extent is zero. -/
def selectedRows (B Q K : Nat) : List (Fin B × Fin Q) :=
  if B*Q*K = 0 then [] else programs B Q

def programIO (M : Algebra A) (op : Operation) (B Q K D : Nat)
    (jobs : List (Fin B × Fin Q)) : ProgramIO A A :=
  { memory A op B Q K D with
    schedule := fun _ => jobs.map (fun p => invocation M op Q K D p.1 p.2)
    encode := id
    observe := fun _ _ => id }

def addressedIO (M : Algebra A) (op : Operation) (B Q K D : Nat)
    (jobs : List (Fin B × Fin Q)) (encode : RegionName → Nat → V → A)
    (observe : A → V) : AddressedProgramIO V A :=
  { memory V op B Q K D with
    schedule := fun _ => jobs.map (fun p => invocation M op Q K D p.1 p.2)
    encodeAt := encode
    observe := fun _ _ => observe }

def left {A : Type} {B Q D : Nat} (v : Inputs A B Q D) (b : Fin B) (r : Fin Q) (i : Fin D) : A :=
  v ⟨0, by simp [inputs]⟩ (b,r,i,PUnit.unit)
def right {A : Type} {B Q D : Nat} (v : Inputs A B Q D) (b : Fin B) (r : Fin Q) (i : Fin D) : A :=
  v ⟨1, by simp [inputs]⟩ (b,r,i,PUnit.unit)
def result {A : Type} {B Q D : Nat} (f : Fin B → Fin Q → Fin D → A) : Outputs A B Q D :=
  fun p => match p with
  | ⟨0, _⟩ => fun j => f j.1 j.2.1 j.2.2.1
  | ⟨n+1, h⟩ => False.elim (by simp [outputs] at h)

/-- Independent ordered scalar/reduction reference. No runner or memory state
occurs here. Primitive callbacks retain source FP32 association and casts. -/
def orderedRow (M : Algebra A) (op : Operation) (K r : Nat) {D : Nat}
    (x y : Fin D → A) (i : Fin D) : A :=
  match op with
  | .forward log causal mode =>
    let a := selectedValue M M.negInf K r causal x
    let score := fun j => if mode.present then M.binary (some .fp32) .real .add (a j)
      (maskedValue M M.negInf K r causal y j) else a j
    let z := fun j => M.binary (some .fp32) .real .sub (score j) (rowMax M score)
    let num := fun j => M.unary (some .fp32) .exp (z j)
    let den := rowSum M num
    if log then M.binary (some .fp32) .real .sub (z i) (M.unary (some .fp32) .log den)
    else M.binary (some .fp32) .real .div (num i) den
  | .backward log causal =>
    let zero := M.literal (some .fp32) .real 0
    let g := selectedValue M zero K r causal x
    let o := selectedValue M zero K r causal y
    let v := if log then M.binary (some .fp32) .real .sub (g i)
        (M.binary (some .fp32) .real .mul (M.unary (some .fp32) .exp (o i)) (rowSum M g))
      else M.binary (some .fp32) .real .mul (o i)
        (M.binary (some .fp32) .real .sub (g i)
          (rowSum M (fun j => M.binary (some .fp32) .real .mul (g j) (o j))))
    if causal then if i.val ≤ r then v else zero else v

def orderedTarget (M : Algebra A) (op : Operation) (B Q K D : Nat)
    (encode : RegionName → Nat → V → A) (observe : A → V)
    (v : Inputs V B Q D) : Outputs V B Q D :=
  result fun b r i => observe (orderedRow M op K r
    (fun j => encode "left" (Native.flat Q K b r j.val) (left v b r j))
    (fun j => encode "right" (rightAddress op Q K b r j.val) (right v b r j)) i)

def forwardScores (K : Nat) (causal : Bool) (mode : Forward.MaskMode)
    (v : Inputs (WithBot ℝ) B Q D) (b : Fin B) (r : Fin Q) : Fin D → WithBot ℝ :=
  KernelExtended.scores K r causal mode (left v b r) (right v b r)
def ForwardDomain (K : Nat) (causal : Bool) (mode : Forward.MaskMode)
    (v : Inputs (WithBot ℝ) B Q D) : Prop :=
  ∀ b r, 0<K → KernelExtended.valid (forwardScores K causal mode v b r)
def forwardFormula (K : Nat) (log causal : Bool) (mode : Forward.MaskMode)
    (v : Inputs (WithBot ℝ) B Q D) : Outputs (WithBot ℝ) B Q D :=
  result fun b r i => KernelExtended.formula (forwardScores K causal mode v b r) log i

/-- Ordinary-Real gradient scope. This is mathematical input interpretation,
not a Python finite-word guard or an experimentally deferred assumption. -/
def BackwardDomain (K : Nat) (causal : Bool) (extendedSaved : Bool)
    (v : Inputs (WithBot ℝ) B Q D) : Prop :=
  ∀ (b : Fin B) (r : Fin Q) (i : Fin D), Forward.readActive K r causal i.val = Bool.true →
    (left v b r i).isSome ∧ (extendedSaved = Bool.true ∨ (right v b r i).isSome)
def backwardFormula (K : Nat) (log causal : Bool)
    (v : Inputs (WithBot ℝ) B Q D) : Outputs (WithBot ℝ) B Q D :=
  result fun b r i => some (gradient K r log causal
    (fun j => (left v b r j).getD 0) (fun j => (right v b r j).getD 0) i)
def logBackwardFormula (K : Nat) (causal : Bool)
    (v : Inputs (WithBot ℝ) B Q D) : Outputs (WithBot ℝ) B Q D :=
  result fun b r i => some (KernelLogBackward.formula K r causal
    (fun j => (left v b r j).getD 0) (right v b r) i)

def nativeEncode (r : RegionName) (i : Nat) (w : Graph.Word) : Graph.Expr :=
  .input (if r = "left" then 0 else 1) i w

end
end VeriTile.Bench.Repairs.KsoftmaxGridYChecked.Interface

namespace VeriTile.Bench.Repairs.KsoftmaxGridYChecked.Interface
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.Ksoftmax
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
noncomputable section

def nativeIO (rest : Algebra Graph.Expr) (platform : KernelCompiler.Platform)
    (op : Operation) (B Q K D : Nat) (dst leftPtr rightPtr : BitVec 64)
    (code : RawKernel.Selected platform op Q K D dst leftPtr rightPtr) : AddressedProgramIO Graph.Word Graph.Expr :=
  addressedIO (Graph.algebra rest code.lowering.profile) op B Q K D (selectedRows B Q K)
    nativeEncode (Graph.eval (platform.instructions code.object.primitives))

def nativeFormula (rest : Algebra Graph.Expr) (platform : KernelCompiler.Platform)
    (op : Operation) (B Q K D : Nat) (dst leftPtr rightPtr : BitVec 64)
    (code : RawKernel.Selected platform op Q K D dst leftPtr rightPtr) :
    Inputs Graph.Word B Q D → Outputs Graph.Word B Q D :=
  orderedTarget (Graph.algebra rest code.lowering.profile) op B Q K D
    nativeEncode (Graph.eval (platform.instructions code.object.primitives))

def reversed (M : Algebra A) (op : Operation) (B Q K D : Nat) :
    ProgramIO.Variant (programIO M op B Q K D (programs B Q)) where
  schedule := fun _ => (programs B Q).reverse.map (fun p => invocation M op Q K D p.1 p.2)
  observe := fun _ _ => id

/-- Embed every old raw word array, including exceptional words, into the
public logical channels. Shared mask addresses receive the same source word. -/
def words (op : Operation) (B Q K D : Nat) (x y : Nat → Graph.Word) : Inputs Graph.Word B Q D :=
  fun p => match p with
  | ⟨0, _⟩ => fun j => x (Native.flat Q K j.1 j.2.1 j.2.2.1)
  | ⟨1, _⟩ => fun j => y (rightAddress op Q K j.1 j.2.1 j.2.2.1)
  | ⟨n+2, h⟩ => False.elim (by simp [inputs] at h)

end
end VeriTile.Bench.Repairs.KsoftmaxGridYChecked.Interface

namespace VeriTile.Bench.Repairs.KsoftmaxGridYChecked.Interface
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.Ksoftmax
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
noncomputable section
structure LogicalInputs (A : Type) (B Q D : Nat) where
  left : Fin B → Fin Q → Fin D → A
  right : Fin B → Fin Q → Fin D → A

def logicalInputs (A : Type) (B Q D : Nat) : Inputs A B Q D ≃ LogicalInputs A B Q D where
  toFun xs := ⟨fun b r i => xs ⟨0,by simp [inputs]⟩ (b,r,i,()),fun b r i => xs ⟨1,by simp [inputs]⟩ (b,r,i,())⟩
  invFun xs := fun o => match o with
    | ⟨0,_⟩ => fun j => xs.left j.1 j.2.1 j.2.2.1
    | ⟨1,_⟩ => fun j => xs.right j.1 j.2.1 j.2.2.1
  left_inv := by intro xs; funext o; fin_cases o <;> funext j <;> rcases j with ⟨b,r,i,⟨⟩⟩ <;> rfl
  right_inv := by intro xs; cases xs; rfl

def logicalOutputs (A : Type) (B Q D : Nat) : Outputs A B Q D ≃ (Fin B → Fin Q → Fin D → A) where
  toFun xs := fun b r i => xs ⟨0,by simp [outputs]⟩ (b,r,i,())
  invFun xs := fun o => match o with | ⟨0,_⟩ => fun j => xs j.1 j.2.1 j.2.2.1
  left_inv := by intro xs; funext o; fin_cases o; funext j; rcases j with ⟨b,r,i,⟨⟩⟩; rfl
  right_inv := by intro xs; rfl

def logicalScore {B Q D : Nat} (K : Nat) (covered : K≤D) (causal : Bool) (mode : Forward.MaskMode)
    (xs : LogicalInputs (WithBot ℝ) B Q D) (b : Fin B) (r : Fin Q) (i : Fin K) : WithBot ℝ :=
  forwardScores K causal mode ((logicalInputs _ B Q D).symm xs) b r (Fin.castLE covered i)
end
end VeriTile.Bench.Repairs.KsoftmaxGridYChecked.Interface

namespace VeriTile.Bench.Repairs.KsoftmaxGridYChecked.Interface
open VeriTile.Triton FP.Structural
noncomputable section
open VeriTile.Bench.Repairs.Ksoftmax
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
def logicalIO (M : Algebra A) (op : Operation) (B Q K D : Nat) : LogicalProgramIO A A (LogicalInputs A B Q D) (Fin B → Fin Q → Fin D → A) where
  core := programIO M op B Q K D (selectedRows B Q K)
  inputs := logicalInputs A B Q D
  outputs := logicalOutputs A B Q D

def logicalForwardDomain (K : Nat) (causal : Bool) (mode : Forward.MaskMode) (xs : LogicalInputs (WithBot ℝ) B Q D) : Prop :=
  ForwardDomain K causal mode ((logicalInputs _ B Q D).symm xs)
def logicalBackwardDomain (K : Nat) (causal extendedSaved : Bool) (xs : LogicalInputs (WithBot ℝ) B Q D) : Prop :=
  BackwardDomain K causal extendedSaved ((logicalInputs _ B Q D).symm xs)
def score (K : Nat) (causal : Bool) (mode : Forward.MaskMode) (xs : LogicalInputs (WithBot ℝ) B Q D) (b : Fin B) (r : Fin Q) : Fin D → WithBot ℝ :=
  KernelExtended.scores K r causal mode (xs.left b r) (xs.right b r)
def logicalLeft {B Q D : Nat} (K : Nat) (covered : K≤D) (xs : LogicalInputs (WithBot ℝ) B Q D) (b : Fin B) (r : Fin Q) (i : Fin K) : WithBot ℝ :=
  xs.left b r (Fin.castLE covered i)
def logicalRight {B Q D : Nat} (K : Nat) (covered : K≤D) (xs : LogicalInputs (WithBot ℝ) B Q D) (b : Fin B) (r : Fin Q) (i : Fin K) : WithBot ℝ :=
  xs.right b r (Fin.castLE covered i)
end
end VeriTile.Bench.Repairs.KsoftmaxGridYChecked.Interface
