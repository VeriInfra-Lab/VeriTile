import VeriTile.Triton.Float.LogicalProgramIO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax
import bench.repairs.tritonbench_g.int8_quantization.Proofs.NativeGraph
import bench.repairs.tritonbench_g.int8_quantization.Proofs.SourceRealTotal
import VeriTile.Triton.Float.ProgramIO

namespace VeriTile.Bench.Repairs.Int8Quantization.QuantizationIO
open VeriTile.Triton FP.Structural Revision002 Words
noncomputable section
set_option autoImplicit false

def count (p : Parameters) := p.batches*p.blocks
def active (p : Parameters) (group lane : Nat) : Prop :=
  lane < p.blockRows*p.channels ∧ group%p.blocks*p.blockRows+lane/p.channels < p.length
instance (p : Parameters) (g l : Nat) : Decidable (active p g l) :=
  inferInstanceAs (Decidable (_ ∧ _))
def offset (p : Parameters) (group lane : Nat) : Nat :=
  group/p.blocks*p.length*p.channels+group%p.blocks*p.blockRows*p.channels+lane

/-- Source storage is FP32/INT8/FP32; the Structural interpreter represents
their cells in the Real/Int/Real channels, without changing source precision. -/
def inputChannels (p : Parameters) : List IOChannel := [⟨.real,[count p,p.width],xRegion⟩]
def outputChannels (p : Parameters) : List IOChannel :=
  [⟨.int,[count p,p.width],outRegion⟩,⟨.real,[count p],scaleRegion⟩]
abbrev Inputs (α : Type) (p : Parameters) := ChannelValues α (inputChannels p)
abbrev Outputs (α : Type) (p : Parameters) := ChannelValues α (outputChannels p)

def memory (α : Type) (p : Parameters) : MemoryIO α where
  inputs := inputChannels p
  outputs := outputChannels p
  scratch := []
  buffers := [xRegion,outRegion,scaleRegion]
  read | ⟨0,_⟩, _, i => offset p i.1.val i.2.1.val
  inputMask | ⟨0,_⟩, _, i => active p i.1.val i.2.1.val
  write
    | ⟨0,_⟩, _, i => offset p i.1.val i.2.1.val
    | ⟨1,_⟩, _, i => i.1.val
  outputMask
    | ⟨0,_⟩, _, i => active p i.1.val i.2.1.val
    | ⟨1,_⟩, _, _ => True
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k

/-- Exact existing finite launch order and base/program-ID ABI. -/
def invocation {α : Type} (M : Algebra α) (ops : Model.Primitives α)
    (p : Parameters) (w : Schedule.Program) : Invocation α where
  kernel := source (Grid.parameters p w)
  backend := Model.backend M ops
  pids := fun _ => w.pid
  setPids := fun ax => ax == 0

def quantizationIO {α : Type} (M : Algebra α) (ops : Model.Primitives α) (p : Parameters) : ProgramIO α α where
  toMemoryIO := memory α p
  schedule := fun _ => (Schedule.programs 0 (count p)).map (invocation M ops p)
  encode := id
  observe := fun _ _ => id

def raw {α : Type} (p : Parameters) (values : Inputs α p) (group : Fin (count p)) (lane : Fin p.width) : α :=
  values ⟨0,by simp [inputChannels]⟩ (group,lane,PUnit.unit)
def padded {α : Type} (M : Algebra α) (p : Parameters) (values : Inputs α p)
    (group : Fin (count p)) (lane : Fin p.width) : α :=
  M.fp32Load (if active p group.val lane.val then raw p values group lane else scalar32 M 0.0)

def logicalBlock (p : Parameters) (values : Inputs ℝ p) (group : Fin (count p)) : List ℝ :=
  ((List.finRange p.width).filter (fun j => decide (active p group.val j.val))).map (raw p values group)
def maximum (p : Parameters) (values : Inputs ℝ p) (group : Fin (count p)) : ℝ :=
  RealSpec.maxAbs ((logicalBlock p values group).map (fun x => RealSpec.factor p.query p.channels*x))
def FiniteBlock (p : Parameters) (values : Inputs ℝ p) (group : Fin (count p)) : Prop :=
  ∀ lane, active p group.val lane.val →
    |RealSpec.factor p.query p.channels*raw p values group lane| ≤ 340282346638528859811704183484516925440

/-- Independent total mathematical formula: max-absolute scaling and nearest
away rounding; every block outside finite FP32 range produces zero. -/
def realTarget (p : Parameters) (values : Inputs ℝ p) : Outputs ℝ p := by
  classical
  exact fun o => match o with
    | ⟨0,_⟩ => fun i => if FiniteBlock p values i.1 then
        RealSpec.nearestAway (127*(RealSpec.factor p.query p.channels*raw p values i.1 i.2.1)/maximum p values i.1)
      else (0 : Int)
    | ⟨1,_⟩ => fun i => if FiniteBlock p values i.1 then maximum p values i.1/127 else (0 : ℝ)

/-- Exactly the two original local context ports, now on logical operands.
The remaining numerical comparison uses the existing fourteen-port record. -/
def scaled {α : Type} (M : Algebra α) (p : Parameters) (values : Inputs α p)
    (group : Fin (count p)) (lane : Fin p.width) : α :=
  if p.query then M.binary (some .fp32) .real .mul (padded M p values group lane) (scalar32 M p.factor)
  else padded M p values group lane

def Context {α : Type} (M : Algebra α) (p : Parameters) (values : Inputs α p)
    (group : Fin (count p)) : Prop :=
  M.literal none .real 0.0 = scalar32 M 0.0 ∧ ∀ lane,
    M.unary none (.abs .real) (scaled M p values group lane) =
      M.unary (some .fp32) (.abs .real) (scaled M p values group lane)

def Agreements {α : Type} (M : Algebra α) (ops : Model.Primitives α) (software : Floating.Numerics α)
    (p : Parameters) (values : Inputs α p) : Prop :=
  ∀ group, Context M p values group ∧ Floating.Local (FPTrace.constants M p) (FPTrace.numerics M ops)
    software p.query (padded M p values group)

def softwareTarget {α : Type} (M : Algebra α) (software : Floating.Numerics α)
    (p : Parameters) (values : Inputs α p) : Outputs α p
  | ⟨0,_⟩ => fun i => (Floating.run (FPTrace.constants M p) software p.query (padded M p values i.1)).1 i.2.1
  | ⟨1,_⟩ => fun i => (Floating.run (FPTrace.constants M p) software p.query (padded M p values i.1)).2

def nativeTarget (op : NativeGraph.Instructions) (plans : NativeGraph.Plans)
    (p : Parameters) (values : Inputs Word p) : Outputs Word p
  | ⟨0,_⟩ => fun i => (NativeGraph.block op plans p.query p.factor
      (fun lane => if active p i.1.val lane.val then raw p values i.1 lane else op.literal32 0.0)).1 i.2.1
  | ⟨1,_⟩ => fun i => (NativeGraph.block op plans p.query p.factor
      (fun lane => if active p i.1.val lane.val then raw p values i.1 lane else op.literal32 0.0)).2

end
end VeriTile.Bench.Repairs.Int8Quantization.QuantizationIO

namespace VeriTile.Bench.Repairs.Int8Quantization.QuantizationIO
open VeriTile.Triton FP.Structural Revision002
noncomputable section
variable {α : Type} [Inhabited α]
def biasInvocation (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (w : Schedule.Program) : Invocation α :=
  { invocation M ops p w with kernel := biasFoldSource (Grid.parameters p w) }

def biasVariant (M : Algebra α) (ops : Model.Primitives α) (p : Parameters) :
    ProgramIO.Variant (quantizationIO M ops p) where
  schedule := fun _ => (Schedule.programs 0 (count p)).map (biasInvocation M ops p)
  observe := fun _ _ => id

def original (M : Algebra α) (ops : Model.Primitives α) (p : Parameters) :=
  ProgramIO.Implementation.original (quantizationIO M ops p) (fun _=>True)

def biasFolded (M : Algebra α) (ops : Model.Primitives α) (p : Parameters) :=
  ProgramIO.Implementation.ofVariant (biasVariant M ops p) (fun _=>True)


end
end VeriTile.Bench.Repairs.Int8Quantization.QuantizationIO

namespace VeriTile.Bench.Repairs.Int8Quantization.QuantizationIO
open VeriTile.Triton FP.Structural Revision002
noncomputable section
structure LogicalOutputs (α : Type) (p : Parameters) where
  codes : Fin (count p) → Fin p.width → Int
  scales : Fin (count p) → α

def logicalInputs (α : Type) (p : Parameters) : Inputs α p ≃ (Fin (count p) → Fin p.width → α) where
  toFun xs := fun g i => xs ⟨0,by simp [inputChannels]⟩ (g,i,())
  invFun xs := fun o => match o with | ⟨0,_⟩ => fun i => xs i.1 i.2.1
  left_inv := by intro xs; funext o; fin_cases o; funext j; rcases j with ⟨g,i,⟨⟩⟩; rfl
  right_inv := by intro xs; rfl

def logicalOutputs (α : Type) (p : Parameters) : Outputs α p ≃ LogicalOutputs α p where
  toFun xs := ⟨fun g i => xs ⟨0,by simp [outputChannels]⟩ (g,i,()),fun g => xs ⟨1,by simp [outputChannels]⟩ (g,())⟩
  invFun xs := fun o => match o with
    | ⟨0,_⟩ => fun i => xs.codes i.1 i.2.1
    | ⟨1,_⟩ => fun i => xs.scales i.1
  left_inv := by
    intro xs; funext o; fin_cases o
    · funext j; rcases j with ⟨g,i,⟨⟩⟩; rfl
    · funext j; rcases j with ⟨g,⟨⟩⟩; rfl
  right_inv := by intro xs; cases xs; rfl
end
end VeriTile.Bench.Repairs.Int8Quantization.QuantizationIO

namespace VeriTile.Bench.Repairs.Int8Quantization.QuantizationIO
open VeriTile.Triton FP.Structural
noncomputable section
open Revision002
def logicalIO {α : Type} (M : Algebra α) (ops : Model.Primitives α) (p : Parameters) : LogicalProgramIO α α (Fin (count p) → Fin p.width → α) (LogicalOutputs α p) where
  core := quantizationIO M ops p
  inputs := logicalInputs α p
  outputs := logicalOutputs α p
end
end VeriTile.Bench.Repairs.Int8Quantization.QuantizationIO

namespace VeriTile.Bench.Repairs.Int8Quantization.QuantizationIO
open Revision002
noncomputable section
def logicalFiniteBlock (p : Parameters) (xs : Fin (count p) → Fin p.width → ℝ) (group : Fin (count p)) : Prop :=
  ∀ lane, active p group.val lane.val →
    |RealSpec.factor p.query p.channels*xs group lane| ≤ 340282346638528859811704183484516925440
end
end VeriTile.Bench.Repairs.Int8Quantization.QuantizationIO
