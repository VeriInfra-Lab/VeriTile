import bench.tritonbench_g.rms_rbe_matmul.Proofs.KernelCoverage
import VeriTile.Triton.Float.ProgramIO
import Mathlib.Tactic.FinCases

import bench.tritonbench_g.rms_rbe_matmul.Proofs.ProgramPairSource
import VeriTile.Triton.Float.ProgramEquivalenceSyntax
import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.rms_rbe_matmul.Proofs.KernelReal


namespace VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.PublicIO
open VeriTile.Bench.Repairs.RmsRbeMatmul
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
variable {α : Type} [Inhabited α]

/-- These are the existing interpreter channels: FP32 source X uses .real,
while W/RMS retain signed E5 bytes or FP16 values, and the output keeps its
selected FP8/FP16 storage format. -/
def inputs (fp8 : Bool) (B : Nat) (p : Parameters) : List IOChannel :=
  [⟨.real,[B,p.M,p.K],"X"⟩,⟨if fp8 then .int else .fp16,[p.N,p.K],"W"⟩,
    ⟨if fp8 then .int else .fp16,[p.K],"RMS"⟩]
def outputs (fp8 : Bool) (B : Nat) (p : Parameters) : List IOChannel :=
  [⟨(outputDType fp8).toTileDType,[B,p.M,p.N],"OUT"⟩]

def read (fp8 : Bool) (B : Nat) (p : Parameters) :
    (i : Fin (inputs fp8 B p).length) → ChannelValues α (inputs fp8 B p) →
      TileIndex ((inputs fp8 B p)[i]).shape → Nat := by
  refine Fin.cases ?_ (Fin.cases ?_ (Fin.cases ?_ (fun k => k.elim0)))
  · intro _ j; exact j.1.val*(p.M*p.K)+j.2.1.val*p.K+j.2.2.1.val
  · intro _ j; exact j.2.1.val+j.1.val*p.K
  · intro _ j; exact j.1.val

def invocation (a : Operations α) (fp8 : Bool) (p : Parameters)
    (q : KernelGrid.Program) : Invocation α where
  kernel := KernelSource.kernel fp8 p
  backend := engine a
  pids := fun ax => if ax=0 then q.1 else if ax=1 then q.2 else 0

/-- The complete original rectangular launch, including the empty-grid case. -/
def io (a : Operations α) (fp8 : Bool) (B : Nat) (p : Parameters) : ProgramIO α α where
  inputs := inputs fp8 B p
  outputs := outputs fp8 B p
  scratch := []
  buffers := ["X","W","RMS","OUT"]
  read := read fp8 B p
  inputMask := fun _ _ _ => True
  write := Fin.cases (fun _ j => j.1.val*(p.M*p.N)+j.2.1.val*p.N+j.2.2.1.val)
    (fun k => k.elim0)
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => (KernelGrid.programs B p).map (invocation a fp8 p)
  encode := id
  observe := fun _ _ => id

/-- Signed-byte validity is a storage-format condition. No restriction is
placed on FP16 values or floating finiteness. -/
def domain (fp8 : Bool) (B : Nat) (p : Parameters) :
    ChannelValues α (inputs fp8 B p) → Prop := by
  cases fp8
  · exact fun _ => True
  · exact fun xs =>
      (∀ j, Float8E5Bits.isInt8 (xs (1 : Fin 3) j)=Bool.true) ∧
      (∀ j, Float8E5Bits.isInt8 (xs (2 : Fin 3) j)=Bool.true)

def tensor1 {β : Type} (padding : β) (n : Nat) (xs : TileIndex [n] → β) (i : Nat) : β :=
  if h : i<n then xs (⟨i,h⟩,PUnit.unit) else padding
def tensor2 {β : Type} (padding : β) (m n : Nat) (xs : TileIndex [m,n] → β) (i j : Nat) : β :=
  if h : i<m ∧ j<n then xs (⟨i,h.1⟩,⟨j,h.2⟩,PUnit.unit) else padding
def tensor3 {β : Type} (padding : β) (b m n : Nat) (xs : TileIndex [b,m,n] → β) (i j k : Nat) : β :=
  if h : i<b ∧ j<m ∧ k<n then xs (⟨i,h.1⟩,⟨j,h.2.1⟩,⟨k,h.2.2⟩,PUnit.unit) else padding

/-- Logical input channels only. Exterior padding is never used by an active
source read; FP16 and raw signed FP8 inputs remain separate representations. -/
def logical (fp8 : Bool) (B : Nat) (p : Parameters) :
    ChannelValues α (inputs fp8 B p) → KernelGraph.Input α := by
  cases fp8
  · intro xs
    exact {
      x := tensor3 default B p.M p.K (xs (0 : Fin 3))
      weightHalf := tensor2 default p.N p.K (xs (1 : Fin 3))
      rmsHalf := tensor1 default p.K (xs (2 : Fin 3))
      weightByte := fun _ _ => 0
      rmsByte := fun _ => 0 }
  · intro xs
    exact {
      x := tensor3 default B p.M p.K (xs (0 : Fin 3))
      weightHalf := fun _ _ => default
      rmsHalf := fun _ => default
      weightByte := tensor2 0 p.N p.K (xs (1 : Fin 3))
      rmsByte := tensor1 0 p.K (xs (2 : Fin 3)) }


end
end VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.PublicIO

namespace VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.PairIO
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.RmsRbeMatmul
noncomputable section
variable {α : Type} [Inhabited α]

/-- Same invocation ABI and native program IDs; only the displayed source changes. -/
def candidateInvocation (a : KernelSource.Operations α) (fp8 : Bool)
    (p : KernelSource.Parameters) (q : KernelGrid.Program) : Invocation α :=
  {PublicIO.invocation a fp8 p q with kernel := SourcePair.candidate fp8 p}

/-- The whole rectangular launch, with the original inputs, outputs and frame. -/
def candidate (a : KernelSource.Operations α) (fp8 : Bool) (B : Nat)
    (p : KernelSource.Parameters) : ProgramIO.Variant (PublicIO.io a fp8 B p) where
  schedule := fun _ => (KernelGrid.programs B p).map (candidateInvocation a fp8 p)
  observe := fun _ _ => id

def originalInvocation (a : KernelSource.Operations α) (fp8 : Bool)
    (p : KernelSource.Parameters) (q : KernelGrid.Program) : Invocation α :=
  {PublicIO.invocation a fp8 p q with kernel := TypedSource.source fp8 (KernelSource.arguments p)}

/-- The actual readable original schedule on the unchanged public IO. -/
def original (a : KernelSource.Operations α) (fp8 : Bool) (B : Nat) (p : KernelSource.Parameters) :
    ProgramIO.Implementation (PublicIO.io a fp8 B p) (PublicIO.domain fp8 B p) where
  schedule := fun _ => (KernelGrid.programs B p).map (originalInvocation a fp8 p)
  observe := fun _ _ => id

def indexCommuted (a : KernelSource.Operations α) (fp8 : Bool) (B : Nat) (p : KernelSource.Parameters) :=
  ProgramIO.Implementation.ofVariant (candidate a fp8 B p) (PublicIO.domain fp8 B p)
end
end VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.PairIO

namespace VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.LogicalIO
open VeriTile.Bench.Repairs.RmsRbeMatmul
open VeriTile.Triton FP.Structural
noncomputable section

abbrev WeightValue (fp8 : Bool) := Value ℝ (if fp8 then .int else .fp16)
abbrev Inputs (fp8 : Bool) (B : Nat) (p : KernelSource.Parameters) :=
  (Fin B → Fin p.M → Fin p.K → ℝ) ×
    ((Fin p.N → Fin p.K → WeightValue fp8) × (Fin p.K → WeightValue fp8))

/-- All three original typed input tensors, with their complete inverses. -/
def inputs (fp8 : Bool) (B : Nat) (p : KernelSource.Parameters) :
    ChannelValues ℝ (PublicIO.inputs fp8 B p) ≃ Inputs fp8 B p :=
  (ChannelValues.consEquiv ℝ _ _).trans (Equiv.prodCongr
    (Values.tensor3Equiv ℝ .real _ _ _)
    ((ChannelValues.consEquiv ℝ _ _).trans (Equiv.prodCongr
      (Values.matrixEquiv ℝ _ _ _)
      ((ChannelValues.singleEquiv ℝ _).trans (Values.vectorEquiv ℝ _ _)))))

/-- Only the one-channel/tuple encoding is removed; the core observer and
selected FP16/E5 cell type stay unchanged. -/
def outputs (fp8 : Bool) (B : Nat) (p : KernelSource.Parameters) :
    ChannelValues ℝ (PublicIO.outputs fp8 B p) ≃ (Fin B → Fin p.M → Fin p.N → ℝ) := by
  cases fp8 <;> exact (ChannelValues.singleEquiv ℝ _).trans (Values.tensor3Equiv ℝ _ _ _ _)

/-- Decode only the source-selected storage branch, on logical tensor inputs. -/
def rowInput (d8 : Float8E5Bits → ℝ) (fp8 : Bool) (B : Nat) (p : KernelSource.Parameters)
    (values : Inputs fp8 B p) (batch row : Nat) : KernelMath.Input :=
  KernelReal.logical d8 fp8 (PublicIO.logical fp8 B p ((inputs fp8 B p).symm values)) batch row

/-- Exact pullback of the original signed-byte storage condition. -/
def domain (fp8 : Bool) (B : Nat) (p : KernelSource.Parameters) (values : Inputs fp8 B p) : Prop :=
  PublicIO.domain fp8 B p ((inputs fp8 B p).symm values)

/-- Whole rectangular launch, all original typed observations and frame. -/
def io (d8 : Float8E5Bits → ℝ) (fp8 : Bool) (B : Nat) (p : KernelSource.Parameters) :
    LogicalProgramIO ℝ ℝ (Inputs fp8 B p) (Fin B → Fin p.M → Fin p.N → ℝ) where
  core := PublicIO.io (KernelReal.operations d8) fp8 B p
  inputs := inputs fp8 B p
  outputs := outputs fp8 B p

end
end VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.LogicalIO

namespace VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard
open VeriTile.Bench.Repairs.RmsRbeMatmul
open VeriTile.Triton FP.Structural
noncomputable section

/-- Compatibility view of the unchanged typed source. -/
def realIO (d8 : Float8E5Bits → ℝ) (fp8 : Bool) (B : Nat)
    (p : KernelSource.Parameters) : ProgramIO ℝ ℝ :=
  PublicIO.io (KernelReal.operations d8) fp8 B p

def realStorageDomain (fp8 : Bool) (B : Nat) (p : KernelSource.Parameters) :=
  @PublicIO.domain ℝ fp8 B p

def realFormula (d8 : Float8E5Bits → ℝ) (fp8 : Bool) (B : Nat) (p : KernelSource.Parameters)
    (xs : ChannelValues ℝ (PublicIO.inputs fp8 B p)) : ChannelValues ℝ (PublicIO.outputs fp8 B p) :=
  Fin.cases (fun j => ofFloat (KernelSource.outputDType fp8)
    (KernelMath.output p.K p.headDim p.start p.rotary
      (KernelReal.logical d8 fp8 (PublicIO.logical fp8 B p xs) j.1.val j.2.1.val)
      j.2.1.val j.2.2.1.val)) (fun k => k.elim0)

end
end VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard
