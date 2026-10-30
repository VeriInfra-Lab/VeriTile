import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.kcache_copy_triton.Kernels
import VeriTile.Triton.Float.ProgramEquivalenceSyntax
import VeriTile.Triton.Float.ProfiledRewrite

namespace VeriTile.Bench.Repairs.KcacheCopyTriton.CopyIO
open VeriTile.Triton FP.Structural Source
noncomputable section

/-- Independent signed metadata routing, before any kernel execution. -/
def sequenceIndex (p : Parameters) (ids : Nat → Nat) : Int := (ids 0 : Int).tdiv p.n
def position (p : Parameters) (ids : Nat → Nat) (length : Int) : Int :=
  length + ((ids 0 : Int) - (p.n : Int) * (sequenceIndex p ids + 1))
def tableIndex (p : Parameters) (ids : Nat → Nat) (length : Int) : Nat :=
  (sequenceIndex p ids * p.strideBts).toNat +
    ((position p ids length).tdiv p.blockSize * p.strideBtb).toNat
def inputIndex (p : Parameters) (ids : Nat → Nat) (i : TileIndex [p.width]) : Nat :=
  ids 0*p.strideKt + ids 1*p.strideKh + (ids 2*p.width+i.1.val)*p.strideKd
def outputIndex (p : Parameters) (ids : Nat → Nat) (length block : Int)
    (i : TileIndex [p.width]) : Nat :=
  (block*p.strideKcb + (ids 1 : Int)*p.strideKch + (ids 2 : Int)*p.strideKcsplit +
    (position p ids length).tmod p.blockSize*p.strideKcs+i.1.val).toNat

abbrev channels (p : Parameters) : List IOChannel :=
  [⟨.int,[],p.lengths.cast⟩,⟨.int,[],p.blockTables.cast⟩,⟨.real,[p.width],p.K⟩]
abbrev Inputs (α : Type) (p : Parameters) := ChannelValues α (channels p)

abbrev io {α : Type} (M : Algebra α) (v : Bool) (p : Parameters)
    (ids : Nat → Nat) : ProgramIO α α where
  inputs := channels p
  outputs := [⟨.real,[p.width],p.KCache⟩]
  scratch := []
  buffers := [p.lengths.cast,p.blockTables.cast,p.K,p.KCache]
  read := fun i => Fin.cases (fun _ _ => (sequenceIndex p ids).toNat)
    (Fin.cases (fun vals _ => tableIndex p ids (vals 0 ()))
      (Fin.cases (fun _ j => inputIndex p ids j) (fun z => Fin.elim0 z))) i
  inputMask := fun _ _ _ => True
  write := fun o => Fin.cases (fun vals j => outputIndex p ids (vals 0 ()) (vals 1 ()) j)
    (fun z => Fin.elim0 z) o
  outputMask := fun _ _ _ => True
  scratchWrite := fun z => Fin.elim0 z
  scratchMask := fun z => Fin.elim0 z
  schedule := fun _ => [{ kernel := kernel v p, backend := M, pids := ids }]
  encode := id
  observe := fun _ _ => id

abbrev start {α : Type} (M : Algebra α) (v : Bool) (p : Parameters)
    (ids : Nat → Nat) (s : State α) : State α :=
  ({ kernel := kernel v p, backend := M, pids := ids } : Invocation α).initial s

abbrev candidate (M : Algebra α) (p : Parameters) (ids : Nat → Nat) :
    ProgramIO.Variant (io M Bool.false p ids) where
  schedule := fun _ => [{ kernel := kernel Bool.true p, backend := M, pids := ids }]
  observe := fun _ _ => id

/-- This source uses only integer arithmetic and FP32 payload loads/stores.
The sole reached floating primitive is the mathematical identity load. -/
def realBackend : Algebra ℝ where
  literal := fun _ _ x => x
  negInf := 0
  binary := fun _ _ op x y => match op with
    | .add => x+y | .sub => x-y | .mul => x*y | .div => x/y | _ => 0
  unary := fun _ _ x => x
  cast := fun _ _ _ x => x
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => 0
  reduceSum := fun _ {_} _ _ _ _ => 0

end
end VeriTile.Bench.Repairs.KcacheCopyTriton.CopyIO

namespace VeriTile.Bench.Repairs.KcacheCopyTriton.FPEquiv
open VeriTile.Triton FP.Structural
noncomputable section

/-- Selected FP32 source and its complete metadata-dependent IO. Payloads and
metadata keep their original types; no finite-word restriction is introduced. -/
abbrev originalIO {α : Type} (M : Algebra α) (p : Source.Parameters)
    (ids : Nat → Nat) : ProgramIO α α :=
  CopyIO.io (FP.ProfiledRewrite.engine M) Bool.false p ids

/-- Only the two scalar source-address summands are commuted. The data and
metadata windows, masks, precision and all program-ID axes are unchanged. -/
abbrev commutedIO {α : Type} (M : Algebra α) (p : Source.Parameters)
    (ids : Nat → Nat) : ProgramIO.Variant (originalIO M p ids) where
  schedule := fun _ => [{
    kernel := Source.commutedKernel p
    backend := FP.ProfiledRewrite.engine M
    pids := ids }]
  observe := fun _ _ => id

/-- Actual selected source schedule on its fixed typed memory interface. -/
def original {α : Type} (M : Algebra α) (p : Source.Parameters) (ids : Nat → Nat) :
    ProgramIO.Implementation (originalIO M p ids) (fun _ => True) :=
  ProgramIO.Implementation.original (originalIO M p ids) (fun _ => True)

/-- Existing address-commuted source schedule on the identical public IO. -/
def candidate {α : Type} (M : Algebra α) (p : Source.Parameters) (ids : Nat → Nat) :
    ProgramIO.Implementation (originalIO M p ids) (fun _ => True) :=
  ProgramIO.Implementation.ofVariant (commutedIO M p ids) (fun _ => True)

end
end VeriTile.Bench.Repairs.KcacheCopyTriton.FPEquiv

namespace VeriTile.Bench.Repairs.KcacheCopyTriton.CopyIO
open VeriTile.Triton FP.Structural Source
noncomputable section
structure LogicalInputs (α : Type) (p : Parameters) where
  length : Int
  selectedBlock : Int
  payload : Fin p.width → α

def logicalInputs (α : Type) (p : Parameters) : Inputs α p ≃ LogicalInputs α p where
  toFun xs := ⟨xs 0 (),xs 1 (),fun i => xs 2 (i,())⟩
  invFun xs := fun o => match o with
    | ⟨0,_⟩ => fun _ => xs.length
    | ⟨1,_⟩ => fun _ => xs.selectedBlock
    | ⟨2,_⟩ => fun i => xs.payload i.1
  left_inv := by
    intro xs; funext o; fin_cases o
    · funext j; cases j; rfl
    · funext j; cases j; rfl
    · funext j; rcases j with ⟨i,⟨⟩⟩; rfl
  right_inv := by intro xs; cases xs; rfl

def logicalOutputs (α : Type) (p : Parameters) : ChannelValues α [⟨.real,[p.width],p.KCache⟩] ≃ (Fin p.width → α) where
  toFun xs := fun i => xs 0 (i,())
  invFun xs := fun o => match o with | ⟨0,_⟩ => fun i => xs i.1
  left_inv := by intro xs; funext o; fin_cases o; funext i; rcases i with ⟨i,⟨⟩⟩; rfl
  right_inv := by intro xs; rfl
end
end VeriTile.Bench.Repairs.KcacheCopyTriton.CopyIO

namespace VeriTile.Bench.Repairs.KcacheCopyTriton.CopyIO
open VeriTile.Triton FP.Structural
noncomputable section
open Source
def logicalIO (M : Algebra α) (v : Bool) (p : Parameters) (ids : Nat → Nat) : LogicalProgramIO α α (LogicalInputs α p) (Fin p.width → α) where
  core := io M v p ids
  inputs := logicalInputs α p
  outputs := logicalOutputs α p
end
end VeriTile.Bench.Repairs.KcacheCopyTriton.CopyIO

namespace VeriTile.Bench.Repairs.KcacheCopyTriton.CopyIO
open VeriTile.Triton FP.Structural Source
noncomputable section
def logicalOutputInjective (p : Parameters) (ids : Nat → Nat) (xs : LogicalInputs ℝ p) : Prop :=
  Function.Injective (fun i : TileIndex [p.width] => outputIndex p ids xs.length xs.selectedBlock i)
end
end VeriTile.Bench.Repairs.KcacheCopyTriton.CopyIO
