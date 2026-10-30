import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.seeded_dropout.Kernels
import VeriTile.Triton.Float.ProgramEquivalenceSyntax
import VeriTile.Triton.Float.ProfiledRewrite

namespace VeriTile.Bench.Repairs.SeededDropout.KernelPublic
open VeriTile.Triton FP.Structural
noncomputable section

/-- The sole input/output channel is one logical source tile. The maps retain
all 1024 lanes, including the inactive tail; masking remains in the core IO. -/
def tileValues {α : Type} {X : RegionName} :
    ChannelValues α [⟨.real,[1024],X⟩] ≃ (Fin 1024 → α) where
  toFun := fun values j => values 0 (j,())
  invFun := fun xs => Fin.cases (fun j => xs j.1) (fun i => i.elim0)
  left_inv := by
    intro values
    funext channel j
    have zero : channel=0 := Fin.eq_zero channel
    subst channel
    exact congrArg (values 0) (Prod.ext rfl (Subsingleton.elim _ _))
  right_inv := fun _ => rfl

abbrev io {α : Type} (M : Algebra α) (k : SeedKind) (X O : RegionName)
    (n : Nat) (p : ℝ) (seed : SeedCarrier k) (pid : Nat) : ProgramIO α α where
  inputs := [⟨.real,[1024],X⟩]
  outputs := [⟨.real,[1024],O⟩]
  scratch := []
  buffers := [X,O]
  read := fun i => Fin.cases (fun _ j => pid*1024+j.1.val) (fun z => Fin.elim0 z) i
  inputMask := fun i => Fin.cases (fun _ j => pid*1024+j.1.val<n) (fun z => Fin.elim0 z) i
  write := fun o => Fin.cases (fun _ j => pid*1024+j.1.val) (fun z => Fin.elim0 z) o
  outputMask := fun o => Fin.cases (fun _ j => pid*1024+j.1.val<n) (fun z => Fin.elim0 z) o
  scratchWrite := fun z => Fin.elim0 z
  scratchMask := fun z => Fin.elim0 z
  schedule := fun _ => [{
    kernel := kernel k X O n p seed
    backend := M
    pids := fun ax => if ax=0 then pid else 0 }]
  encode := id
  observe := fun _ _ => id

abbrev variant {α : Type} (M : Algebra α) (k : SeedKind) (X O : RegionName)
    (n : Nat) (p : ℝ) (seed : SeedCarrier k) (pid : Nat) :
    ProgramIO.Variant (io M k X O n p seed pid) where
  schedule := fun _ => [{
    kernel := indexCommutedKernel k X O n p seed
    backend := M
    pids := fun ax => if ax=0 then pid else 0 }]
  observe := fun _ _ => id


/-- Logical values of the complete original source tile; all ABI and masks are
those of `io`, including inactive lanes and its exact writable frame. -/
def logicalIO {α : Type} (M : Algebra α) (k : SeedKind) (X O : RegionName)
    (n : Nat) (p : ℝ) (seed : SeedCarrier k) (pid : Nat) :
    LogicalProgramIO α α (Fin 1024 → α) (Fin 1024 → α) where
  core := io M k X O n p seed pid
  inputs := tileValues
  outputs := tileValues

end
end VeriTile.Bench.Repairs.SeededDropout.KernelPublic

namespace VeriTile.Bench.Repairs.SeededDropout.FPEquiv
open VeriTile.Triton FP.Structural
noncomputable section

/-- Actual selected FP32 source and its unchanged active memory interface. -/
abbrev originalIO {α : Type} (M : Algebra α) (k : SeedKind) (X O : RegionName)
    (n : Nat) (p : ℝ) (seed : SeedCarrier k) (pid : Nat) : ProgramIO α α :=
  KernelPublic.io (FP.ProfiledRewrite.engine M) k X O n p seed pid

/-- The displayed candidate commutes only the natural block-index product. -/
abbrev commutedIO {α : Type} (M : Algebra α) (k : SeedKind) (X O : RegionName)
    (n : Nat) (p : ℝ) (seed : SeedCarrier k) (pid : Nat) :
    ProgramIO.Variant (originalIO M k X O n p seed pid) :=
  KernelPublic.variant (FP.ProfiledRewrite.engine M) k X O n p seed pid

/-- Complete selected source with the original active IO and no value guard. -/
abbrev original {α : Type} (M : Algebra α) (k : SeedKind) (X O : RegionName)
    (n : Nat) (p : ℝ) (seed : SeedCarrier k) (pid : Nat) :=
  ProgramIO.Implementation.original (originalIO M k X O n p seed pid) (fun _ => True)

/-- Existing index-commuted source with exactly the same active observation. -/
abbrev indexCommuted {α : Type} (M : Algebra α) (k : SeedKind) (X O : RegionName)
    (n : Nat) (p : ℝ) (seed : SeedCarrier k) (pid : Nat) :=
  ProgramIO.Implementation.ofVariant (commutedIO M k X O n p seed pid) (fun _ => True)

end
end VeriTile.Bench.Repairs.SeededDropout.FPEquiv
