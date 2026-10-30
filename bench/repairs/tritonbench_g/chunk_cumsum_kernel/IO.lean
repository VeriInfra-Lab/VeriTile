import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Proofs.CarryDomain
import VeriTile.Triton.Memory.RegionalKernelIO
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Float.ProgramEquivalenceSyntax
import VeriTile.Triton.Memory.LogicalIO
import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Proofs.SourceProjection

namespace VeriTile.Bench.Repairs.ChunkCumsumKernel.KernelIO
open VeriTile.Triton
noncomputable section

abbrev realIO (input output : RegionName) (T : Nat) (cfg : Config)
    (projectionProof : (repairedKernel .fp32 .fp32 input output T cfg.block).eraseDType.toAlgorithm? =
      .ok (repairedKernel .fp32 .fp32 input output T cfg.block).eraseDType.toAlgKernel) : UKernelIO where
  kernel := (repairedKernel .fp32 .fp32 input output T cfg.block).eraseDType
  projection := projectionProof
  nIn := 1
  nOut := 1
  nScr := 0
  bufs := [input,output]
  ity := fun _ => .float
  iarity := fun _ => T
  ibuf := fun _ => input
  oty := fun _ => .float
  oarity := fun _ => T
  obuf := fun _ => output
  obuf_mem := fun _ => by simp
  sarity := fun z => Fin.elim0 z
  sbuf := fun z => Fin.elim0 z
  iwin := fun _ _ pid _ _ j => pid*T+j.val
  imask := fun _ _ _ _ _ _ => True
  owin := fun _ _ pid _ _ j => pid*T+j.val
  omask := fun _ _ _ _ _ _ => True
  swin := fun z => Fin.elim0 z
  smask := fun z => Fin.elim0 z

end
end VeriTile.Bench.Repairs.ChunkCumsumKernel.KernelIO

namespace VeriTile.Bench.Repairs.ChunkCumsumKernel.PublicFP
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section

abbrev io {α : Type} (ops : FPRow.Ops α) (input output : RegionName)
    (T : Nat) (cfg : Config) (pid : Nat) : ProgramIO α α where
  inputs := [⟨.fp32,[T],input⟩]
  outputs := [⟨.fp32,[T],output⟩]
  scratch := []
  buffers := [input,output]
  read := fun i => Fin.cases (fun _ j => pid*T+j.1.val) (fun z => Fin.elim0 z) i
  inputMask := fun _ _ _ => True
  write := fun o => Fin.cases (fun _ j => pid*T+j.1.val) (fun z => Fin.elim0 z) o
  outputMask := fun _ _ _ => True
  scratchWrite := fun z => Fin.elim0 z
  scratchMask := fun z => Fin.elim0 z
  schedule := fun _ => [{
    kernel := repairedKernel .fp32 .fp32 input output T cfg.block
    backend := engine ops.algebra
    pids := fun ax => if ax=0 then pid else 0 }]
  encode := id
  observe := fun _ _ => id

abbrev candidate {α : Type} (ops : FPRow.Ops α) (input output : RegionName)
    (T : Nat) (cfg : Config) (pid : Nat) : ProgramIO.Variant (io ops input output T cfg pid) where
  schedule := fun _ => [{
    kernel := sumLeftKernel .fp32 .fp32 input output T cfg.block
    backend := engine ops.algebra
    pids := fun ax => if ax=0 then pid else 0 }]
  observe := fun _ _ => id

/-- Outside-row extension is irrelevant to the guarded full chunks. -/
def row {α : Type} [Inhabited α] {T : Nat} (xs : TileIndex [T] → α) (j : Nat) : α :=
  if h : j<T then xs (⟨j,h⟩,()) else default

def domain {α : Type} [Inhabited α] (ops : FPRow.Ops α) (D : FP.Guarded.Domain α)
    (input output : RegionName) (T : Nat) (cfg : Config) (pid : Nat)
    (vals : (io ops input output T cfg pid).Inputs) : Prop :=
  CarryDomain.Finite ops D T cfg.block (row (α := α) (vals 0))

abbrev start {α : Type} (pid : Nat) (s : State α) : State α :=
  {s with pids := fun ax => if ax=0 then pid else 0}

end
end VeriTile.Bench.Repairs.ChunkCumsumKernel.PublicFP

namespace VeriTile.Bench.Repairs.ChunkCumsumKernel.FPEquiv
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section

/-- Actual FP32 input/output channels and complete source loop. Read and
write windows coincide for in-place calls; every admitted word is retained. -/
abbrev originalIO {α : Type} (ops : FPRow.Ops α) (input output : RegionName)
    (T : Nat) (cfg : Config) (pid : Nat) : ProgramIO α α :=
  PublicFP.io ops input output T cfg pid

/-- The displayed candidate swaps only the FP32 carry-add operands. -/
abbrev sumLeftIO {α : Type} (ops : FPRow.Ops α) (input output : RegionName)
    (T : Nat) (cfg : Config) (pid : Nat) : ProgramIO.Variant (originalIO ops input output T cfg pid) :=
  PublicFP.candidate ops input output T cfg pid

/-- Complete original source schedule on the unchanged independent carry domain. -/
abbrev original {α : Type} [Inhabited α] (ops : FPRow.Ops α) (D : FP.Guarded.Domain α)
    (input output : RegionName) (T : Nat) (cfg : Config) (pid : Nat) :=
  ProgramIO.Implementation.original (originalIO ops input output T cfg pid)
    (PublicFP.domain ops D input output T cfg pid)

/-- Existing sum-left source with the same FP32 IO, observer and carry domain. -/
abbrev sumLeft {α : Type} [Inhabited α] (ops : FPRow.Ops α) (D : FP.Guarded.Domain α)
    (input output : RegionName) (T : Nat) (cfg : Config) (pid : Nat) :=
  ProgramIO.Implementation.ofVariant (sumLeftIO ops input output T cfg pid)
    (PublicFP.domain ops D input output T cfg pid)

end
end VeriTile.Bench.Repairs.ChunkCumsumKernel.FPEquiv

namespace VeriTile.Bench.Repairs.ChunkCumsumKernel.PublicReal
open VeriTile.Triton
noncomputable section

abbrev io (input output : RegionName) (T : Nat) (cfg : Config) : UKernelIO :=
  KernelIO.realIO input output T cfg (projection input output T cfg.block)

end
end VeriTile.Bench.Repairs.ChunkCumsumKernel.PublicReal

namespace VeriTile.Bench.Repairs.ChunkCumsumKernel.PublicReal
open VeriTile.Triton
noncomputable section

def rowValues (T : Nat) : (Fin 1 → Fin T → ℝ) ≃ (Fin T → ℝ) where
  toFun := fun vals => vals 0
  invFun := fun xs _ => xs
  left_inv := by
    intro vals
    funext i
    have zero : i=0 := Fin.eq_zero i
    subst i
    rfl
  right_inv := fun _ => rfl

/-- The complete time loop at every original row/program coordinate; the
regional input/output windows and exact exterior frame remain unchanged. -/
def prefixLogicalIO (input output : RegionName) (T : Nat) (cfg : Config) :
    LogicalIO (io input output T cfg).Ctx (Fin 1 → Fin T → ℝ)
      (Fin T → ℝ) (Fin T → ℝ)
      (fun f => (io input output T cfg).RegionalImplements (fun _ _ => f)) where
  inputs := rowValues T
  outputs := rowValues T

end
end VeriTile.Bench.Repairs.ChunkCumsumKernel.PublicReal
