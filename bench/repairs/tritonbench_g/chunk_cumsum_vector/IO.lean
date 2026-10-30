import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.NativeIOLayout
import VeriTile.Triton.Float.ProgramEquivalenceSyntax
import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.RealProjection
import VeriTile.Triton.Memory.RegionalScheduledKernelIO
import VeriTile.Triton.Memory.LogicalScheduledKernelIO

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.FloatingIO
open VeriTile.Triton FP.Structural FPValues
noncomputable section
variable {α : Type} [Inhabited α]

/-- Actual typed source IO, including FP32 promotion, the TF32-disabled dot,
the complete time loop, original strides and output storage conversion. -/
abbrev prefixNativeIO := @NativeIO.io
abbrev sumLeftVariant := @NativeIO.candidate

/-- Only active carry updates consumed by a later output need finite operands.
This is the established numerical rewrite domain, independent of execution. -/
abbrev carryDomain := @NativeIO.finiteCarry

/-- Complete source invocation with the existing logical carry domain. -/
def original (M : Algebra α) (dot : DotBackend α) (D : FP.Guarded.Domain α)
    (h : Call) (c : Config) (p : Geometry.Program h) :=
  ProgramIO.Implementation.original (prefixNativeIO M dot h c p)
    (fun xs => carryDomain M D h c p (xs (⟨0,by decide⟩ : Fin 1)))

/-- Existing candidate swaps only the FP32 carry-add operands. -/
def sumLeft (M : Algebra α) (dot : DotBackend α) (D : FP.Guarded.Domain α)
    (h : Call) (c : Config) (p : Geometry.Program h) :=
  ProgramIO.Implementation.ofVariant (sumLeftVariant M dot h c p)
    (fun xs => carryDomain M D h c p (xs (⟨0,by decide⟩ : Fin 1)))

end
end VeriTile.Bench.Repairs.ChunkCumsumVector.FloatingIO

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.PublicRealIO
open VeriTile.Triton
noncomputable section

abbrev io (h : Call) (c : Config) (p : Geometry.Program h) : UKernelIO where
  kernel := (Kernels.source h c).eraseDType
  projection := sourceProjection h c
  nIn := 1
  nOut := 1
  nScr := 0
  bufs := [h.input,h.output]
  ity := fun _ => .float
  iarity := fun _ => h.T*BS
  ibuf := fun _ => h.input
  oty := fun _ => .float
  oarity := fun _ => h.T*BS
  obuf := fun _ => h.output
  obuf_mem := fun _ => by simp
  sarity := fun k => k.elim0
  sbuf := fun k => k.elim0
  iwin := fun _ _ _ _ _ j => h.inputStrides.address h.H p.row
    (j.val/BS) (p.feature*BS+j.val%BS)
  imask := fun _ _ _ _ _ j => p.feature*BS+j.val%BS < h.S
  owin := fun _ _ _ _ _ j => h.outputStrides.address h.H p.row
    (j.val/BS) (p.feature*BS+j.val%BS)
  omask := fun _ _ _ _ _ j => p.feature*BS+j.val%BS < h.S
  swin := fun k => k.elim0
  smask := fun k => k.elim0

def launch (h : Call) (p : Geometry.Program h) : AlgorithmLaunch where
  pids := fun ax => if ax=0 then p.feature else if ax=1 then p.row else 0
  setPids := fun ax => decide (ax<2)

end
end VeriTile.Bench.Repairs.ChunkCumsumVector.PublicRealIO

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.PublicRealIO
open VeriTile.Triton
noncomputable section

/-- Every time/feature value, including inactive tail features, is retained. -/
def matrixInputs (h : Call) (c : Config) (p : Geometry.Program h) :
    (io h c p).Ctx ≃ (Fin h.T → Fin BS → ℝ) where
  toFun := fun values time feature => values 0 (finProdFinEquiv (time, feature))
  invFun := fun xs _ j => xs (finProdFinEquiv.symm j).1 (finProdFinEquiv.symm j).2
  left_inv := by
    intro values
    funext channel j
    have zero : channel = 0 := Fin.eq_zero channel
    subst channel
    exact congrArg (values 0) (finProdFinEquiv.apply_symm_apply j)
  right_inv := by
    intro xs
    funext time feature
    change xs (finProdFinEquiv.symm (finProdFinEquiv (time, feature))).1
      (finProdFinEquiv.symm (finProdFinEquiv (time, feature))).2 = xs time feature
    rw [finProdFinEquiv.symm_apply_apply]

def matrixOutputs (h : Call) (c : Config) (p : Geometry.Program h) :
    ((o : Fin (io h c p).nOut) × Fin ((io h c p).oarity o)) ≃
      (Fin h.T × Fin BS) where
  toFun := fun cell => finProdFinEquiv.symm cell.2
  invFun := fun cell => ⟨0, finProdFinEquiv cell⟩
  left_inv := by
    rintro ⟨channel, j⟩
    have zero : channel = 0 := Fin.eq_zero channel
    subst channel
    change (⟨0, finProdFinEquiv (finProdFinEquiv.symm j)⟩ :
      (o : Fin 1) × Fin (h.T*BS)) = ⟨0, j⟩
    rw [finProdFinEquiv.apply_symm_apply]
  right_inv := fun _ => finProdFinEquiv.symm_apply_apply _

/-- The actual single feature-tile invocation runs the complete time loop. -/
def prefixLogicalIO (h : Call) (c : Config) (p : Geometry.Program h) :
    LogicalScheduledKernelIO (Fin h.T → Fin BS → ℝ) (Fin h.T × Fin BS) where
  core := io h c p
  schedule := fun _ => [launch h p]
  inputs := matrixInputs h c p
  indices := matrixOutputs h c p
  realOutput := fun _ => rfl

end
end VeriTile.Bench.Repairs.ChunkCumsumVector.PublicRealIO
