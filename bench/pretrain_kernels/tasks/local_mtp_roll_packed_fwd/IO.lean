import bench.pretrain_kernels.tasks.local_mtp_roll_packed_fwd.Kernels
import VeriTile.Triton.Memory.LogicalScheduledKernelIO
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Float.GuardedProgramIO
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Tactic.FinCases

namespace VeriTile.Bench.PretrainKernels.local_mtp_roll_packed_fwd
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 2000000
namespace IO

structure Inputs where
  input0 : Fin 2 → Fin 128 → ℝ
  constant53d3e3c65e : Fin 5 → Int

abbrev OutputIndex := Fin 2 × Fin 128

/-- Row-major indexing, with both inverses; no backing cells are dropped. -/
def matrixEquiv (m n : Nat) (α : Type) :
    (Fin (m*n) → α) ≃ (Fin m → Fin n → α) where
  toFun := fun f i j => f (finProdFinEquiv (i,j))
  invFun := fun f k => f (finProdFinEquiv.symm k).1 (finProdFinEquiv.symm k).2
  left_inv := by
    intro f
    funext k
    exact congrArg f (finProdFinEquiv.apply_symm_apply k)
  right_inv := by intro f; funext i j; simp

def input0Equiv : (Fin (2*128) → ℝ) ≃ (Fin 2 → Fin 128 → ℝ) :=
  ((matrixEquiv 2 (128) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 128 → ℝ))))

def constant53d3e3c65eEquiv : (Fin (5) → Int) ≃ (Fin 5 → Int) :=
  (Equiv.refl (Fin 5 → Int))

def core : UKernelIO where
  kernel := Kernels.original.eraseDType
  projection := by
    have h : Kernels.original.toAlgorithm? = .ok Kernels.original.toAlgKernel := rfl
    simp only [ComputeKernel.eraseDType, h]
    simp [ComputeKernel.toAlgKernel]
  nIn := 2
  nOut := 1
  nScr := 8
  bufs := ["input0", "constant53d3e3c65e", "out", "vcat", "vcat_1", "vcat_2", "vcat_3", "vcat_4", "candidate_vsub", "candidate_vremainder", "candidate_vindex_fill"]
  ity := Fin.cases (.float) (Fin.cases (.int) (Fin.elim0))
  iarity := Fin.cases (2*128) (Fin.cases (5) (Fin.elim0))
  ibuf := Fin.cases ("input0") (Fin.cases ("constant53d3e3c65e") (Fin.elim0))
  oty := fun _ => .float
  oarity := fun _ => 2*128
  obuf := fun _ => "out"
  obuf_mem := fun _ => by simp
  sarity := Fin.cases (2*16) (Fin.cases (2*32) (Fin.cases (2*32) (Fin.cases (2*48) (Fin.cases (2*128) (Fin.cases (4) (Fin.cases (4) (Fin.cases (2*128) (Fin.elim0))))))))
  sbuf := Fin.cases ("vcat") (Fin.cases ("vcat_1") (Fin.cases ("vcat_2") (Fin.cases ("vcat_3") (Fin.cases ("vcat_4") (Fin.cases ("candidate_vsub") (Fin.cases ("candidate_vremainder") (Fin.cases ("candidate_vindex_fill") (Fin.elim0))))))))
  iwin := fun _ _ _ _ _ j => j.val
  imask := fun _ _ _ _ _ _ => True
  owin := fun _ _ _ _ _ j => j.val
  omask := fun _ _ _ _ _ _ => True
  swin := fun _ _ _ _ _ j => j.val
  smask := fun _ _ _ _ _ _ => True

def inputs : core.Ctx ≃ Inputs where
  toFun := fun v => ⟨input0Equiv (v (0 : Fin 2)), constant53d3e3c65eEquiv (v (1 : Fin 2))⟩
  invFun := fun v => Fin.cases (input0Equiv.symm v.input0) (Fin.cases (constant53d3e3c65eEquiv.symm v.constant53d3e3c65e) ((fun i => Fin.elim0 i)))
  left_inv := by
    intro v
    funext i
    fin_cases i <;> exact Equiv.symm_apply_apply _ _
  right_inv := by
    intro v
    change (⟨input0Equiv (input0Equiv.symm v.input0), constant53d3e3c65eEquiv (constant53d3e3c65eEquiv.symm v.constant53d3e3c65e)⟩ : Inputs) = v
    simp only [Equiv.apply_symm_apply]

def outputCoords : Fin (2*128) ≃ OutputIndex :=
  (finProdFinEquiv (m := 2) (n := 128)).symm

def indices : ((o : Fin core.nOut) × Fin (core.oarity o)) ≃ OutputIndex where
  toFun := fun p => outputCoords p.2
  invFun := fun p => ⟨(0 : Fin 1), outputCoords.symm p⟩
  left_inv := by
    intro ⟨o,j⟩
    have ho : o = (0 : Fin 1) := Fin.eq_zero o
    subst o
    simp
  right_inv := by intro j; simp

/-- The source computes the complete declared problem in one launch. -/
def realIO : LogicalScheduledKernelIO Inputs OutputIndex where
  core := core
  schedule := fun _ => [{ pids := fun _ => 0 }]
  inputs := inputs
  indices := indices
  realOutput := fun _ => rfl

namespace FP
open scoped BigOperators Classical
open VeriTile.Triton.FP.Structural

/-- fp32 loads use the project's real carrier with an explicit fp32 tag;
narrow storage and discrete inputs retain their own cell tags. -/
def inputStorage : Fin core.nIn → TileDType :=
  Fin.cases (.bf16) (Fin.cases (.int) (Fin.elim0))

def scratchStorage : Fin core.nScr → TileDType :=
  Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.bf16) (Fin.elim0))))))))

/-- The same complete channel layout; dtype casts remain in the source. -/
def io (α : Type) : ProgramIO α α where
  inputs := List.ofFn fun i : Fin core.nIn => ⟨inputStorage i, [core.iarity i], core.ibuf i⟩
  outputs := List.ofFn fun o : Fin core.nOut => ⟨.bf16, [core.oarity o], core.obuf o⟩
  scratch := List.ofFn fun k : Fin core.nScr => ⟨scratchStorage k, [core.sarity k], core.sbuf k⟩
  buffers := core.bufs
  read := fun _ _ j => TileShape.linearIndex _ j
  inputMask := fun _ _ _ => True
  write := fun _ _ j => TileShape.linearIndex _ j
  outputMask := fun _ _ _ => True
  scratchWrite := fun _ _ j => TileShape.linearIndex _ j
  scratchMask := fun _ _ _ => True
  schedule := fun _ => []
  encode := id
  observe := fun _ _ => id

/-- Required discrete input conditions, independent of numerical rewrite guards. -/
def admissible {α : Type} (vals : (io α).Inputs) : Prop :=
  (∀ j : Fin 5, vals (1 : Fin 2) (j, PUnit.unit) = ([0, 16, 48, 80, 128] : List Int).getD j.val 0)

end FP
end IO
end
end VeriTile.Bench.PretrainKernels.local_mtp_roll_packed_fwd
