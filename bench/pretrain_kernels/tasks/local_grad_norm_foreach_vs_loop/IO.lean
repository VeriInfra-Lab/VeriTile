import bench.pretrain_kernels.tasks.local_grad_norm_foreach_vs_loop.Kernels
import VeriTile.Triton.Memory.LogicalScheduledKernelIO
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Float.GuardedProgramIO
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Tactic.FinCases

namespace VeriTile.Bench.PretrainKernels.local_grad_norm_foreach_vs_loop
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 2000000
namespace IO

structure Inputs where
  input0 : Fin 32 → Fin 512 → ℝ

abbrev OutputIndex := Fin 32

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

def input0Equiv : (Fin (32*512) → ℝ) ≃ (Fin 32 → Fin 512 → ℝ) :=
  ((matrixEquiv 32 (512) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 512 → ℝ))))

def core : UKernelIO where
  kernel := Kernels.original.eraseDType
  projection := by
    have h : Kernels.original.toAlgorithm? = .ok Kernels.original.toAlgKernel := rfl
    simp only [ComputeKernel.eraseDType, h]
    simp [ComputeKernel.toAlgKernel]
  nIn := 1
  nOut := 1
  nScr := 133
  bufs := ["input0", "out", "vto_copy", "vlinalg_vector_norm", "vto_copy_1", "vlinalg_vector_norm_1", "vto_copy_2", "vlinalg_vector_norm_2", "vto_copy_3", "vlinalg_vector_norm_3", "vto_copy_4", "vlinalg_vector_norm_4", "vto_copy_5", "vlinalg_vector_norm_5", "vto_copy_6", "vlinalg_vector_norm_6", "vto_copy_7", "vlinalg_vector_norm_7", "vto_copy_8", "vlinalg_vector_norm_8", "vto_copy_9", "vlinalg_vector_norm_9", "vto_copy_10", "vlinalg_vector_norm_10", "vto_copy_11", "vlinalg_vector_norm_11", "vto_copy_12", "vlinalg_vector_norm_12", "vto_copy_13", "vlinalg_vector_norm_13", "vto_copy_14", "vlinalg_vector_norm_14", "vto_copy_15", "vlinalg_vector_norm_15", "vto_copy_16", "vlinalg_vector_norm_16", "vto_copy_17", "vlinalg_vector_norm_17", "vto_copy_18", "vlinalg_vector_norm_18", "vto_copy_19", "vlinalg_vector_norm_19", "vto_copy_20", "vlinalg_vector_norm_20", "vto_copy_21", "vlinalg_vector_norm_21", "vto_copy_22", "vlinalg_vector_norm_22", "vto_copy_23", "vlinalg_vector_norm_23", "vto_copy_24", "vlinalg_vector_norm_24", "vto_copy_25", "vlinalg_vector_norm_25", "vto_copy_26", "vlinalg_vector_norm_26", "vto_copy_27", "vlinalg_vector_norm_27", "vto_copy_28", "vlinalg_vector_norm_28", "vto_copy_29", "vlinalg_vector_norm_29", "vto_copy_30", "vlinalg_vector_norm_30", "vto_copy_31", "vlinalg_vector_norm_31", "vstack", "vto_copy_32", "candidate_vto_copy", "candidate_vto_copy_1", "candidate_vto_copy_2", "candidate_vto_copy_3", "candidate_vto_copy_4", "candidate_vto_copy_5", "candidate_vto_copy_6", "candidate_vto_copy_7", "candidate_vto_copy_8", "candidate_vto_copy_9", "candidate_vto_copy_10", "candidate_vto_copy_11", "candidate_vto_copy_12", "candidate_vto_copy_13", "candidate_vto_copy_14", "candidate_vto_copy_15", "candidate_vto_copy_16", "candidate_vto_copy_17", "candidate_vto_copy_18", "candidate_vto_copy_19", "candidate_vto_copy_20", "candidate_vto_copy_21", "candidate_vto_copy_22", "candidate_vto_copy_23", "candidate_vto_copy_24", "candidate_vto_copy_25", "candidate_vto_copy_26", "candidate_vto_copy_27", "candidate_vto_copy_28", "candidate_vto_copy_29", "candidate_vto_copy_30", "candidate_vto_copy_31", "candidate_vlinalg_vector_norm", "candidate_vlinalg_vector_norm_1", "candidate_vlinalg_vector_norm_2", "candidate_vlinalg_vector_norm_3", "candidate_vlinalg_vector_norm_4", "candidate_vlinalg_vector_norm_5", "candidate_vlinalg_vector_norm_6", "candidate_vlinalg_vector_norm_7", "candidate_vlinalg_vector_norm_8", "candidate_vlinalg_vector_norm_9", "candidate_vlinalg_vector_norm_10", "candidate_vlinalg_vector_norm_11", "candidate_vlinalg_vector_norm_12", "candidate_vlinalg_vector_norm_13", "candidate_vlinalg_vector_norm_14", "candidate_vlinalg_vector_norm_15", "candidate_vlinalg_vector_norm_16", "candidate_vlinalg_vector_norm_17", "candidate_vlinalg_vector_norm_18", "candidate_vlinalg_vector_norm_19", "candidate_vlinalg_vector_norm_20", "candidate_vlinalg_vector_norm_21", "candidate_vlinalg_vector_norm_22", "candidate_vlinalg_vector_norm_23", "candidate_vlinalg_vector_norm_24", "candidate_vlinalg_vector_norm_25", "candidate_vlinalg_vector_norm_26", "candidate_vlinalg_vector_norm_27", "candidate_vlinalg_vector_norm_28", "candidate_vlinalg_vector_norm_29", "candidate_vlinalg_vector_norm_30", "candidate_vlinalg_vector_norm_31", "candidate_vstack", "candidate_vto_copy_32", "candidate_vcat"]
  ity := Fin.cases (.float) (Fin.elim0)
  iarity := Fin.cases (32*512) (Fin.elim0)
  ibuf := Fin.cases ("input0") (Fin.elim0)
  oty := fun _ => .float
  oarity := fun _ => 32
  obuf := fun _ => "out"
  obuf_mem := fun _ => by simp
  sarity := Fin.cases (512) (Fin.cases (1) (Fin.cases (512) (Fin.cases (1) (Fin.cases (512) (Fin.cases (1) (Fin.cases (512) (Fin.cases (1) (Fin.cases (512) (Fin.cases (1) (Fin.cases (512) (Fin.cases (1) (Fin.cases (512) (Fin.cases (1) (Fin.cases (512) (Fin.cases (1) (Fin.cases (512) (Fin.cases (1) (Fin.cases (512) (Fin.cases (1) (Fin.cases (512) (Fin.cases (1) (Fin.cases (512) (Fin.cases (1) (Fin.cases (512) (Fin.cases (1) (Fin.cases (512) (Fin.cases (1) (Fin.cases (512) (Fin.cases (1) (Fin.cases (512) (Fin.cases (1) (Fin.cases (512) (Fin.cases (1) (Fin.cases (512) (Fin.cases (1) (Fin.cases (512) (Fin.cases (1) (Fin.cases (512) (Fin.cases (1) (Fin.cases (512) (Fin.cases (1) (Fin.cases (512) (Fin.cases (1) (Fin.cases (512) (Fin.cases (1) (Fin.cases (512) (Fin.cases (1) (Fin.cases (512) (Fin.cases (1) (Fin.cases (512) (Fin.cases (1) (Fin.cases (512) (Fin.cases (1) (Fin.cases (512) (Fin.cases (1) (Fin.cases (512) (Fin.cases (1) (Fin.cases (512) (Fin.cases (1) (Fin.cases (512) (Fin.cases (1) (Fin.cases (512) (Fin.cases (1) (Fin.cases (32) (Fin.cases (32) (Fin.cases (512) (Fin.cases (512) (Fin.cases (512) (Fin.cases (512) (Fin.cases (512) (Fin.cases (512) (Fin.cases (512) (Fin.cases (512) (Fin.cases (512) (Fin.cases (512) (Fin.cases (512) (Fin.cases (512) (Fin.cases (512) (Fin.cases (512) (Fin.cases (512) (Fin.cases (512) (Fin.cases (512) (Fin.cases (512) (Fin.cases (512) (Fin.cases (512) (Fin.cases (512) (Fin.cases (512) (Fin.cases (512) (Fin.cases (512) (Fin.cases (512) (Fin.cases (512) (Fin.cases (512) (Fin.cases (512) (Fin.cases (512) (Fin.cases (512) (Fin.cases (512) (Fin.cases (512) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (32) (Fin.cases (32) (Fin.cases (32) (Fin.elim0)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
  sbuf := Fin.cases ("vto_copy") (Fin.cases ("vlinalg_vector_norm") (Fin.cases ("vto_copy_1") (Fin.cases ("vlinalg_vector_norm_1") (Fin.cases ("vto_copy_2") (Fin.cases ("vlinalg_vector_norm_2") (Fin.cases ("vto_copy_3") (Fin.cases ("vlinalg_vector_norm_3") (Fin.cases ("vto_copy_4") (Fin.cases ("vlinalg_vector_norm_4") (Fin.cases ("vto_copy_5") (Fin.cases ("vlinalg_vector_norm_5") (Fin.cases ("vto_copy_6") (Fin.cases ("vlinalg_vector_norm_6") (Fin.cases ("vto_copy_7") (Fin.cases ("vlinalg_vector_norm_7") (Fin.cases ("vto_copy_8") (Fin.cases ("vlinalg_vector_norm_8") (Fin.cases ("vto_copy_9") (Fin.cases ("vlinalg_vector_norm_9") (Fin.cases ("vto_copy_10") (Fin.cases ("vlinalg_vector_norm_10") (Fin.cases ("vto_copy_11") (Fin.cases ("vlinalg_vector_norm_11") (Fin.cases ("vto_copy_12") (Fin.cases ("vlinalg_vector_norm_12") (Fin.cases ("vto_copy_13") (Fin.cases ("vlinalg_vector_norm_13") (Fin.cases ("vto_copy_14") (Fin.cases ("vlinalg_vector_norm_14") (Fin.cases ("vto_copy_15") (Fin.cases ("vlinalg_vector_norm_15") (Fin.cases ("vto_copy_16") (Fin.cases ("vlinalg_vector_norm_16") (Fin.cases ("vto_copy_17") (Fin.cases ("vlinalg_vector_norm_17") (Fin.cases ("vto_copy_18") (Fin.cases ("vlinalg_vector_norm_18") (Fin.cases ("vto_copy_19") (Fin.cases ("vlinalg_vector_norm_19") (Fin.cases ("vto_copy_20") (Fin.cases ("vlinalg_vector_norm_20") (Fin.cases ("vto_copy_21") (Fin.cases ("vlinalg_vector_norm_21") (Fin.cases ("vto_copy_22") (Fin.cases ("vlinalg_vector_norm_22") (Fin.cases ("vto_copy_23") (Fin.cases ("vlinalg_vector_norm_23") (Fin.cases ("vto_copy_24") (Fin.cases ("vlinalg_vector_norm_24") (Fin.cases ("vto_copy_25") (Fin.cases ("vlinalg_vector_norm_25") (Fin.cases ("vto_copy_26") (Fin.cases ("vlinalg_vector_norm_26") (Fin.cases ("vto_copy_27") (Fin.cases ("vlinalg_vector_norm_27") (Fin.cases ("vto_copy_28") (Fin.cases ("vlinalg_vector_norm_28") (Fin.cases ("vto_copy_29") (Fin.cases ("vlinalg_vector_norm_29") (Fin.cases ("vto_copy_30") (Fin.cases ("vlinalg_vector_norm_30") (Fin.cases ("vto_copy_31") (Fin.cases ("vlinalg_vector_norm_31") (Fin.cases ("vstack") (Fin.cases ("vto_copy_32") (Fin.cases ("candidate_vto_copy") (Fin.cases ("candidate_vto_copy_1") (Fin.cases ("candidate_vto_copy_2") (Fin.cases ("candidate_vto_copy_3") (Fin.cases ("candidate_vto_copy_4") (Fin.cases ("candidate_vto_copy_5") (Fin.cases ("candidate_vto_copy_6") (Fin.cases ("candidate_vto_copy_7") (Fin.cases ("candidate_vto_copy_8") (Fin.cases ("candidate_vto_copy_9") (Fin.cases ("candidate_vto_copy_10") (Fin.cases ("candidate_vto_copy_11") (Fin.cases ("candidate_vto_copy_12") (Fin.cases ("candidate_vto_copy_13") (Fin.cases ("candidate_vto_copy_14") (Fin.cases ("candidate_vto_copy_15") (Fin.cases ("candidate_vto_copy_16") (Fin.cases ("candidate_vto_copy_17") (Fin.cases ("candidate_vto_copy_18") (Fin.cases ("candidate_vto_copy_19") (Fin.cases ("candidate_vto_copy_20") (Fin.cases ("candidate_vto_copy_21") (Fin.cases ("candidate_vto_copy_22") (Fin.cases ("candidate_vto_copy_23") (Fin.cases ("candidate_vto_copy_24") (Fin.cases ("candidate_vto_copy_25") (Fin.cases ("candidate_vto_copy_26") (Fin.cases ("candidate_vto_copy_27") (Fin.cases ("candidate_vto_copy_28") (Fin.cases ("candidate_vto_copy_29") (Fin.cases ("candidate_vto_copy_30") (Fin.cases ("candidate_vto_copy_31") (Fin.cases ("candidate_vlinalg_vector_norm") (Fin.cases ("candidate_vlinalg_vector_norm_1") (Fin.cases ("candidate_vlinalg_vector_norm_2") (Fin.cases ("candidate_vlinalg_vector_norm_3") (Fin.cases ("candidate_vlinalg_vector_norm_4") (Fin.cases ("candidate_vlinalg_vector_norm_5") (Fin.cases ("candidate_vlinalg_vector_norm_6") (Fin.cases ("candidate_vlinalg_vector_norm_7") (Fin.cases ("candidate_vlinalg_vector_norm_8") (Fin.cases ("candidate_vlinalg_vector_norm_9") (Fin.cases ("candidate_vlinalg_vector_norm_10") (Fin.cases ("candidate_vlinalg_vector_norm_11") (Fin.cases ("candidate_vlinalg_vector_norm_12") (Fin.cases ("candidate_vlinalg_vector_norm_13") (Fin.cases ("candidate_vlinalg_vector_norm_14") (Fin.cases ("candidate_vlinalg_vector_norm_15") (Fin.cases ("candidate_vlinalg_vector_norm_16") (Fin.cases ("candidate_vlinalg_vector_norm_17") (Fin.cases ("candidate_vlinalg_vector_norm_18") (Fin.cases ("candidate_vlinalg_vector_norm_19") (Fin.cases ("candidate_vlinalg_vector_norm_20") (Fin.cases ("candidate_vlinalg_vector_norm_21") (Fin.cases ("candidate_vlinalg_vector_norm_22") (Fin.cases ("candidate_vlinalg_vector_norm_23") (Fin.cases ("candidate_vlinalg_vector_norm_24") (Fin.cases ("candidate_vlinalg_vector_norm_25") (Fin.cases ("candidate_vlinalg_vector_norm_26") (Fin.cases ("candidate_vlinalg_vector_norm_27") (Fin.cases ("candidate_vlinalg_vector_norm_28") (Fin.cases ("candidate_vlinalg_vector_norm_29") (Fin.cases ("candidate_vlinalg_vector_norm_30") (Fin.cases ("candidate_vlinalg_vector_norm_31") (Fin.cases ("candidate_vstack") (Fin.cases ("candidate_vto_copy_32") (Fin.cases ("candidate_vcat") (Fin.elim0)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
  iwin := fun _ _ _ _ _ j => j.val
  imask := fun _ _ _ _ _ _ => True
  owin := fun _ _ _ _ _ j => j.val
  omask := fun _ _ _ _ _ _ => True
  swin := fun _ _ _ _ _ j => j.val
  smask := fun _ _ _ _ _ _ => True

def inputs : core.Ctx ≃ Inputs where
  toFun := fun v => ⟨input0Equiv (v (0 : Fin 1))⟩
  invFun := fun v => Fin.cases (input0Equiv.symm v.input0) ((fun i => Fin.elim0 i))
  left_inv := by
    intro v
    funext i
    fin_cases i <;> exact Equiv.symm_apply_apply _ _
  right_inv := by
    intro v
    change (⟨input0Equiv (input0Equiv.symm v.input0)⟩ : Inputs) = v
    simp only [Equiv.apply_symm_apply]

def outputCoords : Fin (32) ≃ OutputIndex :=
  Equiv.refl _

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
  Fin.cases (.bf16) (Fin.elim0)

def scratchStorage : Fin core.nScr → TileDType :=
  Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.elim0)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

/-- The same complete channel layout; dtype casts remain in the source. -/
def io (α : Type) : ProgramIO α α where
  inputs := List.ofFn fun i : Fin core.nIn => ⟨inputStorage i, [core.iarity i], core.ibuf i⟩
  outputs := List.ofFn fun o : Fin core.nOut => ⟨.real, [core.oarity o], core.obuf o⟩
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
def admissible {α : Type} (_vals : (io α).Inputs) : Prop :=
  True

end FP
end IO
end
end VeriTile.Bench.PretrainKernels.local_grad_norm_foreach_vs_loop
