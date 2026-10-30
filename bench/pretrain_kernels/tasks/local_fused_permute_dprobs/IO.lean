import bench.pretrain_kernels.tasks.local_fused_permute_dprobs.Kernels
import VeriTile.Triton.Memory.LogicalScheduledKernelIO
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Float.GuardedProgramIO
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Tactic.FinCases

namespace VeriTile.Bench.PretrainKernels.local_fused_permute_dprobs
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 2000000
namespace IO

structure Inputs where
  parameter0 : Fin 256 → Fin 128 → ℝ
  input0 : Fin 1536 → ℝ

abbrev OutputIndex := Fin 256 × Fin 128

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

def parameter0Equiv : (Fin (256*128) → ℝ) ≃ (Fin 256 → Fin 128 → ℝ) :=
  ((matrixEquiv 256 (128) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 128 → ℝ))))

def input0Equiv : (Fin (1536) → ℝ) ≃ (Fin 1536 → ℝ) :=
  (Equiv.refl (Fin 1536 → ℝ))

def core : UKernelIO where
  kernel := Kernels.original.eraseDType
  projection := by
    have h : Kernels.original.toAlgorithm? = .ok Kernels.original.toAlgKernel := rfl
    simp only [ComputeKernel.eraseDType, h]
    simp [ComputeKernel.toAlgKernel]
  nIn := 2
  nOut := 1
  nScr := 26
  bufs := ["parameter0", "input0", "out", "vtopk_value", "vtopk_index", "vscatter", "vto_copy_3", "vsort_value", "vsort_index", "vindex_put", "candidate_vtopk_value", "candidate_vtopk_index", "candidate_vscatter", "candidate_vto_copy_3", "candidate_vto_copy_4", "candidate_vto_copy_5", "candidate_vto_copy_6", "candidate_vsum_2", "candidate_vcumsum", "candidate_vsub", "candidate_vto_copy_7", "candidate_vcumsum_1", "candidate_vsub_1", "candidate_vadd", "candidate_vwhere", "candidate_vwhere_1", "candidate_vslice_backward", "candidate_vgather", "candidate_vwhere_4"]
  ity := Fin.cases (.float) (Fin.cases (.float) (Fin.elim0))
  iarity := Fin.cases (256*128) (Fin.cases (1536) (Fin.elim0))
  ibuf := Fin.cases ("parameter0") (Fin.cases ("input0") (Fin.elim0))
  oty := fun _ => .float
  oarity := fun _ => 256*128
  obuf := fun _ => "out"
  obuf_mem := fun _ => by simp
  sarity := Fin.cases (256*6) (Fin.cases (256*6) (Fin.cases (256*128) (Fin.cases (1536) (Fin.cases (32768) (Fin.cases (32768) (Fin.cases (32768) (Fin.cases (256*6) (Fin.cases (256*6) (Fin.cases (256*128) (Fin.cases (1536) (Fin.cases (256*128) (Fin.cases (256*128) (Fin.cases (256*128) (Fin.cases (128) (Fin.cases (128) (Fin.cases (128) (Fin.cases (256*128) (Fin.cases (256*128) (Fin.cases (256*128) (Fin.cases (256*128) (Fin.cases (256*128) (Fin.cases (256*128) (Fin.cases (1537) (Fin.cases (32768) (Fin.cases (256*128) (Fin.elim0))))))))))))))))))))))))))
  sbuf := Fin.cases ("vtopk_value") (Fin.cases ("vtopk_index") (Fin.cases ("vscatter") (Fin.cases ("vto_copy_3") (Fin.cases ("vsort_value") (Fin.cases ("vsort_index") (Fin.cases ("vindex_put") (Fin.cases ("candidate_vtopk_value") (Fin.cases ("candidate_vtopk_index") (Fin.cases ("candidate_vscatter") (Fin.cases ("candidate_vto_copy_3") (Fin.cases ("candidate_vto_copy_4") (Fin.cases ("candidate_vto_copy_5") (Fin.cases ("candidate_vto_copy_6") (Fin.cases ("candidate_vsum_2") (Fin.cases ("candidate_vcumsum") (Fin.cases ("candidate_vsub") (Fin.cases ("candidate_vto_copy_7") (Fin.cases ("candidate_vcumsum_1") (Fin.cases ("candidate_vsub_1") (Fin.cases ("candidate_vadd") (Fin.cases ("candidate_vwhere") (Fin.cases ("candidate_vwhere_1") (Fin.cases ("candidate_vslice_backward") (Fin.cases ("candidate_vgather") (Fin.cases ("candidate_vwhere_4") (Fin.elim0))))))))))))))))))))))))))
  iwin := fun _ _ _ _ _ j => j.val
  imask := fun _ _ _ _ _ _ => True
  owin := fun _ _ _ _ _ j => j.val
  omask := fun _ _ _ _ _ _ => True
  swin := fun _ _ _ _ _ j => j.val
  smask := fun _ _ _ _ _ _ => True

def inputs : core.Ctx ≃ Inputs where
  toFun := fun v => ⟨parameter0Equiv (v (0 : Fin 2)), input0Equiv (v (1 : Fin 2))⟩
  invFun := fun v => Fin.cases (parameter0Equiv.symm v.parameter0) (Fin.cases (input0Equiv.symm v.input0) ((fun i => Fin.elim0 i)))
  left_inv := by
    intro v
    funext i
    fin_cases i <;> exact Equiv.symm_apply_apply _ _
  right_inv := by
    intro v
    change (⟨parameter0Equiv (parameter0Equiv.symm v.parameter0), input0Equiv (input0Equiv.symm v.input0)⟩ : Inputs) = v
    simp only [Equiv.apply_symm_apply]

def outputCoords : Fin (256*128) ≃ OutputIndex :=
  (finProdFinEquiv (m := 256) (n := 128)).symm

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
  Fin.cases (.real) (Fin.cases (.real) (Fin.elim0))

def scratchStorage : Fin core.nScr → TileDType :=
  Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.bool) (Fin.cases (.bf16) (Fin.cases (.bool) (Fin.cases (.int) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.bool) (Fin.cases (.bf16) (Fin.cases (.int) (Fin.cases (.bool) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.elim0))))))))))))))))))))))))))

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
def admissible {α : Type} (_vals : (io α).Inputs) : Prop :=
  True

end FP
end IO
end
end VeriTile.Bench.PretrainKernels.local_fused_permute_dprobs
