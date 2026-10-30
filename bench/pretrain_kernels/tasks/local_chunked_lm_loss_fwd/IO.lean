import bench.pretrain_kernels.tasks.local_chunked_lm_loss_fwd.Kernels
import VeriTile.Triton.Memory.LogicalScheduledKernelIO
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Float.GuardedProgramIO
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Tactic.FinCases

namespace VeriTile.Bench.PretrainKernels.local_chunked_lm_loss_fwd
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 2000000
namespace IO

structure Inputs where
  parameter0 : Fin 1024 → Fin 64 → ℝ
  parameter1 : Fin 320 → Int
  parameter2 : Fin 320 → ℝ
  input0 : Fin 320 → Fin 64 → ℝ

abbrev OutputIndex := Fin 1

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

def parameter0Equiv : (Fin (1024*64) → ℝ) ≃ (Fin 1024 → Fin 64 → ℝ) :=
  ((matrixEquiv 1024 (64) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 64 → ℝ))))

def parameter1Equiv : (Fin (320) → Int) ≃ (Fin 320 → Int) :=
  (Equiv.refl (Fin 320 → Int))

def parameter2Equiv : (Fin (320) → ℝ) ≃ (Fin 320 → ℝ) :=
  (Equiv.refl (Fin 320 → ℝ))

def input0Equiv : (Fin (320*64) → ℝ) ≃ (Fin 320 → Fin 64 → ℝ) :=
  ((matrixEquiv 320 (64) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 64 → ℝ))))

def core : UKernelIO where
  kernel := Kernels.original.eraseDType
  projection := by
    have h : Kernels.original.toAlgorithm? = .ok Kernels.original.toAlgKernel := rfl
    simp only [ComputeKernel.eraseDType, h]
    simp [ComputeKernel.toAlgKernel]
  nIn := 4
  nOut := 1
  nScr := 80
  bufs := ["parameter0", "parameter1", "parameter2", "input0", "out", "vmul", "vgt", "vto_copy", "vmm", "vto_copy_1", "vmax_1_value", "vmax_1_index", "vsub", "vexp", "vsum_1", "vlog", "vgather", "vsub_1", "vsub_2", "vneg", "vmul_1", "vsum_2", "candidate_vmul", "candidate_vgt", "candidate_vto_copy", "candidate_vmm", "candidate_vto_copy_1", "candidate_vmax_1_value", "candidate_vmax_1_index", "candidate_vsub", "candidate_vexp", "candidate_vsum_1", "candidate_vlog", "candidate_vgather", "candidate_vsub_1", "candidate_vsub_2", "candidate_vneg", "candidate_vmul_1", "candidate_vsum_2", "candidate_vadd", "candidate_vmm_1", "candidate_vto_copy_2", "candidate_vmax_2_value", "candidate_vmax_2_index", "candidate_vsub_3", "candidate_vexp_1", "candidate_vsum_3", "candidate_vlog_1", "candidate_vgather_1", "candidate_vsub_4", "candidate_vsub_5", "candidate_vneg_1", "candidate_vmul_2", "candidate_vsum_4", "candidate_vadd_1", "candidate_vmm_2", "candidate_vto_copy_3", "candidate_vmax_3_value", "candidate_vmax_3_index", "candidate_vsub_6", "candidate_vexp_2", "candidate_vsum_5", "candidate_vlog_2", "candidate_vgather_2", "candidate_vsub_7", "candidate_vsub_8", "candidate_vneg_2", "candidate_vmul_3", "candidate_vsum_6", "candidate_vadd_2", "candidate_vmm_3", "candidate_vto_copy_4", "candidate_vmax_4_value", "candidate_vmax_4_index", "candidate_vsub_9", "candidate_vexp_3", "candidate_vsum_7", "candidate_vlog_3", "candidate_vgather_3", "candidate_vsub_10", "candidate_vsub_11", "candidate_vneg_3", "candidate_vmul_4", "candidate_vsum_8", "candidate_vadd_3"]
  ity := Fin.cases (.float) (Fin.cases (.int) (Fin.cases (.float) (Fin.cases (.float) (Fin.elim0))))
  iarity := Fin.cases (1024*64) (Fin.cases (320) (Fin.cases (320) (Fin.cases (320*64) (Fin.elim0))))
  ibuf := Fin.cases ("parameter0") (Fin.cases ("parameter1") (Fin.cases ("parameter2") (Fin.cases ("input0") (Fin.elim0))))
  oty := fun _ => .float
  oarity := fun _ => 1
  obuf := fun _ => "out"
  obuf_mem := fun _ => by simp
  sarity := Fin.cases (1024*64) (Fin.cases (320) (Fin.cases (320) (Fin.cases (320*1024) (Fin.cases (320*1024) (Fin.cases (320*1) (Fin.cases (320*1) (Fin.cases (320*1024) (Fin.cases (320*1024) (Fin.cases (320) (Fin.cases (320) (Fin.cases (320*1) (Fin.cases (320) (Fin.cases (320) (Fin.cases (320) (Fin.cases (320) (Fin.cases (1) (Fin.cases (1024*64) (Fin.cases (320) (Fin.cases (320) (Fin.cases (96*1024) (Fin.cases (96*1024) (Fin.cases (96*1) (Fin.cases (96*1) (Fin.cases (96*1024) (Fin.cases (96*1024) (Fin.cases (96) (Fin.cases (96) (Fin.cases (96*1) (Fin.cases (96) (Fin.cases (96) (Fin.cases (96) (Fin.cases (96) (Fin.cases (1) (Fin.cases (1) (Fin.cases (96*1024) (Fin.cases (96*1024) (Fin.cases (96*1) (Fin.cases (96*1) (Fin.cases (96*1024) (Fin.cases (96*1024) (Fin.cases (96) (Fin.cases (96) (Fin.cases (96*1) (Fin.cases (96) (Fin.cases (96) (Fin.cases (96) (Fin.cases (96) (Fin.cases (1) (Fin.cases (1) (Fin.cases (96*1024) (Fin.cases (96*1024) (Fin.cases (96*1) (Fin.cases (96*1) (Fin.cases (96*1024) (Fin.cases (96*1024) (Fin.cases (96) (Fin.cases (96) (Fin.cases (96*1) (Fin.cases (96) (Fin.cases (96) (Fin.cases (96) (Fin.cases (96) (Fin.cases (1) (Fin.cases (1) (Fin.cases (32*1024) (Fin.cases (32*1024) (Fin.cases (32*1) (Fin.cases (32*1) (Fin.cases (32*1024) (Fin.cases (32*1024) (Fin.cases (32) (Fin.cases (32) (Fin.cases (32*1) (Fin.cases (32) (Fin.cases (32) (Fin.cases (32) (Fin.cases (32) (Fin.cases (1) (Fin.cases (1) (Fin.elim0))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
  sbuf := Fin.cases ("vmul") (Fin.cases ("vgt") (Fin.cases ("vto_copy") (Fin.cases ("vmm") (Fin.cases ("vto_copy_1") (Fin.cases ("vmax_1_value") (Fin.cases ("vmax_1_index") (Fin.cases ("vsub") (Fin.cases ("vexp") (Fin.cases ("vsum_1") (Fin.cases ("vlog") (Fin.cases ("vgather") (Fin.cases ("vsub_1") (Fin.cases ("vsub_2") (Fin.cases ("vneg") (Fin.cases ("vmul_1") (Fin.cases ("vsum_2") (Fin.cases ("candidate_vmul") (Fin.cases ("candidate_vgt") (Fin.cases ("candidate_vto_copy") (Fin.cases ("candidate_vmm") (Fin.cases ("candidate_vto_copy_1") (Fin.cases ("candidate_vmax_1_value") (Fin.cases ("candidate_vmax_1_index") (Fin.cases ("candidate_vsub") (Fin.cases ("candidate_vexp") (Fin.cases ("candidate_vsum_1") (Fin.cases ("candidate_vlog") (Fin.cases ("candidate_vgather") (Fin.cases ("candidate_vsub_1") (Fin.cases ("candidate_vsub_2") (Fin.cases ("candidate_vneg") (Fin.cases ("candidate_vmul_1") (Fin.cases ("candidate_vsum_2") (Fin.cases ("candidate_vadd") (Fin.cases ("candidate_vmm_1") (Fin.cases ("candidate_vto_copy_2") (Fin.cases ("candidate_vmax_2_value") (Fin.cases ("candidate_vmax_2_index") (Fin.cases ("candidate_vsub_3") (Fin.cases ("candidate_vexp_1") (Fin.cases ("candidate_vsum_3") (Fin.cases ("candidate_vlog_1") (Fin.cases ("candidate_vgather_1") (Fin.cases ("candidate_vsub_4") (Fin.cases ("candidate_vsub_5") (Fin.cases ("candidate_vneg_1") (Fin.cases ("candidate_vmul_2") (Fin.cases ("candidate_vsum_4") (Fin.cases ("candidate_vadd_1") (Fin.cases ("candidate_vmm_2") (Fin.cases ("candidate_vto_copy_3") (Fin.cases ("candidate_vmax_3_value") (Fin.cases ("candidate_vmax_3_index") (Fin.cases ("candidate_vsub_6") (Fin.cases ("candidate_vexp_2") (Fin.cases ("candidate_vsum_5") (Fin.cases ("candidate_vlog_2") (Fin.cases ("candidate_vgather_2") (Fin.cases ("candidate_vsub_7") (Fin.cases ("candidate_vsub_8") (Fin.cases ("candidate_vneg_2") (Fin.cases ("candidate_vmul_3") (Fin.cases ("candidate_vsum_6") (Fin.cases ("candidate_vadd_2") (Fin.cases ("candidate_vmm_3") (Fin.cases ("candidate_vto_copy_4") (Fin.cases ("candidate_vmax_4_value") (Fin.cases ("candidate_vmax_4_index") (Fin.cases ("candidate_vsub_9") (Fin.cases ("candidate_vexp_3") (Fin.cases ("candidate_vsum_7") (Fin.cases ("candidate_vlog_3") (Fin.cases ("candidate_vgather_3") (Fin.cases ("candidate_vsub_10") (Fin.cases ("candidate_vsub_11") (Fin.cases ("candidate_vneg_3") (Fin.cases ("candidate_vmul_4") (Fin.cases ("candidate_vsum_8") (Fin.cases ("candidate_vadd_3") (Fin.elim0))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
  iwin := fun _ _ _ _ _ j => j.val
  imask := fun _ _ _ _ _ _ => True
  owin := fun _ _ _ _ _ j => j.val
  omask := fun _ _ _ _ _ _ => True
  swin := fun _ _ _ _ _ j => j.val
  smask := fun _ _ _ _ _ _ => True

def inputs : core.Ctx ≃ Inputs where
  toFun := fun v => ⟨parameter0Equiv (v (0 : Fin 4)), parameter1Equiv (v (1 : Fin 4)), parameter2Equiv (v (2 : Fin 4)), input0Equiv (v (3 : Fin 4))⟩
  invFun := fun v => Fin.cases (parameter0Equiv.symm v.parameter0) (Fin.cases (parameter1Equiv.symm v.parameter1) (Fin.cases (parameter2Equiv.symm v.parameter2) (Fin.cases (input0Equiv.symm v.input0) ((fun i => Fin.elim0 i)))))
  left_inv := by
    intro v
    funext i
    fin_cases i <;> exact Equiv.symm_apply_apply _ _
  right_inv := by
    intro v
    change (⟨parameter0Equiv (parameter0Equiv.symm v.parameter0), parameter1Equiv (parameter1Equiv.symm v.parameter1), parameter2Equiv (parameter2Equiv.symm v.parameter2), input0Equiv (input0Equiv.symm v.input0)⟩ : Inputs) = v
    simp only [Equiv.apply_symm_apply]

def outputCoords : Fin (1) ≃ OutputIndex :=
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
  Fin.cases (.bf16) (Fin.cases (.int) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.elim0))))

def scratchStorage : Fin core.nScr → TileDType :=
  Fin.cases (.bf16) (Fin.cases (.bool) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bool) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.elim0))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

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
def admissible {α : Type} (vals : (io α).Inputs) : Prop :=
  (∀ j : Fin 320, Int.le 0 (vals (1 : Fin 4) (j, PUnit.unit) : Int) ∧ Int.lt (vals (1 : Fin 4) (j, PUnit.unit) : Int) 1024)

end FP
end IO
end
end VeriTile.Bench.PretrainKernels.local_chunked_lm_loss_fwd
