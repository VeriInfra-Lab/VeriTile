import bench.pretrain_kernels.tasks.local_chunked_lm_loss_wgrad.Kernels
import VeriTile.Triton.Memory.LogicalScheduledKernelIO
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Float.GuardedProgramIO
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Tactic.FinCases

namespace VeriTile.Bench.PretrainKernels.local_chunked_lm_loss_wgrad
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
  input1 : Fin 1 → ℝ

abbrev OutputIndex := Fin 1024 × Fin 64

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

def input1Equiv : (Fin (1) → ℝ) ≃ (Fin 1 → ℝ) :=
  (Equiv.refl (Fin 1 → ℝ))

def core : UKernelIO where
  kernel := Kernels.original.eraseDType
  projection := by
    have h : Kernels.original.toAlgorithm? = .ok Kernels.original.toAlgKernel := rfl
    simp only [ComputeKernel.eraseDType, h]
    simp [ComputeKernel.toAlgKernel]
  nIn := 5
  nOut := 1
  nScr := 202
  bufs := ["parameter0", "parameter1", "parameter2", "input0", "input1", "out", "vmul", "vgt", "vto_copy", "vmm", "vto_copy_1", "vmax_1_value", "vmax_1_index", "vsub", "vexp", "vsum_1", "vto_copy_2", "vmul_2", "vneg_1", "vneg_2", "vsub_3", "vexp_1", "vdiv", "vmul_3", "vneg_3", "vsum_4", "vadd", "vneg_4", "vscatter_add", "vgather_1", "vadd_1", "vadd_2", "vscatter", "vgather_2", "vadd_3", "vadd_4", "vscatter_1", "vto_copy_3", "vmm_1", "candidate_vmul", "candidate_vgt", "candidate_vto_copy", "candidate_vto_copy_5", "candidate_vmm_4", "candidate_vto_copy_6", "candidate_vmax_5_value", "candidate_vmax_5_index", "candidate_vsub_12", "candidate_vexp_4", "candidate_vsum_10", "candidate_vlog_4", "candidate_vgather_4", "candidate_vsub_13", "candidate_vsub_14", "candidate_vneg_4", "candidate_vmul_5", "candidate_vsum_11", "candidate_vmul_6", "candidate_vmul_7", "candidate_vmul_8", "candidate_vneg_5", "candidate_vneg_6", "candidate_vsub_15", "candidate_vexp_5", "candidate_vdiv", "candidate_vmul_9", "candidate_vneg_7", "candidate_vsum_12", "candidate_vadd_4", "candidate_vneg_8", "candidate_vscatter_add", "candidate_vgather_5", "candidate_vadd_5", "candidate_vadd_6", "candidate_vscatter", "candidate_vgather_6", "candidate_vadd_7", "candidate_vadd_8", "candidate_vscatter_1", "candidate_vto_copy_7", "candidate_vto_copy_8", "candidate_vto_copy_9", "candidate_vmm_6", "candidate_vadd_9", "candidate_vmm_7", "candidate_vto_copy_10", "candidate_vmax_6_value", "candidate_vmax_6_index", "candidate_vsub_16", "candidate_vexp_6", "candidate_vsum_13", "candidate_vlog_5", "candidate_vgather_7", "candidate_vsub_17", "candidate_vsub_18", "candidate_vneg_9", "candidate_vmul_10", "candidate_vsum_14", "candidate_vmul_11", "candidate_vmul_12", "candidate_vmul_13", "candidate_vneg_10", "candidate_vneg_11", "candidate_vsub_19", "candidate_vexp_7", "candidate_vdiv_1", "candidate_vmul_14", "candidate_vneg_12", "candidate_vsum_15", "candidate_vadd_10", "candidate_vneg_13", "candidate_vscatter_add_1", "candidate_vgather_8", "candidate_vadd_11", "candidate_vadd_12", "candidate_vscatter_2", "candidate_vgather_9", "candidate_vadd_13", "candidate_vadd_14", "candidate_vscatter_3", "candidate_vto_copy_11", "candidate_vto_copy_12", "candidate_vto_copy_13", "candidate_vmm_9", "candidate_vadd_15", "candidate_vmm_10", "candidate_vto_copy_14", "candidate_vmax_7_value", "candidate_vmax_7_index", "candidate_vsub_20", "candidate_vexp_8", "candidate_vsum_16", "candidate_vlog_6", "candidate_vgather_10", "candidate_vsub_21", "candidate_vsub_22", "candidate_vneg_14", "candidate_vmul_15", "candidate_vsum_17", "candidate_vmul_16", "candidate_vmul_17", "candidate_vmul_18", "candidate_vneg_15", "candidate_vneg_16", "candidate_vsub_23", "candidate_vexp_9", "candidate_vdiv_2", "candidate_vmul_19", "candidate_vneg_17", "candidate_vsum_18", "candidate_vadd_16", "candidate_vneg_18", "candidate_vscatter_add_2", "candidate_vgather_11", "candidate_vadd_17", "candidate_vadd_18", "candidate_vscatter_4", "candidate_vgather_12", "candidate_vadd_19", "candidate_vadd_20", "candidate_vscatter_5", "candidate_vto_copy_15", "candidate_vto_copy_16", "candidate_vto_copy_17", "candidate_vmm_12", "candidate_vadd_21", "candidate_vmm_13", "candidate_vto_copy_18", "candidate_vmax_8_value", "candidate_vmax_8_index", "candidate_vsub_24", "candidate_vexp_10", "candidate_vsum_19", "candidate_vlog_7", "candidate_vgather_13", "candidate_vsub_25", "candidate_vsub_26", "candidate_vneg_19", "candidate_vmul_20", "candidate_vsum_20", "candidate_vmul_21", "candidate_vmul_22", "candidate_vmul_23", "candidate_vneg_20", "candidate_vneg_21", "candidate_vsub_27", "candidate_vexp_11", "candidate_vdiv_3", "candidate_vmul_24", "candidate_vneg_22", "candidate_vsum_21", "candidate_vadd_22", "candidate_vneg_23", "candidate_vscatter_add_3", "candidate_vgather_14", "candidate_vadd_23", "candidate_vadd_24", "candidate_vscatter_6", "candidate_vgather_15", "candidate_vadd_25", "candidate_vadd_26", "candidate_vscatter_7", "candidate_vto_copy_19", "candidate_vto_copy_20", "candidate_vto_copy_21", "candidate_vmm_15", "candidate_vadd_27", "candidate_vto_copy_22"]
  ity := Fin.cases (.float) (Fin.cases (.int) (Fin.cases (.float) (Fin.cases (.float) (Fin.cases (.float) (Fin.elim0)))))
  iarity := Fin.cases (1024*64) (Fin.cases (320) (Fin.cases (320) (Fin.cases (320*64) (Fin.cases (1) (Fin.elim0)))))
  ibuf := Fin.cases ("parameter0") (Fin.cases ("parameter1") (Fin.cases ("parameter2") (Fin.cases ("input0") (Fin.cases ("input1") (Fin.elim0)))))
  oty := fun _ => .float
  oarity := fun _ => 1024*64
  obuf := fun _ => "out"
  obuf_mem := fun _ => by simp
  sarity := Fin.cases (1024*64) (Fin.cases (320) (Fin.cases (320) (Fin.cases (320*1024) (Fin.cases (320*1024) (Fin.cases (320*1) (Fin.cases (320*1) (Fin.cases (320*1024) (Fin.cases (320*1024) (Fin.cases (320) (Fin.cases (1) (Fin.cases (320) (Fin.cases (320) (Fin.cases (320) (Fin.cases (320*1024) (Fin.cases (320*1024) (Fin.cases (320) (Fin.cases (320*1024) (Fin.cases (320*1024) (Fin.cases (320*1) (Fin.cases (320*1) (Fin.cases (320*1024) (Fin.cases (320*1) (Fin.cases (320*1) (Fin.cases (320*1) (Fin.cases (320*1024) (Fin.cases (320*1024) (Fin.cases (320*1) (Fin.cases (320*1) (Fin.cases (320*1024) (Fin.cases (320*1024) (Fin.cases (320*1024) (Fin.cases (1024*64) (Fin.cases (1024*64) (Fin.cases (320) (Fin.cases (320) (Fin.cases (1) (Fin.cases (96*1024) (Fin.cases (96*1024) (Fin.cases (96*1) (Fin.cases (96*1) (Fin.cases (96*1024) (Fin.cases (96*1024) (Fin.cases (96) (Fin.cases (96) (Fin.cases (96*1) (Fin.cases (96) (Fin.cases (96) (Fin.cases (96) (Fin.cases (96) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (96) (Fin.cases (96) (Fin.cases (96) (Fin.cases (96*1024) (Fin.cases (96*1024) (Fin.cases (96) (Fin.cases (96*1024) (Fin.cases (96*1024) (Fin.cases (96*1) (Fin.cases (96*1) (Fin.cases (96*1024) (Fin.cases (96*1) (Fin.cases (96*1) (Fin.cases (96*1) (Fin.cases (96*1024) (Fin.cases (96*1024) (Fin.cases (96*1) (Fin.cases (96*1) (Fin.cases (96*1024) (Fin.cases (96*1024) (Fin.cases (96*1024) (Fin.cases (96*1024) (Fin.cases (96*64) (Fin.cases (1024*64) (Fin.cases (1024*64) (Fin.cases (96*1024) (Fin.cases (96*1024) (Fin.cases (96*1) (Fin.cases (96*1) (Fin.cases (96*1024) (Fin.cases (96*1024) (Fin.cases (96) (Fin.cases (96) (Fin.cases (96*1) (Fin.cases (96) (Fin.cases (96) (Fin.cases (96) (Fin.cases (96) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (96) (Fin.cases (96) (Fin.cases (96) (Fin.cases (96*1024) (Fin.cases (96*1024) (Fin.cases (96) (Fin.cases (96*1024) (Fin.cases (96*1024) (Fin.cases (96*1) (Fin.cases (96*1) (Fin.cases (96*1024) (Fin.cases (96*1) (Fin.cases (96*1) (Fin.cases (96*1) (Fin.cases (96*1024) (Fin.cases (96*1024) (Fin.cases (96*1) (Fin.cases (96*1) (Fin.cases (96*1024) (Fin.cases (96*1024) (Fin.cases (96*1024) (Fin.cases (96*1024) (Fin.cases (96*64) (Fin.cases (1024*64) (Fin.cases (1024*64) (Fin.cases (96*1024) (Fin.cases (96*1024) (Fin.cases (96*1) (Fin.cases (96*1) (Fin.cases (96*1024) (Fin.cases (96*1024) (Fin.cases (96) (Fin.cases (96) (Fin.cases (96*1) (Fin.cases (96) (Fin.cases (96) (Fin.cases (96) (Fin.cases (96) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (96) (Fin.cases (96) (Fin.cases (96) (Fin.cases (96*1024) (Fin.cases (96*1024) (Fin.cases (96) (Fin.cases (96*1024) (Fin.cases (96*1024) (Fin.cases (96*1) (Fin.cases (96*1) (Fin.cases (96*1024) (Fin.cases (96*1) (Fin.cases (96*1) (Fin.cases (96*1) (Fin.cases (96*1024) (Fin.cases (96*1024) (Fin.cases (96*1) (Fin.cases (96*1) (Fin.cases (96*1024) (Fin.cases (96*1024) (Fin.cases (96*1024) (Fin.cases (96*1024) (Fin.cases (96*64) (Fin.cases (1024*64) (Fin.cases (1024*64) (Fin.cases (32*1024) (Fin.cases (32*1024) (Fin.cases (32*1) (Fin.cases (32*1) (Fin.cases (32*1024) (Fin.cases (32*1024) (Fin.cases (32) (Fin.cases (32) (Fin.cases (32*1) (Fin.cases (32) (Fin.cases (32) (Fin.cases (32) (Fin.cases (32) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (32) (Fin.cases (32) (Fin.cases (32) (Fin.cases (32*1024) (Fin.cases (32*1024) (Fin.cases (32) (Fin.cases (32*1024) (Fin.cases (32*1024) (Fin.cases (32*1) (Fin.cases (32*1) (Fin.cases (32*1024) (Fin.cases (32*1) (Fin.cases (32*1) (Fin.cases (32*1) (Fin.cases (32*1024) (Fin.cases (32*1024) (Fin.cases (32*1) (Fin.cases (32*1) (Fin.cases (32*1024) (Fin.cases (32*1024) (Fin.cases (32*1024) (Fin.cases (32*1024) (Fin.cases (32*64) (Fin.cases (1024*64) (Fin.cases (1024*64) (Fin.cases (1024*64) (Fin.elim0))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
  sbuf := Fin.cases ("vmul") (Fin.cases ("vgt") (Fin.cases ("vto_copy") (Fin.cases ("vmm") (Fin.cases ("vto_copy_1") (Fin.cases ("vmax_1_value") (Fin.cases ("vmax_1_index") (Fin.cases ("vsub") (Fin.cases ("vexp") (Fin.cases ("vsum_1") (Fin.cases ("vto_copy_2") (Fin.cases ("vmul_2") (Fin.cases ("vneg_1") (Fin.cases ("vneg_2") (Fin.cases ("vsub_3") (Fin.cases ("vexp_1") (Fin.cases ("vdiv") (Fin.cases ("vmul_3") (Fin.cases ("vneg_3") (Fin.cases ("vsum_4") (Fin.cases ("vadd") (Fin.cases ("vneg_4") (Fin.cases ("vscatter_add") (Fin.cases ("vgather_1") (Fin.cases ("vadd_1") (Fin.cases ("vadd_2") (Fin.cases ("vscatter") (Fin.cases ("vgather_2") (Fin.cases ("vadd_3") (Fin.cases ("vadd_4") (Fin.cases ("vscatter_1") (Fin.cases ("vto_copy_3") (Fin.cases ("vmm_1") (Fin.cases ("candidate_vmul") (Fin.cases ("candidate_vgt") (Fin.cases ("candidate_vto_copy") (Fin.cases ("candidate_vto_copy_5") (Fin.cases ("candidate_vmm_4") (Fin.cases ("candidate_vto_copy_6") (Fin.cases ("candidate_vmax_5_value") (Fin.cases ("candidate_vmax_5_index") (Fin.cases ("candidate_vsub_12") (Fin.cases ("candidate_vexp_4") (Fin.cases ("candidate_vsum_10") (Fin.cases ("candidate_vlog_4") (Fin.cases ("candidate_vgather_4") (Fin.cases ("candidate_vsub_13") (Fin.cases ("candidate_vsub_14") (Fin.cases ("candidate_vneg_4") (Fin.cases ("candidate_vmul_5") (Fin.cases ("candidate_vsum_11") (Fin.cases ("candidate_vmul_6") (Fin.cases ("candidate_vmul_7") (Fin.cases ("candidate_vmul_8") (Fin.cases ("candidate_vneg_5") (Fin.cases ("candidate_vneg_6") (Fin.cases ("candidate_vsub_15") (Fin.cases ("candidate_vexp_5") (Fin.cases ("candidate_vdiv") (Fin.cases ("candidate_vmul_9") (Fin.cases ("candidate_vneg_7") (Fin.cases ("candidate_vsum_12") (Fin.cases ("candidate_vadd_4") (Fin.cases ("candidate_vneg_8") (Fin.cases ("candidate_vscatter_add") (Fin.cases ("candidate_vgather_5") (Fin.cases ("candidate_vadd_5") (Fin.cases ("candidate_vadd_6") (Fin.cases ("candidate_vscatter") (Fin.cases ("candidate_vgather_6") (Fin.cases ("candidate_vadd_7") (Fin.cases ("candidate_vadd_8") (Fin.cases ("candidate_vscatter_1") (Fin.cases ("candidate_vto_copy_7") (Fin.cases ("candidate_vto_copy_8") (Fin.cases ("candidate_vto_copy_9") (Fin.cases ("candidate_vmm_6") (Fin.cases ("candidate_vadd_9") (Fin.cases ("candidate_vmm_7") (Fin.cases ("candidate_vto_copy_10") (Fin.cases ("candidate_vmax_6_value") (Fin.cases ("candidate_vmax_6_index") (Fin.cases ("candidate_vsub_16") (Fin.cases ("candidate_vexp_6") (Fin.cases ("candidate_vsum_13") (Fin.cases ("candidate_vlog_5") (Fin.cases ("candidate_vgather_7") (Fin.cases ("candidate_vsub_17") (Fin.cases ("candidate_vsub_18") (Fin.cases ("candidate_vneg_9") (Fin.cases ("candidate_vmul_10") (Fin.cases ("candidate_vsum_14") (Fin.cases ("candidate_vmul_11") (Fin.cases ("candidate_vmul_12") (Fin.cases ("candidate_vmul_13") (Fin.cases ("candidate_vneg_10") (Fin.cases ("candidate_vneg_11") (Fin.cases ("candidate_vsub_19") (Fin.cases ("candidate_vexp_7") (Fin.cases ("candidate_vdiv_1") (Fin.cases ("candidate_vmul_14") (Fin.cases ("candidate_vneg_12") (Fin.cases ("candidate_vsum_15") (Fin.cases ("candidate_vadd_10") (Fin.cases ("candidate_vneg_13") (Fin.cases ("candidate_vscatter_add_1") (Fin.cases ("candidate_vgather_8") (Fin.cases ("candidate_vadd_11") (Fin.cases ("candidate_vadd_12") (Fin.cases ("candidate_vscatter_2") (Fin.cases ("candidate_vgather_9") (Fin.cases ("candidate_vadd_13") (Fin.cases ("candidate_vadd_14") (Fin.cases ("candidate_vscatter_3") (Fin.cases ("candidate_vto_copy_11") (Fin.cases ("candidate_vto_copy_12") (Fin.cases ("candidate_vto_copy_13") (Fin.cases ("candidate_vmm_9") (Fin.cases ("candidate_vadd_15") (Fin.cases ("candidate_vmm_10") (Fin.cases ("candidate_vto_copy_14") (Fin.cases ("candidate_vmax_7_value") (Fin.cases ("candidate_vmax_7_index") (Fin.cases ("candidate_vsub_20") (Fin.cases ("candidate_vexp_8") (Fin.cases ("candidate_vsum_16") (Fin.cases ("candidate_vlog_6") (Fin.cases ("candidate_vgather_10") (Fin.cases ("candidate_vsub_21") (Fin.cases ("candidate_vsub_22") (Fin.cases ("candidate_vneg_14") (Fin.cases ("candidate_vmul_15") (Fin.cases ("candidate_vsum_17") (Fin.cases ("candidate_vmul_16") (Fin.cases ("candidate_vmul_17") (Fin.cases ("candidate_vmul_18") (Fin.cases ("candidate_vneg_15") (Fin.cases ("candidate_vneg_16") (Fin.cases ("candidate_vsub_23") (Fin.cases ("candidate_vexp_9") (Fin.cases ("candidate_vdiv_2") (Fin.cases ("candidate_vmul_19") (Fin.cases ("candidate_vneg_17") (Fin.cases ("candidate_vsum_18") (Fin.cases ("candidate_vadd_16") (Fin.cases ("candidate_vneg_18") (Fin.cases ("candidate_vscatter_add_2") (Fin.cases ("candidate_vgather_11") (Fin.cases ("candidate_vadd_17") (Fin.cases ("candidate_vadd_18") (Fin.cases ("candidate_vscatter_4") (Fin.cases ("candidate_vgather_12") (Fin.cases ("candidate_vadd_19") (Fin.cases ("candidate_vadd_20") (Fin.cases ("candidate_vscatter_5") (Fin.cases ("candidate_vto_copy_15") (Fin.cases ("candidate_vto_copy_16") (Fin.cases ("candidate_vto_copy_17") (Fin.cases ("candidate_vmm_12") (Fin.cases ("candidate_vadd_21") (Fin.cases ("candidate_vmm_13") (Fin.cases ("candidate_vto_copy_18") (Fin.cases ("candidate_vmax_8_value") (Fin.cases ("candidate_vmax_8_index") (Fin.cases ("candidate_vsub_24") (Fin.cases ("candidate_vexp_10") (Fin.cases ("candidate_vsum_19") (Fin.cases ("candidate_vlog_7") (Fin.cases ("candidate_vgather_13") (Fin.cases ("candidate_vsub_25") (Fin.cases ("candidate_vsub_26") (Fin.cases ("candidate_vneg_19") (Fin.cases ("candidate_vmul_20") (Fin.cases ("candidate_vsum_20") (Fin.cases ("candidate_vmul_21") (Fin.cases ("candidate_vmul_22") (Fin.cases ("candidate_vmul_23") (Fin.cases ("candidate_vneg_20") (Fin.cases ("candidate_vneg_21") (Fin.cases ("candidate_vsub_27") (Fin.cases ("candidate_vexp_11") (Fin.cases ("candidate_vdiv_3") (Fin.cases ("candidate_vmul_24") (Fin.cases ("candidate_vneg_22") (Fin.cases ("candidate_vsum_21") (Fin.cases ("candidate_vadd_22") (Fin.cases ("candidate_vneg_23") (Fin.cases ("candidate_vscatter_add_3") (Fin.cases ("candidate_vgather_14") (Fin.cases ("candidate_vadd_23") (Fin.cases ("candidate_vadd_24") (Fin.cases ("candidate_vscatter_6") (Fin.cases ("candidate_vgather_15") (Fin.cases ("candidate_vadd_25") (Fin.cases ("candidate_vadd_26") (Fin.cases ("candidate_vscatter_7") (Fin.cases ("candidate_vto_copy_19") (Fin.cases ("candidate_vto_copy_20") (Fin.cases ("candidate_vto_copy_21") (Fin.cases ("candidate_vmm_15") (Fin.cases ("candidate_vadd_27") (Fin.cases ("candidate_vto_copy_22") (Fin.elim0))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
  iwin := fun _ _ _ _ _ j => j.val
  imask := fun _ _ _ _ _ _ => True
  owin := fun _ _ _ _ _ j => j.val
  omask := fun _ _ _ _ _ _ => True
  swin := fun _ _ _ _ _ j => j.val
  smask := fun _ _ _ _ _ _ => True

def inputs : core.Ctx ≃ Inputs where
  toFun := fun v => ⟨parameter0Equiv (v (0 : Fin 5)), parameter1Equiv (v (1 : Fin 5)), parameter2Equiv (v (2 : Fin 5)), input0Equiv (v (3 : Fin 5)), input1Equiv (v (4 : Fin 5))⟩
  invFun := fun v => Fin.cases (parameter0Equiv.symm v.parameter0) (Fin.cases (parameter1Equiv.symm v.parameter1) (Fin.cases (parameter2Equiv.symm v.parameter2) (Fin.cases (input0Equiv.symm v.input0) (Fin.cases (input1Equiv.symm v.input1) ((fun i => Fin.elim0 i))))))
  left_inv := by
    intro v
    funext i
    fin_cases i <;> exact Equiv.symm_apply_apply _ _
  right_inv := by
    intro v
    change (⟨parameter0Equiv (parameter0Equiv.symm v.parameter0), parameter1Equiv (parameter1Equiv.symm v.parameter1), parameter2Equiv (parameter2Equiv.symm v.parameter2), input0Equiv (input0Equiv.symm v.input0), input1Equiv (input1Equiv.symm v.input1)⟩ : Inputs) = v
    simp only [Equiv.apply_symm_apply]

def outputCoords : Fin (1024*64) ≃ OutputIndex :=
  (finProdFinEquiv (m := 1024) (n := 64)).symm

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
  Fin.cases (.bf16) (Fin.cases (.int) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.elim0)))))

def scratchStorage : Fin core.nScr → TileDType :=
  Fin.cases (.bf16) (Fin.cases (.bool) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bool) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.elim0))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

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
  (∀ j : Fin 320, Int.le 0 (vals (1 : Fin 5) (j, PUnit.unit) : Int) ∧ Int.lt (vals (1 : Fin 5) (j, PUnit.unit) : Int) 1024)

end FP
end IO
end
end VeriTile.Bench.PretrainKernels.local_chunked_lm_loss_wgrad
