import bench.pretrain_kernels.tasks.local_mla_fa4_absorb_wgrad.Kernels
import VeriTile.Triton.Memory.LogicalScheduledKernelIO
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Float.GuardedProgramIO
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Tactic.FinCases

namespace VeriTile.Bench.PretrainKernels.local_mla_fa4_absorb_wgrad
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 2000000
namespace IO

structure Inputs where
  parameter0 : Fin 256 → Fin 512 → ℝ
  parameter1 : Fin 256 → ℝ
  parameter2 : Fin 1536 → Fin 256 → ℝ
  parameter3 : Fin 576 → Fin 512 → ℝ
  parameter4 : Fin 512 → ℝ
  parameter5 : Fin 2048 → Fin 512 → ℝ
  parameter6 : Fin 512 → Fin 1024 → ℝ
  input0 : Fin 2 → Fin 128 → Fin 512 → ℝ
  input1 : Fin 2 → Fin 128 → Fin 512 → ℝ

abbrev OutputIndex := Fin 2048 × Fin 512

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

def parameter0Equiv : (Fin (256*512) → ℝ) ≃ (Fin 256 → Fin 512 → ℝ) :=
  ((matrixEquiv 256 (512) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 512 → ℝ))))

def parameter1Equiv : (Fin (256) → ℝ) ≃ (Fin 256 → ℝ) :=
  (Equiv.refl (Fin 256 → ℝ))

def parameter2Equiv : (Fin (1536*256) → ℝ) ≃ (Fin 1536 → Fin 256 → ℝ) :=
  ((matrixEquiv 1536 (256) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 256 → ℝ))))

def parameter3Equiv : (Fin (576*512) → ℝ) ≃ (Fin 576 → Fin 512 → ℝ) :=
  ((matrixEquiv 576 (512) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 512 → ℝ))))

def parameter4Equiv : (Fin (512) → ℝ) ≃ (Fin 512 → ℝ) :=
  (Equiv.refl (Fin 512 → ℝ))

def parameter5Equiv : (Fin (2048*512) → ℝ) ≃ (Fin 2048 → Fin 512 → ℝ) :=
  ((matrixEquiv 2048 (512) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 512 → ℝ))))

def parameter6Equiv : (Fin (512*1024) → ℝ) ≃ (Fin 512 → Fin 1024 → ℝ) :=
  ((matrixEquiv 512 (1024) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 1024 → ℝ))))

def input0Equiv : (Fin (2*128*512) → ℝ) ≃ (Fin 2 → Fin 128 → Fin 512 → ℝ) :=
  ((matrixEquiv 2 (128*512) ℝ).trans (Equiv.piCongrRight fun _ => ((matrixEquiv 128 (512) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 512 → ℝ))))))

def input1Equiv : (Fin (2*128*512) → ℝ) ≃ (Fin 2 → Fin 128 → Fin 512 → ℝ) :=
  ((matrixEquiv 2 (128*512) ℝ).trans (Equiv.piCongrRight fun _ => ((matrixEquiv 128 (512) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 512 → ℝ))))))

def core : UKernelIO where
  kernel := Kernels.original.eraseDType
  projection := by
    have h : Kernels.original.toAlgorithm? = .ok Kernels.original.toAlgKernel := rfl
    simp only [ComputeKernel.eraseDType, h]
    simp [ComputeKernel.toAlgKernel]
  nIn := 9
  nOut := 1
  nScr := 201
  bufs := ["parameter0", "parameter1", "parameter2", "parameter3", "parameter4", "parameter5", "parameter6", "input0", "input1", "out", "vmul", "vmul_1", "vadd", "vmul_2", "vmul_3", "vmul_4", "vadd_1", "vmul_5", "vmul_6", "vto_copy", "vto_copy_1", "vto_copy_2", "vto_copy_3", "vto_copy_4", "vto_copy_6", "vtriu", "vmasked_fill", "vto_copy_16", "vmm", "vto_copy_17", "vpow_1", "vmean", "vadd_2", "vrsqrt", "vmul_7", "vmul_8", "vto_copy_18", "vmm_1", "vmm_2", "vto_copy_19", "vpow_2", "vmean_1", "vadd_3", "vrsqrt_1", "vmul_9", "vmul_10", "vto_copy_20", "vmm_3", "vmul_11", "vmul_12", "vsub", "vmul_13", "vmul_14", "vadd_4", "vcat", "vmul_15", "vmul_16", "vsub_1", "vmul_17", "vmul_18", "vadd_5", "vcat_1", "vcat_2", "vcat_3", "vbmm", "vmul_19", "vadd_6", "vto_copy_21", "vsoftmax", "vto_copy_22", "vmm_5", "vbmm_2", "vbmm_3", "vto_copy_23", "vsoftmax_backward_data", "vto_copy_24", "vmul_20", "vbmm_4", "vcat_4", "vmm_6", "candidate_vmul", "candidate_vmul_1", "candidate_vadd", "candidate_vmul_2", "candidate_vmul_3", "candidate_vmul_4", "candidate_vadd_1", "candidate_vmul_5", "candidate_vmul_6", "candidate_vto_copy", "candidate_vto_copy_1", "candidate_vto_copy_2", "candidate_vto_copy_3", "candidate_vto_copy_4", "candidate_vto_copy_6", "candidate_vto_copy_16", "candidate_vmm", "candidate_vto_copy_17", "candidate_vpow_1", "candidate_vmean", "candidate_vadd_2", "candidate_vrsqrt", "candidate_vmul_7", "candidate_vmul_8", "candidate_vto_copy_18", "candidate_vmm_1", "candidate_vmm_2", "candidate_vto_copy_19", "candidate_vpow_2", "candidate_vmean_1", "candidate_vadd_3", "candidate_vrsqrt_1", "candidate_vmul_9", "candidate_vmul_10", "candidate_vto_copy_20", "candidate_vmul_11", "candidate_vmul_12", "candidate_vsub", "candidate_vmul_13", "candidate_vmul_14", "candidate_vadd_4", "candidate_vcat", "candidate_vmul_15", "candidate_vmul_16", "candidate_vsub_1", "candidate_vmul_17", "candidate_vmul_18", "candidate_vadd_5", "candidate_vcat_1", "candidate_vmm_8", "candidate_vbmm_4", "candidate_vbmm_5", "candidate_vcat_3", "candidate_vcat_4", "candidate_vto_copy_31", "candidate_vto_copy_32", "candidate_vbmm_6", "candidate_vmul_21", "candidate_vadd_10", "candidate_vsub_4", "candidate_vgt_2", "candidate_vmasked_fill_4", "candidate_vsoftmax_2", "candidate_vto_copy_33", "candidate_vbmm_7", "candidate_vlogsumexp_2", "candidate_vto_copy_34", "candidate_vto_copy_35", "candidate_vbmm_8", "candidate_vmul_22", "candidate_vadd_11", "candidate_vsub_5", "candidate_vgt_3", "candidate_vmasked_fill_5", "candidate_vsoftmax_3", "candidate_vto_copy_36", "candidate_vbmm_9", "candidate_vlogsumexp_3", "candidate_vcat_5", "candidate_vcat_6", "candidate_vto_copy_37", "candidate_vto_copy_38", "candidate_vto_copy_39", "candidate_vto_copy_40", "candidate_vto_copy_41", "candidate_vbmm_10", "candidate_vmul_23", "candidate_vadd_12", "candidate_vsub_6", "candidate_vgt_4", "candidate_vmasked_fill_6", "candidate_vsub_7", "candidate_vexp", "candidate_vmul_24", "candidate_vsum_1", "candidate_vbmm_11", "candidate_vsub_8", "candidate_vmul_25", "candidate_vbmm_13", "candidate_vmul_27", "candidate_vto_copy_43", "candidate_vbmm_14", "candidate_vto_copy_44", "candidate_vto_copy_45", "candidate_vto_copy_46", "candidate_vto_copy_47", "candidate_vto_copy_48", "candidate_vto_copy_49", "candidate_vbmm_15", "candidate_vmul_28", "candidate_vadd_13", "candidate_vsub_9", "candidate_vgt_5", "candidate_vmasked_fill_7", "candidate_vsub_10", "candidate_vexp_1", "candidate_vmul_29", "candidate_vsum_2", "candidate_vbmm_16", "candidate_vsub_11", "candidate_vmul_30", "candidate_vbmm_18", "candidate_vmul_32", "candidate_vto_copy_51", "candidate_vbmm_19", "candidate_vto_copy_52", "candidate_vcat_8", "candidate_vcat_9", "candidate_vbmm_22", "candidate_vbmm_23", "candidate_vcat_10"]
  ity := Fin.cases (.float) (Fin.cases (.float) (Fin.cases (.float) (Fin.cases (.float) (Fin.cases (.float) (Fin.cases (.float) (Fin.cases (.float) (Fin.cases (.float) (Fin.cases (.float) (Fin.elim0)))))))))
  iarity := Fin.cases (256*512) (Fin.cases (256) (Fin.cases (1536*256) (Fin.cases (576*512) (Fin.cases (512) (Fin.cases (2048*512) (Fin.cases (512*1024) (Fin.cases (2*128*512) (Fin.cases (2*128*512) (Fin.elim0)))))))))
  ibuf := Fin.cases ("parameter0") (Fin.cases ("parameter1") (Fin.cases ("parameter2") (Fin.cases ("parameter3") (Fin.cases ("parameter4") (Fin.cases ("parameter5") (Fin.cases ("parameter6") (Fin.cases ("input0") (Fin.cases ("input1") (Fin.elim0)))))))))
  oty := fun _ => .float
  oarity := fun _ => 2048*512
  obuf := fun _ => "out"
  obuf_mem := fun _ => by simp
  sarity := Fin.cases (256*512) (Fin.cases (256) (Fin.cases (256) (Fin.cases (1536*256) (Fin.cases (576*512) (Fin.cases (512) (Fin.cases (512) (Fin.cases (2048*512) (Fin.cases (512*1024) (Fin.cases (256*512) (Fin.cases (256) (Fin.cases (1536*256) (Fin.cases (576*512) (Fin.cases (512) (Fin.cases (512*1024) (Fin.cases (128*128) (Fin.cases (2*1*128*128) (Fin.cases (2048*512) (Fin.cases (256*256) (Fin.cases (2*128*256) (Fin.cases (2*128*256) (Fin.cases (2*128*1) (Fin.cases (2*128*1) (Fin.cases (2*128*1) (Fin.cases (2*128*256) (Fin.cases (2*128*256) (Fin.cases (2*128*256) (Fin.cases (256*1536) (Fin.cases (256*576) (Fin.cases (2*128*512) (Fin.cases (2*128*512) (Fin.cases (2*128*1) (Fin.cases (2*128*1) (Fin.cases (2*128*1) (Fin.cases (2*128*512) (Fin.cases (2*128*512) (Fin.cases (2*128*512) (Fin.cases (256*2048) (Fin.cases (2*8*128*32) (Fin.cases (2*8*128*32) (Fin.cases (2*8*128*32) (Fin.cases (2*8*128*32) (Fin.cases (2*8*128*32) (Fin.cases (2*8*128*32) (Fin.cases (2*8*128*64) (Fin.cases (2*1*128*32) (Fin.cases (2*1*128*32) (Fin.cases (2*1*128*32) (Fin.cases (2*1*128*32) (Fin.cases (2*1*128*32) (Fin.cases (2*1*128*32) (Fin.cases (2*1*128*64) (Fin.cases (2*8*128*192) (Fin.cases (2*8*128*192) (Fin.cases (16*128*128) (Fin.cases (2*8*128*128) (Fin.cases (2*8*128*128) (Fin.cases (2*8*128*128) (Fin.cases (2*8*128*128) (Fin.cases (2*8*128*128) (Fin.cases (256*1024) (Fin.cases (16*128*128) (Fin.cases (16*128*128) (Fin.cases (2*8*128*128) (Fin.cases (2*8*128*128) (Fin.cases (2*8*128*128) (Fin.cases (2*8*128*128) (Fin.cases (16*192*128) (Fin.cases (2*8*128*256) (Fin.cases (2048*512) (Fin.cases (256*512) (Fin.cases (256) (Fin.cases (256) (Fin.cases (1536*256) (Fin.cases (576*512) (Fin.cases (512) (Fin.cases (512) (Fin.cases (2048*512) (Fin.cases (512*1024) (Fin.cases (256*512) (Fin.cases (256) (Fin.cases (1536*256) (Fin.cases (576*512) (Fin.cases (512) (Fin.cases (512*1024) (Fin.cases (2048*512) (Fin.cases (256*256) (Fin.cases (2*128*256) (Fin.cases (2*128*256) (Fin.cases (2*128*1) (Fin.cases (2*128*1) (Fin.cases (2*128*1) (Fin.cases (2*128*256) (Fin.cases (2*128*256) (Fin.cases (2*128*256) (Fin.cases (256*1536) (Fin.cases (256*576) (Fin.cases (2*128*512) (Fin.cases (2*128*512) (Fin.cases (2*128*1) (Fin.cases (2*128*1) (Fin.cases (2*128*1) (Fin.cases (2*128*512) (Fin.cases (2*128*512) (Fin.cases (2*128*512) (Fin.cases (2*8*128*32) (Fin.cases (2*8*128*32) (Fin.cases (2*8*128*32) (Fin.cases (2*8*128*32) (Fin.cases (2*8*128*32) (Fin.cases (2*8*128*32) (Fin.cases (2*8*128*64) (Fin.cases (2*1*128*32) (Fin.cases (2*1*128*32) (Fin.cases (2*1*128*32) (Fin.cases (2*1*128*32) (Fin.cases (2*1*128*32) (Fin.cases (2*1*128*32) (Fin.cases (2*1*128*64) (Fin.cases (256*1024) (Fin.cases (1*256*1024) (Fin.cases (1*256*1024) (Fin.cases (256*8*192) (Fin.cases (256*8*192) (Fin.cases (8*128*192) (Fin.cases (8*128*192) (Fin.cases (8*128*128) (Fin.cases (8*128*128) (Fin.cases (128*1) (Fin.cases (128*1) (Fin.cases (128*128) (Fin.cases (8*128*128) (Fin.cases (8*128*128) (Fin.cases (8*128*128) (Fin.cases (8*128*128) (Fin.cases (8*128) (Fin.cases (8*128*192) (Fin.cases (8*128*192) (Fin.cases (8*128*128) (Fin.cases (8*128*128) (Fin.cases (128*1) (Fin.cases (128*1) (Fin.cases (128*128) (Fin.cases (8*128*128) (Fin.cases (8*128*128) (Fin.cases (8*128*128) (Fin.cases (8*128*128) (Fin.cases (8*128) (Fin.cases (256*8*128) (Fin.cases (256*8) (Fin.cases (8*128*192) (Fin.cases (8*128*192) (Fin.cases (8*128*128) (Fin.cases (8*128*128) (Fin.cases (8*128*128) (Fin.cases (8*128*128) (Fin.cases (8*128*128) (Fin.cases (128*1) (Fin.cases (128*1) (Fin.cases (128*128) (Fin.cases (8*128*128) (Fin.cases (8*128*128) (Fin.cases (8*128*128) (Fin.cases (8*128*128) (Fin.cases (8*128*1) (Fin.cases (8*128*128) (Fin.cases (8*128*128) (Fin.cases (8*128*128) (Fin.cases (8*128*192) (Fin.cases (8*128*192) (Fin.cases (128*8*192) (Fin.cases (8*128*128) (Fin.cases (128*8*128) (Fin.cases (8*128*192) (Fin.cases (8*128*192) (Fin.cases (8*128*128) (Fin.cases (8*128*128) (Fin.cases (8*128*128) (Fin.cases (8*128*128) (Fin.cases (8*128*128) (Fin.cases (128*1) (Fin.cases (128*1) (Fin.cases (128*128) (Fin.cases (8*128*128) (Fin.cases (8*128*128) (Fin.cases (8*128*128) (Fin.cases (8*128*128) (Fin.cases (8*128*1) (Fin.cases (8*128*128) (Fin.cases (8*128*128) (Fin.cases (8*128*128) (Fin.cases (8*128*192) (Fin.cases (8*128*192) (Fin.cases (128*8*192) (Fin.cases (8*128*128) (Fin.cases (128*8*128) (Fin.cases (256*8*192) (Fin.cases (256*8*128) (Fin.cases (1*1024*512) (Fin.cases (1*1024*512) (Fin.cases (8*256*512) (Fin.elim0)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
  sbuf := Fin.cases ("vmul") (Fin.cases ("vmul_1") (Fin.cases ("vadd") (Fin.cases ("vmul_2") (Fin.cases ("vmul_3") (Fin.cases ("vmul_4") (Fin.cases ("vadd_1") (Fin.cases ("vmul_5") (Fin.cases ("vmul_6") (Fin.cases ("vto_copy") (Fin.cases ("vto_copy_1") (Fin.cases ("vto_copy_2") (Fin.cases ("vto_copy_3") (Fin.cases ("vto_copy_4") (Fin.cases ("vto_copy_6") (Fin.cases ("vtriu") (Fin.cases ("vmasked_fill") (Fin.cases ("vto_copy_16") (Fin.cases ("vmm") (Fin.cases ("vto_copy_17") (Fin.cases ("vpow_1") (Fin.cases ("vmean") (Fin.cases ("vadd_2") (Fin.cases ("vrsqrt") (Fin.cases ("vmul_7") (Fin.cases ("vmul_8") (Fin.cases ("vto_copy_18") (Fin.cases ("vmm_1") (Fin.cases ("vmm_2") (Fin.cases ("vto_copy_19") (Fin.cases ("vpow_2") (Fin.cases ("vmean_1") (Fin.cases ("vadd_3") (Fin.cases ("vrsqrt_1") (Fin.cases ("vmul_9") (Fin.cases ("vmul_10") (Fin.cases ("vto_copy_20") (Fin.cases ("vmm_3") (Fin.cases ("vmul_11") (Fin.cases ("vmul_12") (Fin.cases ("vsub") (Fin.cases ("vmul_13") (Fin.cases ("vmul_14") (Fin.cases ("vadd_4") (Fin.cases ("vcat") (Fin.cases ("vmul_15") (Fin.cases ("vmul_16") (Fin.cases ("vsub_1") (Fin.cases ("vmul_17") (Fin.cases ("vmul_18") (Fin.cases ("vadd_5") (Fin.cases ("vcat_1") (Fin.cases ("vcat_2") (Fin.cases ("vcat_3") (Fin.cases ("vbmm") (Fin.cases ("vmul_19") (Fin.cases ("vadd_6") (Fin.cases ("vto_copy_21") (Fin.cases ("vsoftmax") (Fin.cases ("vto_copy_22") (Fin.cases ("vmm_5") (Fin.cases ("vbmm_2") (Fin.cases ("vbmm_3") (Fin.cases ("vto_copy_23") (Fin.cases ("vsoftmax_backward_data") (Fin.cases ("vto_copy_24") (Fin.cases ("vmul_20") (Fin.cases ("vbmm_4") (Fin.cases ("vcat_4") (Fin.cases ("vmm_6") (Fin.cases ("candidate_vmul") (Fin.cases ("candidate_vmul_1") (Fin.cases ("candidate_vadd") (Fin.cases ("candidate_vmul_2") (Fin.cases ("candidate_vmul_3") (Fin.cases ("candidate_vmul_4") (Fin.cases ("candidate_vadd_1") (Fin.cases ("candidate_vmul_5") (Fin.cases ("candidate_vmul_6") (Fin.cases ("candidate_vto_copy") (Fin.cases ("candidate_vto_copy_1") (Fin.cases ("candidate_vto_copy_2") (Fin.cases ("candidate_vto_copy_3") (Fin.cases ("candidate_vto_copy_4") (Fin.cases ("candidate_vto_copy_6") (Fin.cases ("candidate_vto_copy_16") (Fin.cases ("candidate_vmm") (Fin.cases ("candidate_vto_copy_17") (Fin.cases ("candidate_vpow_1") (Fin.cases ("candidate_vmean") (Fin.cases ("candidate_vadd_2") (Fin.cases ("candidate_vrsqrt") (Fin.cases ("candidate_vmul_7") (Fin.cases ("candidate_vmul_8") (Fin.cases ("candidate_vto_copy_18") (Fin.cases ("candidate_vmm_1") (Fin.cases ("candidate_vmm_2") (Fin.cases ("candidate_vto_copy_19") (Fin.cases ("candidate_vpow_2") (Fin.cases ("candidate_vmean_1") (Fin.cases ("candidate_vadd_3") (Fin.cases ("candidate_vrsqrt_1") (Fin.cases ("candidate_vmul_9") (Fin.cases ("candidate_vmul_10") (Fin.cases ("candidate_vto_copy_20") (Fin.cases ("candidate_vmul_11") (Fin.cases ("candidate_vmul_12") (Fin.cases ("candidate_vsub") (Fin.cases ("candidate_vmul_13") (Fin.cases ("candidate_vmul_14") (Fin.cases ("candidate_vadd_4") (Fin.cases ("candidate_vcat") (Fin.cases ("candidate_vmul_15") (Fin.cases ("candidate_vmul_16") (Fin.cases ("candidate_vsub_1") (Fin.cases ("candidate_vmul_17") (Fin.cases ("candidate_vmul_18") (Fin.cases ("candidate_vadd_5") (Fin.cases ("candidate_vcat_1") (Fin.cases ("candidate_vmm_8") (Fin.cases ("candidate_vbmm_4") (Fin.cases ("candidate_vbmm_5") (Fin.cases ("candidate_vcat_3") (Fin.cases ("candidate_vcat_4") (Fin.cases ("candidate_vto_copy_31") (Fin.cases ("candidate_vto_copy_32") (Fin.cases ("candidate_vbmm_6") (Fin.cases ("candidate_vmul_21") (Fin.cases ("candidate_vadd_10") (Fin.cases ("candidate_vsub_4") (Fin.cases ("candidate_vgt_2") (Fin.cases ("candidate_vmasked_fill_4") (Fin.cases ("candidate_vsoftmax_2") (Fin.cases ("candidate_vto_copy_33") (Fin.cases ("candidate_vbmm_7") (Fin.cases ("candidate_vlogsumexp_2") (Fin.cases ("candidate_vto_copy_34") (Fin.cases ("candidate_vto_copy_35") (Fin.cases ("candidate_vbmm_8") (Fin.cases ("candidate_vmul_22") (Fin.cases ("candidate_vadd_11") (Fin.cases ("candidate_vsub_5") (Fin.cases ("candidate_vgt_3") (Fin.cases ("candidate_vmasked_fill_5") (Fin.cases ("candidate_vsoftmax_3") (Fin.cases ("candidate_vto_copy_36") (Fin.cases ("candidate_vbmm_9") (Fin.cases ("candidate_vlogsumexp_3") (Fin.cases ("candidate_vcat_5") (Fin.cases ("candidate_vcat_6") (Fin.cases ("candidate_vto_copy_37") (Fin.cases ("candidate_vto_copy_38") (Fin.cases ("candidate_vto_copy_39") (Fin.cases ("candidate_vto_copy_40") (Fin.cases ("candidate_vto_copy_41") (Fin.cases ("candidate_vbmm_10") (Fin.cases ("candidate_vmul_23") (Fin.cases ("candidate_vadd_12") (Fin.cases ("candidate_vsub_6") (Fin.cases ("candidate_vgt_4") (Fin.cases ("candidate_vmasked_fill_6") (Fin.cases ("candidate_vsub_7") (Fin.cases ("candidate_vexp") (Fin.cases ("candidate_vmul_24") (Fin.cases ("candidate_vsum_1") (Fin.cases ("candidate_vbmm_11") (Fin.cases ("candidate_vsub_8") (Fin.cases ("candidate_vmul_25") (Fin.cases ("candidate_vbmm_13") (Fin.cases ("candidate_vmul_27") (Fin.cases ("candidate_vto_copy_43") (Fin.cases ("candidate_vbmm_14") (Fin.cases ("candidate_vto_copy_44") (Fin.cases ("candidate_vto_copy_45") (Fin.cases ("candidate_vto_copy_46") (Fin.cases ("candidate_vto_copy_47") (Fin.cases ("candidate_vto_copy_48") (Fin.cases ("candidate_vto_copy_49") (Fin.cases ("candidate_vbmm_15") (Fin.cases ("candidate_vmul_28") (Fin.cases ("candidate_vadd_13") (Fin.cases ("candidate_vsub_9") (Fin.cases ("candidate_vgt_5") (Fin.cases ("candidate_vmasked_fill_7") (Fin.cases ("candidate_vsub_10") (Fin.cases ("candidate_vexp_1") (Fin.cases ("candidate_vmul_29") (Fin.cases ("candidate_vsum_2") (Fin.cases ("candidate_vbmm_16") (Fin.cases ("candidate_vsub_11") (Fin.cases ("candidate_vmul_30") (Fin.cases ("candidate_vbmm_18") (Fin.cases ("candidate_vmul_32") (Fin.cases ("candidate_vto_copy_51") (Fin.cases ("candidate_vbmm_19") (Fin.cases ("candidate_vto_copy_52") (Fin.cases ("candidate_vcat_8") (Fin.cases ("candidate_vcat_9") (Fin.cases ("candidate_vbmm_22") (Fin.cases ("candidate_vbmm_23") (Fin.cases ("candidate_vcat_10") (Fin.elim0)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
  iwin := fun _ _ _ _ _ j => j.val
  imask := fun _ _ _ _ _ _ => True
  owin := fun _ _ _ _ _ j => j.val
  omask := fun _ _ _ _ _ _ => True
  swin := fun _ _ _ _ _ j => j.val
  smask := fun _ _ _ _ _ _ => True

def inputs : core.Ctx ≃ Inputs where
  toFun := fun v => ⟨parameter0Equiv (v (0 : Fin 9)), parameter1Equiv (v (1 : Fin 9)), parameter2Equiv (v (2 : Fin 9)), parameter3Equiv (v (3 : Fin 9)), parameter4Equiv (v (4 : Fin 9)), parameter5Equiv (v (5 : Fin 9)), parameter6Equiv (v (6 : Fin 9)), input0Equiv (v (7 : Fin 9)), input1Equiv (v (8 : Fin 9))⟩
  invFun := fun v => Fin.cases (parameter0Equiv.symm v.parameter0) (Fin.cases (parameter1Equiv.symm v.parameter1) (Fin.cases (parameter2Equiv.symm v.parameter2) (Fin.cases (parameter3Equiv.symm v.parameter3) (Fin.cases (parameter4Equiv.symm v.parameter4) (Fin.cases (parameter5Equiv.symm v.parameter5) (Fin.cases (parameter6Equiv.symm v.parameter6) (Fin.cases (input0Equiv.symm v.input0) (Fin.cases (input1Equiv.symm v.input1) ((fun i => Fin.elim0 i))))))))))
  left_inv := by
    intro v
    funext i
    fin_cases i <;> exact Equiv.symm_apply_apply _ _
  right_inv := by
    intro v
    change (⟨parameter0Equiv (parameter0Equiv.symm v.parameter0), parameter1Equiv (parameter1Equiv.symm v.parameter1), parameter2Equiv (parameter2Equiv.symm v.parameter2), parameter3Equiv (parameter3Equiv.symm v.parameter3), parameter4Equiv (parameter4Equiv.symm v.parameter4), parameter5Equiv (parameter5Equiv.symm v.parameter5), parameter6Equiv (parameter6Equiv.symm v.parameter6), input0Equiv (input0Equiv.symm v.input0), input1Equiv (input1Equiv.symm v.input1)⟩ : Inputs) = v
    simp only [Equiv.apply_symm_apply]

def outputCoords : Fin (2048*512) ≃ OutputIndex :=
  (finProdFinEquiv (m := 2048) (n := 512)).symm

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
  Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.elim0)))))))))

def scratchStorage : Fin core.nScr → TileDType :=
  Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bool) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.bool) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.bool) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.bool) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.bool) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.elim0)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

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
end VeriTile.Bench.PretrainKernels.local_mla_fa4_absorb_wgrad
