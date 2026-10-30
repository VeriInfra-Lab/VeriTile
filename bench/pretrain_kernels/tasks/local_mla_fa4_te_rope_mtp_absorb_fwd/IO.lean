import bench.pretrain_kernels.tasks.local_mla_fa4_te_rope_mtp_absorb_fwd.Kernels
import VeriTile.Triton.Memory.LogicalScheduledKernelIO
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Float.GuardedProgramIO
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Tactic.FinCases

namespace VeriTile.Bench.PretrainKernels.local_mla_fa4_te_rope_mtp_absorb_fwd
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
  input0 : Fin 1 → Fin 128 → Fin 512 → ℝ

abbrev OutputIndex := Fin 1 × Fin 128 × Fin 512

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

def input0Equiv : (Fin (1*128*512) → ℝ) ≃ (Fin 1 → Fin 128 → Fin 512 → ℝ) :=
  ((matrixEquiv 1 (128*512) ℝ).trans (Equiv.piCongrRight fun _ => ((matrixEquiv 128 (512) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 512 → ℝ))))))

def core : UKernelIO where
  kernel := Kernels.original.eraseDType
  projection := by
    have h : Kernels.original.toAlgorithm? = .ok Kernels.original.toAlgKernel := rfl
    simp only [ComputeKernel.eraseDType, h]
    simp [ComputeKernel.toAlgKernel]
  nIn := 8
  nOut := 1
  nScr := 194
  bufs := ["parameter0", "parameter1", "parameter2", "parameter3", "parameter4", "parameter5", "parameter6", "input0", "out", "vmul", "vmul_1", "vadd", "vmul_2", "vmul_3", "vmul_4", "vadd_1", "vmul_5", "vmul_6", "vto_copy", "vto_copy_1", "vto_copy_2", "vto_copy_3", "vto_copy_4", "vto_copy_5", "vto_copy_6", "vcat", "vadd_2", "vto_copy_14", "vdiv", "vpow_1", "vreciprocal", "vmul_7", "vto_copy_16", "vbmm", "vcat_1", "vcos", "vmul_9", "vsin", "vmul_10", "vto_copy_17", "vto_copy_18", "vsub", "vcat_3", "vsub_1", "vne", "vcumsum_1", "veq", "vtril", "vbitwise_and", "vwhere", "vmm", "vto_copy_20", "vpow_3", "vmean", "vadd_3", "vrsqrt", "vmul_13", "vmul_14", "vto_copy_21", "vmm_1", "vmm_2", "vto_copy_22", "vpow_4", "vmean_1", "vadd_4", "vrsqrt_1", "vmul_15", "vmul_16", "vto_copy_23", "vmm_3", "vmul_17", "vmul_18", "vsub_4", "vmul_19", "vmul_20", "vadd_5", "vcat_5", "vmul_21", "vmul_22", "vsub_5", "vmul_23", "vmul_24", "vadd_6", "vcat_6", "vcat_7", "vcat_8", "vbmm_2", "vmul_25", "vadd_7", "vto_copy_24", "vsoftmax", "vto_copy_25", "vbmm_3", "vmm_4", "candidate_vmul", "candidate_vmul_1", "candidate_vadd", "candidate_vmul_2", "candidate_vmul_3", "candidate_vmul_4", "candidate_vadd_1", "candidate_vmul_5", "candidate_vmul_6", "candidate_vto_copy", "candidate_vto_copy_1", "candidate_vto_copy_2", "candidate_vto_copy_3", "candidate_vto_copy_4", "candidate_vto_copy_5", "candidate_vto_copy_6", "candidate_vmm", "candidate_vto_copy_20", "candidate_vpow_3", "candidate_vmean", "candidate_vadd_3", "candidate_vrsqrt", "candidate_vmul_13", "candidate_vmul_14", "candidate_vto_copy_21", "candidate_vmm_1", "candidate_vmm_2", "candidate_vto_copy_22", "candidate_vpow_4", "candidate_vmean_1", "candidate_vadd_4", "candidate_vrsqrt_1", "candidate_vmul_15", "candidate_vmul_16", "candidate_vto_copy_23", "candidate_vdiv_2", "candidate_vpow_5", "candidate_vreciprocal_2", "candidate_vmul_17", "candidate_vmul_18", "candidate_vcat_5", "candidate_vcos_2", "candidate_vto_copy_24", "candidate_vsin_2", "candidate_vto_copy_25", "candidate_vneg", "candidate_vstack", "candidate_vmul_19", "candidate_vmul_20", "candidate_vadd_5", "candidate_vcat_6", "candidate_vcos_3", "candidate_vto_copy_26", "candidate_vsin_3", "candidate_vto_copy_27", "candidate_vneg_1", "candidate_vstack_1", "candidate_vmul_21", "candidate_vmul_22", "candidate_vadd_6", "candidate_vbmm_2", "candidate_vto_copy_28", "candidate_vto_copy_29", "candidate_vbmm_3", "candidate_vto_copy_30", "candidate_vto_copy_31", "candidate_vmm_3", "candidate_vadd_7", "candidate_vmul_23", "candidate_vadd_8", "candidate_vsub_4", "candidate_vgt", "candidate_vmasked_fill", "candidate_vsoftmax", "candidate_vto_copy_32", "candidate_vmm_4", "candidate_vto_copy_33", "candidate_vto_copy_34", "candidate_vbmm_4", "candidate_vto_copy_35", "candidate_vto_copy_36", "candidate_vmm_5", "candidate_vadd_9", "candidate_vmul_24", "candidate_vadd_10", "candidate_vsub_5", "candidate_vgt_1", "candidate_vmasked_fill_1", "candidate_vsoftmax_1", "candidate_vto_copy_37", "candidate_vmm_6", "candidate_vto_copy_38", "candidate_vto_copy_39", "candidate_vbmm_5", "candidate_vto_copy_40", "candidate_vto_copy_41", "candidate_vmm_7", "candidate_vadd_11", "candidate_vmul_25", "candidate_vadd_12", "candidate_vsub_6", "candidate_vgt_2", "candidate_vmasked_fill_2", "candidate_vsoftmax_2", "candidate_vto_copy_42", "candidate_vmm_8", "candidate_vcat_7", "candidate_vbmm_6", "candidate_vmm_9"]
  ity := Fin.cases (.float) (Fin.cases (.float) (Fin.cases (.float) (Fin.cases (.float) (Fin.cases (.float) (Fin.cases (.float) (Fin.cases (.float) (Fin.cases (.float) (Fin.elim0))))))))
  iarity := Fin.cases (256*512) (Fin.cases (256) (Fin.cases (1536*256) (Fin.cases (576*512) (Fin.cases (512) (Fin.cases (2048*512) (Fin.cases (512*1024) (Fin.cases (1*128*512) (Fin.elim0))))))))
  ibuf := Fin.cases ("parameter0") (Fin.cases ("parameter1") (Fin.cases ("parameter2") (Fin.cases ("parameter3") (Fin.cases ("parameter4") (Fin.cases ("parameter5") (Fin.cases ("parameter6") (Fin.cases ("input0") (Fin.elim0))))))))
  oty := fun _ => .float
  oarity := fun _ => 1*128*512
  obuf := fun _ => "out"
  obuf_mem := fun _ => by simp
  sarity := Fin.cases (256*512) (Fin.cases (256) (Fin.cases (256) (Fin.cases (1536*256) (Fin.cases (576*512) (Fin.cases (512) (Fin.cases (512) (Fin.cases (2048*512) (Fin.cases (512*1024) (Fin.cases (256*512) (Fin.cases (256) (Fin.cases (1536*256) (Fin.cases (576*512) (Fin.cases (512) (Fin.cases (2048*512) (Fin.cases (512*1024) (Fin.cases (128) (Fin.cases (1*128) (Fin.cases (32) (Fin.cases (32) (Fin.cases (32) (Fin.cases (32) (Fin.cases (32) (Fin.cases (1*1*128) (Fin.cases (1*32*128) (Fin.cases (1*128*64) (Fin.cases (1*128*64) (Fin.cases (1*128*64) (Fin.cases (1*128*64) (Fin.cases (1*128*64) (Fin.cases (1*128*64) (Fin.cases (1*128*64) (Fin.cases (1*1) (Fin.cases (1*129) (Fin.cases (1*128) (Fin.cases (1*128) (Fin.cases (1*128) (Fin.cases (1*1*128*128) (Fin.cases (128*128) (Fin.cases (1*1*128*128) (Fin.cases (1*1*128*128) (Fin.cases (128*256) (Fin.cases (1*128*256) (Fin.cases (1*128*256) (Fin.cases (1*128*1) (Fin.cases (1*128*1) (Fin.cases (1*128*1) (Fin.cases (1*128*256) (Fin.cases (1*128*256) (Fin.cases (1*128*256) (Fin.cases (128*1536) (Fin.cases (128*576) (Fin.cases (1*128*512) (Fin.cases (1*128*512) (Fin.cases (1*128*1) (Fin.cases (1*128*1) (Fin.cases (1*128*1) (Fin.cases (1*128*512) (Fin.cases (1*128*512) (Fin.cases (1*128*512) (Fin.cases (128*2048) (Fin.cases (1*8*128*32) (Fin.cases (1*8*128*32) (Fin.cases (1*8*128*32) (Fin.cases (1*8*128*32) (Fin.cases (1*8*128*32) (Fin.cases (1*8*128*32) (Fin.cases (1*8*128*64) (Fin.cases (1*1*128*32) (Fin.cases (1*1*128*32) (Fin.cases (1*1*128*32) (Fin.cases (1*1*128*32) (Fin.cases (1*1*128*32) (Fin.cases (1*1*128*32) (Fin.cases (1*1*128*64) (Fin.cases (1*8*128*192) (Fin.cases (1*8*128*192) (Fin.cases (8*128*128) (Fin.cases (1*8*128*128) (Fin.cases (1*8*128*128) (Fin.cases (1*8*128*128) (Fin.cases (1*8*128*128) (Fin.cases (1*8*128*128) (Fin.cases (8*128*128) (Fin.cases (128*512) (Fin.cases (256*512) (Fin.cases (256) (Fin.cases (256) (Fin.cases (1536*256) (Fin.cases (576*512) (Fin.cases (512) (Fin.cases (512) (Fin.cases (2048*512) (Fin.cases (512*1024) (Fin.cases (256*512) (Fin.cases (256) (Fin.cases (1536*256) (Fin.cases (576*512) (Fin.cases (512) (Fin.cases (2048*512) (Fin.cases (512*1024) (Fin.cases (128*256) (Fin.cases (1*128*256) (Fin.cases (1*128*256) (Fin.cases (1*128*1) (Fin.cases (1*128*1) (Fin.cases (1*128*1) (Fin.cases (1*128*256) (Fin.cases (1*128*256) (Fin.cases (1*128*256) (Fin.cases (128*1536) (Fin.cases (128*576) (Fin.cases (1*128*512) (Fin.cases (1*128*512) (Fin.cases (1*128*1) (Fin.cases (1*128*1) (Fin.cases (1*128*1) (Fin.cases (1*128*512) (Fin.cases (1*128*512) (Fin.cases (1*128*512) (Fin.cases (32) (Fin.cases (32) (Fin.cases (32) (Fin.cases (32) (Fin.cases (64*32) (Fin.cases (128*64) (Fin.cases (128*1*64) (Fin.cases (128*1*64) (Fin.cases (128*1*64) (Fin.cases (128*1*64) (Fin.cases (128*8*32) (Fin.cases (128*8*32*2) (Fin.cases (128*8*64) (Fin.cases (128*8*64) (Fin.cases (128*8*64) (Fin.cases (128*64) (Fin.cases (128*1*64) (Fin.cases (128*1*64) (Fin.cases (128*1*64) (Fin.cases (128*1*64) (Fin.cases (128*1*32) (Fin.cases (128*1*32*2) (Fin.cases (128*1*64) (Fin.cases (128*1*64) (Fin.cases (128*1*64) (Fin.cases (8*128*512) (Fin.cases (8*48*64) (Fin.cases (1*48*64) (Fin.cases (8*48*48) (Fin.cases (8*48*512) (Fin.cases (1*48*512) (Fin.cases (384*48) (Fin.cases (8*48*48) (Fin.cases (8*48*48) (Fin.cases (48*1) (Fin.cases (48*1) (Fin.cases (48*48) (Fin.cases (8*48*48) (Fin.cases (8*48*48) (Fin.cases (8*48*48) (Fin.cases (384*512) (Fin.cases (8*16*64) (Fin.cases (1*16*64) (Fin.cases (8*16*16) (Fin.cases (8*16*512) (Fin.cases (1*16*512) (Fin.cases (128*16) (Fin.cases (8*16*16) (Fin.cases (8*16*16) (Fin.cases (16*1) (Fin.cases (16*1) (Fin.cases (16*16) (Fin.cases (8*16*16) (Fin.cases (8*16*16) (Fin.cases (8*16*16) (Fin.cases (128*512) (Fin.cases (8*64*64) (Fin.cases (1*64*64) (Fin.cases (8*64*64) (Fin.cases (8*64*512) (Fin.cases (1*64*512) (Fin.cases (512*64) (Fin.cases (8*64*64) (Fin.cases (8*64*64) (Fin.cases (64*1) (Fin.cases (64*1) (Fin.cases (64*64) (Fin.cases (8*64*64) (Fin.cases (8*64*64) (Fin.cases (8*64*64) (Fin.cases (512*512) (Fin.cases (128*8*512) (Fin.cases (8*128*128) (Fin.cases (128*512) (Fin.elim0))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
  sbuf := Fin.cases ("vmul") (Fin.cases ("vmul_1") (Fin.cases ("vadd") (Fin.cases ("vmul_2") (Fin.cases ("vmul_3") (Fin.cases ("vmul_4") (Fin.cases ("vadd_1") (Fin.cases ("vmul_5") (Fin.cases ("vmul_6") (Fin.cases ("vto_copy") (Fin.cases ("vto_copy_1") (Fin.cases ("vto_copy_2") (Fin.cases ("vto_copy_3") (Fin.cases ("vto_copy_4") (Fin.cases ("vto_copy_5") (Fin.cases ("vto_copy_6") (Fin.cases ("vcat") (Fin.cases ("vadd_2") (Fin.cases ("vto_copy_14") (Fin.cases ("vdiv") (Fin.cases ("vpow_1") (Fin.cases ("vreciprocal") (Fin.cases ("vmul_7") (Fin.cases ("vto_copy_16") (Fin.cases ("vbmm") (Fin.cases ("vcat_1") (Fin.cases ("vcos") (Fin.cases ("vmul_9") (Fin.cases ("vsin") (Fin.cases ("vmul_10") (Fin.cases ("vto_copy_17") (Fin.cases ("vto_copy_18") (Fin.cases ("vsub") (Fin.cases ("vcat_3") (Fin.cases ("vsub_1") (Fin.cases ("vne") (Fin.cases ("vcumsum_1") (Fin.cases ("veq") (Fin.cases ("vtril") (Fin.cases ("vbitwise_and") (Fin.cases ("vwhere") (Fin.cases ("vmm") (Fin.cases ("vto_copy_20") (Fin.cases ("vpow_3") (Fin.cases ("vmean") (Fin.cases ("vadd_3") (Fin.cases ("vrsqrt") (Fin.cases ("vmul_13") (Fin.cases ("vmul_14") (Fin.cases ("vto_copy_21") (Fin.cases ("vmm_1") (Fin.cases ("vmm_2") (Fin.cases ("vto_copy_22") (Fin.cases ("vpow_4") (Fin.cases ("vmean_1") (Fin.cases ("vadd_4") (Fin.cases ("vrsqrt_1") (Fin.cases ("vmul_15") (Fin.cases ("vmul_16") (Fin.cases ("vto_copy_23") (Fin.cases ("vmm_3") (Fin.cases ("vmul_17") (Fin.cases ("vmul_18") (Fin.cases ("vsub_4") (Fin.cases ("vmul_19") (Fin.cases ("vmul_20") (Fin.cases ("vadd_5") (Fin.cases ("vcat_5") (Fin.cases ("vmul_21") (Fin.cases ("vmul_22") (Fin.cases ("vsub_5") (Fin.cases ("vmul_23") (Fin.cases ("vmul_24") (Fin.cases ("vadd_6") (Fin.cases ("vcat_6") (Fin.cases ("vcat_7") (Fin.cases ("vcat_8") (Fin.cases ("vbmm_2") (Fin.cases ("vmul_25") (Fin.cases ("vadd_7") (Fin.cases ("vto_copy_24") (Fin.cases ("vsoftmax") (Fin.cases ("vto_copy_25") (Fin.cases ("vbmm_3") (Fin.cases ("vmm_4") (Fin.cases ("candidate_vmul") (Fin.cases ("candidate_vmul_1") (Fin.cases ("candidate_vadd") (Fin.cases ("candidate_vmul_2") (Fin.cases ("candidate_vmul_3") (Fin.cases ("candidate_vmul_4") (Fin.cases ("candidate_vadd_1") (Fin.cases ("candidate_vmul_5") (Fin.cases ("candidate_vmul_6") (Fin.cases ("candidate_vto_copy") (Fin.cases ("candidate_vto_copy_1") (Fin.cases ("candidate_vto_copy_2") (Fin.cases ("candidate_vto_copy_3") (Fin.cases ("candidate_vto_copy_4") (Fin.cases ("candidate_vto_copy_5") (Fin.cases ("candidate_vto_copy_6") (Fin.cases ("candidate_vmm") (Fin.cases ("candidate_vto_copy_20") (Fin.cases ("candidate_vpow_3") (Fin.cases ("candidate_vmean") (Fin.cases ("candidate_vadd_3") (Fin.cases ("candidate_vrsqrt") (Fin.cases ("candidate_vmul_13") (Fin.cases ("candidate_vmul_14") (Fin.cases ("candidate_vto_copy_21") (Fin.cases ("candidate_vmm_1") (Fin.cases ("candidate_vmm_2") (Fin.cases ("candidate_vto_copy_22") (Fin.cases ("candidate_vpow_4") (Fin.cases ("candidate_vmean_1") (Fin.cases ("candidate_vadd_4") (Fin.cases ("candidate_vrsqrt_1") (Fin.cases ("candidate_vmul_15") (Fin.cases ("candidate_vmul_16") (Fin.cases ("candidate_vto_copy_23") (Fin.cases ("candidate_vdiv_2") (Fin.cases ("candidate_vpow_5") (Fin.cases ("candidate_vreciprocal_2") (Fin.cases ("candidate_vmul_17") (Fin.cases ("candidate_vmul_18") (Fin.cases ("candidate_vcat_5") (Fin.cases ("candidate_vcos_2") (Fin.cases ("candidate_vto_copy_24") (Fin.cases ("candidate_vsin_2") (Fin.cases ("candidate_vto_copy_25") (Fin.cases ("candidate_vneg") (Fin.cases ("candidate_vstack") (Fin.cases ("candidate_vmul_19") (Fin.cases ("candidate_vmul_20") (Fin.cases ("candidate_vadd_5") (Fin.cases ("candidate_vcat_6") (Fin.cases ("candidate_vcos_3") (Fin.cases ("candidate_vto_copy_26") (Fin.cases ("candidate_vsin_3") (Fin.cases ("candidate_vto_copy_27") (Fin.cases ("candidate_vneg_1") (Fin.cases ("candidate_vstack_1") (Fin.cases ("candidate_vmul_21") (Fin.cases ("candidate_vmul_22") (Fin.cases ("candidate_vadd_6") (Fin.cases ("candidate_vbmm_2") (Fin.cases ("candidate_vto_copy_28") (Fin.cases ("candidate_vto_copy_29") (Fin.cases ("candidate_vbmm_3") (Fin.cases ("candidate_vto_copy_30") (Fin.cases ("candidate_vto_copy_31") (Fin.cases ("candidate_vmm_3") (Fin.cases ("candidate_vadd_7") (Fin.cases ("candidate_vmul_23") (Fin.cases ("candidate_vadd_8") (Fin.cases ("candidate_vsub_4") (Fin.cases ("candidate_vgt") (Fin.cases ("candidate_vmasked_fill") (Fin.cases ("candidate_vsoftmax") (Fin.cases ("candidate_vto_copy_32") (Fin.cases ("candidate_vmm_4") (Fin.cases ("candidate_vto_copy_33") (Fin.cases ("candidate_vto_copy_34") (Fin.cases ("candidate_vbmm_4") (Fin.cases ("candidate_vto_copy_35") (Fin.cases ("candidate_vto_copy_36") (Fin.cases ("candidate_vmm_5") (Fin.cases ("candidate_vadd_9") (Fin.cases ("candidate_vmul_24") (Fin.cases ("candidate_vadd_10") (Fin.cases ("candidate_vsub_5") (Fin.cases ("candidate_vgt_1") (Fin.cases ("candidate_vmasked_fill_1") (Fin.cases ("candidate_vsoftmax_1") (Fin.cases ("candidate_vto_copy_37") (Fin.cases ("candidate_vmm_6") (Fin.cases ("candidate_vto_copy_38") (Fin.cases ("candidate_vto_copy_39") (Fin.cases ("candidate_vbmm_5") (Fin.cases ("candidate_vto_copy_40") (Fin.cases ("candidate_vto_copy_41") (Fin.cases ("candidate_vmm_7") (Fin.cases ("candidate_vadd_11") (Fin.cases ("candidate_vmul_25") (Fin.cases ("candidate_vadd_12") (Fin.cases ("candidate_vsub_6") (Fin.cases ("candidate_vgt_2") (Fin.cases ("candidate_vmasked_fill_2") (Fin.cases ("candidate_vsoftmax_2") (Fin.cases ("candidate_vto_copy_42") (Fin.cases ("candidate_vmm_8") (Fin.cases ("candidate_vcat_7") (Fin.cases ("candidate_vbmm_6") (Fin.cases ("candidate_vmm_9") (Fin.elim0))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
  iwin := fun _ _ _ _ _ j => j.val
  imask := fun _ _ _ _ _ _ => True
  owin := fun _ _ _ _ _ j => j.val
  omask := fun _ _ _ _ _ _ => True
  swin := fun _ _ _ _ _ j => j.val
  smask := fun _ _ _ _ _ _ => True

def inputs : core.Ctx ≃ Inputs where
  toFun := fun v => ⟨parameter0Equiv (v (0 : Fin 8)), parameter1Equiv (v (1 : Fin 8)), parameter2Equiv (v (2 : Fin 8)), parameter3Equiv (v (3 : Fin 8)), parameter4Equiv (v (4 : Fin 8)), parameter5Equiv (v (5 : Fin 8)), parameter6Equiv (v (6 : Fin 8)), input0Equiv (v (7 : Fin 8))⟩
  invFun := fun v => Fin.cases (parameter0Equiv.symm v.parameter0) (Fin.cases (parameter1Equiv.symm v.parameter1) (Fin.cases (parameter2Equiv.symm v.parameter2) (Fin.cases (parameter3Equiv.symm v.parameter3) (Fin.cases (parameter4Equiv.symm v.parameter4) (Fin.cases (parameter5Equiv.symm v.parameter5) (Fin.cases (parameter6Equiv.symm v.parameter6) (Fin.cases (input0Equiv.symm v.input0) ((fun i => Fin.elim0 i)))))))))
  left_inv := by
    intro v
    funext i
    fin_cases i <;> exact Equiv.symm_apply_apply _ _
  right_inv := by
    intro v
    change (⟨parameter0Equiv (parameter0Equiv.symm v.parameter0), parameter1Equiv (parameter1Equiv.symm v.parameter1), parameter2Equiv (parameter2Equiv.symm v.parameter2), parameter3Equiv (parameter3Equiv.symm v.parameter3), parameter4Equiv (parameter4Equiv.symm v.parameter4), parameter5Equiv (parameter5Equiv.symm v.parameter5), parameter6Equiv (parameter6Equiv.symm v.parameter6), input0Equiv (input0Equiv.symm v.input0)⟩ : Inputs) = v
    simp only [Equiv.apply_symm_apply]

def outputCoords : Fin (1*128*512) ≃ OutputIndex :=
  (finProdFinEquiv (m := 1) (n := 128*512)).symm.trans (Equiv.prodCongr (Equiv.refl _) ((finProdFinEquiv (m := 128) (n := 512)).symm))

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
  Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.elim0))))))))

def scratchStorage : Fin core.nScr → TileDType :=
  Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.bool) (Fin.cases (.int) (Fin.cases (.bool) (Fin.cases (.bool) (Fin.cases (.bool) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.bool) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.bool) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.bool) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.elim0))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

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
end VeriTile.Bench.PretrainKernels.local_mla_fa4_te_rope_mtp_absorb_fwd
