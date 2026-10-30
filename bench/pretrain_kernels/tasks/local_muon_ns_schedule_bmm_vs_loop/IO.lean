import bench.pretrain_kernels.tasks.local_muon_ns_schedule_bmm_vs_loop.Kernels
import VeriTile.Triton.Memory.LogicalScheduledKernelIO
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Float.GuardedProgramIO
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Tactic.FinCases

namespace VeriTile.Bench.PretrainKernels.local_muon_ns_schedule_bmm_vs_loop
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 2000000
namespace IO

structure Inputs where
  parameter0 : Fin 4 → Fin 256 → Fin 128 → ℝ
  input0 : Fin 4 → Fin 256 → Fin 128 → ℝ

abbrev OutputIndex := Fin 4 × Fin 256 × Fin 128

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

def parameter0Equiv : (Fin (4*256*128) → ℝ) ≃ (Fin 4 → Fin 256 → Fin 128 → ℝ) :=
  ((matrixEquiv 4 (256*128) ℝ).trans (Equiv.piCongrRight fun _ => ((matrixEquiv 256 (128) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 128 → ℝ))))))

def input0Equiv : (Fin (4*256*128) → ℝ) ≃ (Fin 4 → Fin 256 → Fin 128 → ℝ) :=
  ((matrixEquiv 4 (256*128) ℝ).trans (Equiv.piCongrRight fun _ => ((matrixEquiv 256 (128) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 128 → ℝ))))))

def core : UKernelIO where
  kernel := Kernels.original.eraseDType
  projection := by
    have h : Kernels.original.toAlgorithm? = .ok Kernels.original.toAlgKernel := rfl
    simp only [ComputeKernel.eraseDType, h]
    simp [ComputeKernel.toAlgKernel]
  nIn := 2
  nOut := 1
  nScr := 197
  bufs := ["parameter0", "input0", "out", "vlerp", "vselect_scatter", "vlerp_1", "vlinalg_vector_norm", "vclamp", "vdiv", "vmm", "vaddmm", "vaddmm_1", "vmm_1", "vaddmm_2", "vaddmm_3", "vmm_2", "vaddmm_4", "vaddmm_5", "vmm_3", "vaddmm_6", "vaddmm_7", "vmm_4", "vaddmm_8", "vaddmm_9", "vmm_5", "vaddmm_10", "vaddmm_11", "vmm_6", "vaddmm_12", "vaddmm_13", "vmm_7", "vaddmm_14", "vaddmm_15", "vmm_8", "vaddmm_16", "vaddmm_17", "vmm_9", "vaddmm_18", "vaddmm_19", "vmul", "vselect_scatter_1", "vadd", "vselect_scatter_2", "vlerp_2", "vselect_scatter_3", "vlerp_3", "vlinalg_vector_norm_1", "vclamp_1", "vdiv_1", "vmm_10", "vaddmm_20", "vaddmm_21", "vmm_11", "vaddmm_22", "vaddmm_23", "vmm_12", "vaddmm_24", "vaddmm_25", "vmm_13", "vaddmm_26", "vaddmm_27", "vmm_14", "vaddmm_28", "vaddmm_29", "vmm_15", "vaddmm_30", "vaddmm_31", "vmm_16", "vaddmm_32", "vaddmm_33", "vmm_17", "vaddmm_34", "vaddmm_35", "vmm_18", "vaddmm_36", "vaddmm_37", "vmm_19", "vaddmm_38", "vaddmm_39", "vmul_1", "vselect_scatter_4", "vadd_1", "vselect_scatter_5", "vlerp_4", "vselect_scatter_6", "vlerp_5", "vlinalg_vector_norm_2", "vclamp_2", "vdiv_2", "vmm_20", "vaddmm_40", "vaddmm_41", "vmm_21", "vaddmm_42", "vaddmm_43", "vmm_22", "vaddmm_44", "vaddmm_45", "vmm_23", "vaddmm_46", "vaddmm_47", "vmm_24", "vaddmm_48", "vaddmm_49", "vmm_25", "vaddmm_50", "vaddmm_51", "vmm_26", "vaddmm_52", "vaddmm_53", "vmm_27", "vaddmm_54", "vaddmm_55", "vmm_28", "vaddmm_56", "vaddmm_57", "vmm_29", "vaddmm_58", "vaddmm_59", "vmul_2", "vselect_scatter_7", "vadd_2", "vselect_scatter_8", "vlerp_6", "vselect_scatter_9", "vlerp_7", "vlinalg_vector_norm_3", "vclamp_3", "vdiv_3", "vmm_30", "vaddmm_60", "vaddmm_61", "vmm_31", "vaddmm_62", "vaddmm_63", "vmm_32", "vaddmm_64", "vaddmm_65", "vmm_33", "vaddmm_66", "vaddmm_67", "vmm_34", "vaddmm_68", "vaddmm_69", "vmm_35", "vaddmm_70", "vaddmm_71", "vmm_36", "vaddmm_72", "vaddmm_73", "vmm_37", "vaddmm_74", "vaddmm_75", "vmm_38", "vaddmm_76", "vaddmm_77", "vmm_39", "vaddmm_78", "vaddmm_79", "vmul_3", "vselect_scatter_10", "vadd_3", "vselect_scatter_11", "candidate_vlerp", "candidate_vlerp_1", "candidate_vlinalg_vector_norm", "candidate_vclamp", "candidate_vdiv", "candidate_vbmm", "candidate_vbaddbmm", "candidate_vbaddbmm_1", "candidate_vbmm_1", "candidate_vbaddbmm_2", "candidate_vbaddbmm_3", "candidate_vbmm_2", "candidate_vbaddbmm_4", "candidate_vbaddbmm_5", "candidate_vbmm_3", "candidate_vbaddbmm_6", "candidate_vbaddbmm_7", "candidate_vbmm_4", "candidate_vbaddbmm_8", "candidate_vbaddbmm_9", "candidate_vbmm_5", "candidate_vbaddbmm_10", "candidate_vbaddbmm_11", "candidate_vbmm_6", "candidate_vbaddbmm_12", "candidate_vbaddbmm_13", "candidate_vbmm_7", "candidate_vbaddbmm_14", "candidate_vbaddbmm_15", "candidate_vbmm_8", "candidate_vbaddbmm_16", "candidate_vbaddbmm_17", "candidate_vbmm_9", "candidate_vbaddbmm_18", "candidate_vbaddbmm_19", "candidate_vmul", "candidate_vadd"]
  ity := Fin.cases (.float) (Fin.cases (.float) (Fin.elim0))
  iarity := Fin.cases (4*256*128) (Fin.cases (4*256*128) (Fin.elim0))
  ibuf := Fin.cases ("parameter0") (Fin.cases ("input0") (Fin.elim0))
  oty := fun _ => .float
  oarity := fun _ => 4*256*128
  obuf := fun _ => "out"
  obuf_mem := fun _ => by simp
  sarity := Fin.cases (256*128) (Fin.cases (4*256*128) (Fin.cases (256*128) (Fin.cases (1) (Fin.cases (1) (Fin.cases (128*256) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*256) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*256) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*256) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*256) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*256) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*256) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*256) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*256) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*256) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*256) (Fin.cases (256*128) (Fin.cases (4*256*128) (Fin.cases (256*128) (Fin.cases (4*256*128) (Fin.cases (256*128) (Fin.cases (4*256*128) (Fin.cases (256*128) (Fin.cases (1) (Fin.cases (1) (Fin.cases (128*256) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*256) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*256) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*256) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*256) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*256) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*256) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*256) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*256) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*256) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*256) (Fin.cases (256*128) (Fin.cases (4*256*128) (Fin.cases (256*128) (Fin.cases (4*256*128) (Fin.cases (256*128) (Fin.cases (4*256*128) (Fin.cases (256*128) (Fin.cases (1) (Fin.cases (1) (Fin.cases (128*256) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*256) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*256) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*256) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*256) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*256) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*256) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*256) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*256) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*256) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*256) (Fin.cases (256*128) (Fin.cases (4*256*128) (Fin.cases (256*128) (Fin.cases (4*256*128) (Fin.cases (256*128) (Fin.cases (4*256*128) (Fin.cases (256*128) (Fin.cases (1) (Fin.cases (1) (Fin.cases (128*256) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*256) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*256) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*256) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*256) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*256) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*256) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*256) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*256) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*256) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*256) (Fin.cases (256*128) (Fin.cases (4*256*128) (Fin.cases (256*128) (Fin.cases (4*256*128) (Fin.cases (4*256*128) (Fin.cases (4*256*128) (Fin.cases (4*1*1) (Fin.cases (4*1*1) (Fin.cases (4*128*256) (Fin.cases (4*128*128) (Fin.cases (4*128*128) (Fin.cases (4*128*256) (Fin.cases (4*128*128) (Fin.cases (4*128*128) (Fin.cases (4*128*256) (Fin.cases (4*128*128) (Fin.cases (4*128*128) (Fin.cases (4*128*256) (Fin.cases (4*128*128) (Fin.cases (4*128*128) (Fin.cases (4*128*256) (Fin.cases (4*128*128) (Fin.cases (4*128*128) (Fin.cases (4*128*256) (Fin.cases (4*128*128) (Fin.cases (4*128*128) (Fin.cases (4*128*256) (Fin.cases (4*128*128) (Fin.cases (4*128*128) (Fin.cases (4*128*256) (Fin.cases (4*128*128) (Fin.cases (4*128*128) (Fin.cases (4*128*256) (Fin.cases (4*128*128) (Fin.cases (4*128*128) (Fin.cases (4*128*256) (Fin.cases (4*128*128) (Fin.cases (4*128*128) (Fin.cases (4*128*256) (Fin.cases (4*256*128) (Fin.cases (4*256*128) (Fin.elim0)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
  sbuf := Fin.cases ("vlerp") (Fin.cases ("vselect_scatter") (Fin.cases ("vlerp_1") (Fin.cases ("vlinalg_vector_norm") (Fin.cases ("vclamp") (Fin.cases ("vdiv") (Fin.cases ("vmm") (Fin.cases ("vaddmm") (Fin.cases ("vaddmm_1") (Fin.cases ("vmm_1") (Fin.cases ("vaddmm_2") (Fin.cases ("vaddmm_3") (Fin.cases ("vmm_2") (Fin.cases ("vaddmm_4") (Fin.cases ("vaddmm_5") (Fin.cases ("vmm_3") (Fin.cases ("vaddmm_6") (Fin.cases ("vaddmm_7") (Fin.cases ("vmm_4") (Fin.cases ("vaddmm_8") (Fin.cases ("vaddmm_9") (Fin.cases ("vmm_5") (Fin.cases ("vaddmm_10") (Fin.cases ("vaddmm_11") (Fin.cases ("vmm_6") (Fin.cases ("vaddmm_12") (Fin.cases ("vaddmm_13") (Fin.cases ("vmm_7") (Fin.cases ("vaddmm_14") (Fin.cases ("vaddmm_15") (Fin.cases ("vmm_8") (Fin.cases ("vaddmm_16") (Fin.cases ("vaddmm_17") (Fin.cases ("vmm_9") (Fin.cases ("vaddmm_18") (Fin.cases ("vaddmm_19") (Fin.cases ("vmul") (Fin.cases ("vselect_scatter_1") (Fin.cases ("vadd") (Fin.cases ("vselect_scatter_2") (Fin.cases ("vlerp_2") (Fin.cases ("vselect_scatter_3") (Fin.cases ("vlerp_3") (Fin.cases ("vlinalg_vector_norm_1") (Fin.cases ("vclamp_1") (Fin.cases ("vdiv_1") (Fin.cases ("vmm_10") (Fin.cases ("vaddmm_20") (Fin.cases ("vaddmm_21") (Fin.cases ("vmm_11") (Fin.cases ("vaddmm_22") (Fin.cases ("vaddmm_23") (Fin.cases ("vmm_12") (Fin.cases ("vaddmm_24") (Fin.cases ("vaddmm_25") (Fin.cases ("vmm_13") (Fin.cases ("vaddmm_26") (Fin.cases ("vaddmm_27") (Fin.cases ("vmm_14") (Fin.cases ("vaddmm_28") (Fin.cases ("vaddmm_29") (Fin.cases ("vmm_15") (Fin.cases ("vaddmm_30") (Fin.cases ("vaddmm_31") (Fin.cases ("vmm_16") (Fin.cases ("vaddmm_32") (Fin.cases ("vaddmm_33") (Fin.cases ("vmm_17") (Fin.cases ("vaddmm_34") (Fin.cases ("vaddmm_35") (Fin.cases ("vmm_18") (Fin.cases ("vaddmm_36") (Fin.cases ("vaddmm_37") (Fin.cases ("vmm_19") (Fin.cases ("vaddmm_38") (Fin.cases ("vaddmm_39") (Fin.cases ("vmul_1") (Fin.cases ("vselect_scatter_4") (Fin.cases ("vadd_1") (Fin.cases ("vselect_scatter_5") (Fin.cases ("vlerp_4") (Fin.cases ("vselect_scatter_6") (Fin.cases ("vlerp_5") (Fin.cases ("vlinalg_vector_norm_2") (Fin.cases ("vclamp_2") (Fin.cases ("vdiv_2") (Fin.cases ("vmm_20") (Fin.cases ("vaddmm_40") (Fin.cases ("vaddmm_41") (Fin.cases ("vmm_21") (Fin.cases ("vaddmm_42") (Fin.cases ("vaddmm_43") (Fin.cases ("vmm_22") (Fin.cases ("vaddmm_44") (Fin.cases ("vaddmm_45") (Fin.cases ("vmm_23") (Fin.cases ("vaddmm_46") (Fin.cases ("vaddmm_47") (Fin.cases ("vmm_24") (Fin.cases ("vaddmm_48") (Fin.cases ("vaddmm_49") (Fin.cases ("vmm_25") (Fin.cases ("vaddmm_50") (Fin.cases ("vaddmm_51") (Fin.cases ("vmm_26") (Fin.cases ("vaddmm_52") (Fin.cases ("vaddmm_53") (Fin.cases ("vmm_27") (Fin.cases ("vaddmm_54") (Fin.cases ("vaddmm_55") (Fin.cases ("vmm_28") (Fin.cases ("vaddmm_56") (Fin.cases ("vaddmm_57") (Fin.cases ("vmm_29") (Fin.cases ("vaddmm_58") (Fin.cases ("vaddmm_59") (Fin.cases ("vmul_2") (Fin.cases ("vselect_scatter_7") (Fin.cases ("vadd_2") (Fin.cases ("vselect_scatter_8") (Fin.cases ("vlerp_6") (Fin.cases ("vselect_scatter_9") (Fin.cases ("vlerp_7") (Fin.cases ("vlinalg_vector_norm_3") (Fin.cases ("vclamp_3") (Fin.cases ("vdiv_3") (Fin.cases ("vmm_30") (Fin.cases ("vaddmm_60") (Fin.cases ("vaddmm_61") (Fin.cases ("vmm_31") (Fin.cases ("vaddmm_62") (Fin.cases ("vaddmm_63") (Fin.cases ("vmm_32") (Fin.cases ("vaddmm_64") (Fin.cases ("vaddmm_65") (Fin.cases ("vmm_33") (Fin.cases ("vaddmm_66") (Fin.cases ("vaddmm_67") (Fin.cases ("vmm_34") (Fin.cases ("vaddmm_68") (Fin.cases ("vaddmm_69") (Fin.cases ("vmm_35") (Fin.cases ("vaddmm_70") (Fin.cases ("vaddmm_71") (Fin.cases ("vmm_36") (Fin.cases ("vaddmm_72") (Fin.cases ("vaddmm_73") (Fin.cases ("vmm_37") (Fin.cases ("vaddmm_74") (Fin.cases ("vaddmm_75") (Fin.cases ("vmm_38") (Fin.cases ("vaddmm_76") (Fin.cases ("vaddmm_77") (Fin.cases ("vmm_39") (Fin.cases ("vaddmm_78") (Fin.cases ("vaddmm_79") (Fin.cases ("vmul_3") (Fin.cases ("vselect_scatter_10") (Fin.cases ("vadd_3") (Fin.cases ("vselect_scatter_11") (Fin.cases ("candidate_vlerp") (Fin.cases ("candidate_vlerp_1") (Fin.cases ("candidate_vlinalg_vector_norm") (Fin.cases ("candidate_vclamp") (Fin.cases ("candidate_vdiv") (Fin.cases ("candidate_vbmm") (Fin.cases ("candidate_vbaddbmm") (Fin.cases ("candidate_vbaddbmm_1") (Fin.cases ("candidate_vbmm_1") (Fin.cases ("candidate_vbaddbmm_2") (Fin.cases ("candidate_vbaddbmm_3") (Fin.cases ("candidate_vbmm_2") (Fin.cases ("candidate_vbaddbmm_4") (Fin.cases ("candidate_vbaddbmm_5") (Fin.cases ("candidate_vbmm_3") (Fin.cases ("candidate_vbaddbmm_6") (Fin.cases ("candidate_vbaddbmm_7") (Fin.cases ("candidate_vbmm_4") (Fin.cases ("candidate_vbaddbmm_8") (Fin.cases ("candidate_vbaddbmm_9") (Fin.cases ("candidate_vbmm_5") (Fin.cases ("candidate_vbaddbmm_10") (Fin.cases ("candidate_vbaddbmm_11") (Fin.cases ("candidate_vbmm_6") (Fin.cases ("candidate_vbaddbmm_12") (Fin.cases ("candidate_vbaddbmm_13") (Fin.cases ("candidate_vbmm_7") (Fin.cases ("candidate_vbaddbmm_14") (Fin.cases ("candidate_vbaddbmm_15") (Fin.cases ("candidate_vbmm_8") (Fin.cases ("candidate_vbaddbmm_16") (Fin.cases ("candidate_vbaddbmm_17") (Fin.cases ("candidate_vbmm_9") (Fin.cases ("candidate_vbaddbmm_18") (Fin.cases ("candidate_vbaddbmm_19") (Fin.cases ("candidate_vmul") (Fin.cases ("candidate_vadd") (Fin.elim0)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
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

def outputCoords : Fin (4*256*128) ≃ OutputIndex :=
  (finProdFinEquiv (m := 4) (n := 256*128)).symm.trans (Equiv.prodCongr (Equiv.refl _) ((finProdFinEquiv (m := 256) (n := 128)).symm))

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
  Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.elim0))

def scratchStorage : Fin core.nScr → TileDType :=
  Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.elim0)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

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
end VeriTile.Bench.PretrainKernels.local_muon_ns_schedule_bmm_vs_loop
