import bench.pretrain_kernels.tasks.local_masked_moe_global_aux_dx.Kernels
import VeriTile.Triton.Memory.LogicalScheduledKernelIO
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Float.GuardedProgramIO
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Tactic.FinCases

namespace VeriTile.Bench.PretrainKernels.local_masked_moe_global_aux_dx
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 2000000
namespace IO

structure Inputs where
  constantf1ebabb642 : Fin 4 → Int
  constant0d12534efc : Fin 4 → Int
  input0 : Fin 32 → Fin 16 → ℝ

abbrev OutputIndex := Fin 32 × Fin 16

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

def constantf1ebabb642Equiv : (Fin (4) → Int) ≃ (Fin 4 → Int) :=
  (Equiv.refl (Fin 4 → Int))

def constant0d12534efcEquiv : (Fin (4) → Int) ≃ (Fin 4 → Int) :=
  (Equiv.refl (Fin 4 → Int))

def input0Equiv : (Fin (32*16) → ℝ) ≃ (Fin 32 → Fin 16 → ℝ) :=
  ((matrixEquiv 32 (16) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 16 → ℝ))))

def core : UKernelIO where
  kernel := Kernels.realOriginal.eraseDType
  projection := by
    have h : Kernels.realOriginal.toAlgorithm? = .ok Kernels.realOriginal.toAlgKernel := rfl
    simp only [ComputeKernel.eraseDType, h]
    simp [ComputeKernel.toAlgKernel]
  nIn := 3
  nOut := 1
  nScr := 128
  bufs := ["constantf1ebabb642", "constant0d12534efc", "input0", "out", "literal_nan0", "literal_nan1", "vge", "vge_1", "vto_copy", "vsigmoid", "vsum_1", "vadd", "vdiv", "visnan", "vwhere", "vsort_value", "vsort_index", "vto_copy_1", "vscatter", "vto_copy_2", "vbitwise_not", "vbitwise_and", "vsum_2", "vto_copy_3", "vsigmoid_1", "vsum_7", "vadd_1", "vdiv_3", "visnan_1", "vwhere_1", "vsort_1_value", "vsort_1_index", "vto_copy_4", "vscatter_1", "vto_copy_5", "vbitwise_not_1", "vbitwise_and_1", "vmul_8", "vsum_8", "vsum_9", "vfloor_divide_2", "vadd_2", "vsum_10", "vfloor_divide_3", "vmul_9", "vclamp_min_2", "vdiv_4", "vdiv_5", "vmul_10", "vclamp_min_3", "vsum_11", "vmul_11", "vsum_12", "vmul_12", "vmul_13", "vreciprocal_1", "vmul_14", "vmul_15", "vmul_16", "vmul_17", "vmul_18", "vdiv_6", "vdiv_7", "vneg", "vmul_19", "vdiv_8", "vsum_13", "vadd_3", "vsigmoid_backward", "vto_copy_6", "candidate_literal_nan0", "candidate_literal_nan1", "candidate_vge", "candidate_vge_1", "candidate_vto_copy", "candidate_vsigmoid", "candidate_vsum_1", "candidate_vadd", "candidate_vdiv", "candidate_visnan", "candidate_vwhere", "candidate_vsort_value", "candidate_vsort_index", "candidate_vto_copy_1", "candidate_vscatter", "candidate_vto_copy_2", "candidate_vbitwise_not", "candidate_vmul", "candidate_vsum_2", "candidate_vto_copy_3", "candidate_vsigmoid_1", "candidate_vsum_7", "candidate_vadd_1", "candidate_vdiv_1", "candidate_visnan_1", "candidate_vwhere_1", "candidate_vsort_1_value", "candidate_vsort_1_index", "candidate_vto_copy_4", "candidate_vscatter_1", "candidate_vto_copy_5", "candidate_vbitwise_not_1", "candidate_vmul_7", "candidate_vmul_8", "candidate_vsum_8", "candidate_vsum_9", "candidate_vfloor_divide_2", "candidate_vclamp_min_2", "candidate_vadd_2", "candidate_vsum_10", "candidate_vfloor_divide_3", "candidate_vclamp_min_3", "candidate_vsum_11", "candidate_vmul_9", "candidate_vsum_12", "candidate_vmul_10", "candidate_vmul_11", "candidate_vreciprocal_1", "candidate_vmul_12", "candidate_vmul_13", "candidate_vmul_14", "candidate_vmul_15", "candidate_vmul_16", "candidate_vdiv_2", "candidate_vdiv_3", "candidate_vneg", "candidate_vmul_17", "candidate_vdiv_4", "candidate_vsum_13", "candidate_vadd_3", "candidate_vsigmoid_backward", "candidate_vto_copy_6"]
  ity := Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.float) (Fin.elim0)))
  iarity := Fin.cases (4) (Fin.cases (4) (Fin.cases (32*16) (Fin.elim0)))
  ibuf := Fin.cases ("constantf1ebabb642") (Fin.cases ("constant0d12534efc") (Fin.cases ("input0") (Fin.elim0)))
  oty := fun _ => .float
  oarity := fun _ => 32*16
  obuf := fun _ => "out"
  obuf_mem := fun _ => by simp
  sarity := Fin.cases (1) (Fin.cases (1) (Fin.cases (4*8) (Fin.cases (4*8) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*1) (Fin.cases (32*1) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32) (Fin.cases (32*16) (Fin.cases (16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*1) (Fin.cases (32*1) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (16) (Fin.cases (1) (Fin.cases (1) (Fin.cases (16) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (16) (Fin.cases (16) (Fin.cases (1) (Fin.cases (16) (Fin.cases (16) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*1) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (1) (Fin.cases (1) (Fin.cases (4*8) (Fin.cases (4*8) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*1) (Fin.cases (32*1) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32) (Fin.cases (32*16) (Fin.cases (16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*1) (Fin.cases (32*1) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (16) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (16) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (16) (Fin.cases (16) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*1) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.elim0))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
  sbuf := Fin.cases ("literal_nan0") (Fin.cases ("literal_nan1") (Fin.cases ("vge") (Fin.cases ("vge_1") (Fin.cases ("vto_copy") (Fin.cases ("vsigmoid") (Fin.cases ("vsum_1") (Fin.cases ("vadd") (Fin.cases ("vdiv") (Fin.cases ("visnan") (Fin.cases ("vwhere") (Fin.cases ("vsort_value") (Fin.cases ("vsort_index") (Fin.cases ("vto_copy_1") (Fin.cases ("vscatter") (Fin.cases ("vto_copy_2") (Fin.cases ("vbitwise_not") (Fin.cases ("vbitwise_and") (Fin.cases ("vsum_2") (Fin.cases ("vto_copy_3") (Fin.cases ("vsigmoid_1") (Fin.cases ("vsum_7") (Fin.cases ("vadd_1") (Fin.cases ("vdiv_3") (Fin.cases ("visnan_1") (Fin.cases ("vwhere_1") (Fin.cases ("vsort_1_value") (Fin.cases ("vsort_1_index") (Fin.cases ("vto_copy_4") (Fin.cases ("vscatter_1") (Fin.cases ("vto_copy_5") (Fin.cases ("vbitwise_not_1") (Fin.cases ("vbitwise_and_1") (Fin.cases ("vmul_8") (Fin.cases ("vsum_8") (Fin.cases ("vsum_9") (Fin.cases ("vfloor_divide_2") (Fin.cases ("vadd_2") (Fin.cases ("vsum_10") (Fin.cases ("vfloor_divide_3") (Fin.cases ("vmul_9") (Fin.cases ("vclamp_min_2") (Fin.cases ("vdiv_4") (Fin.cases ("vdiv_5") (Fin.cases ("vmul_10") (Fin.cases ("vclamp_min_3") (Fin.cases ("vsum_11") (Fin.cases ("vmul_11") (Fin.cases ("vsum_12") (Fin.cases ("vmul_12") (Fin.cases ("vmul_13") (Fin.cases ("vreciprocal_1") (Fin.cases ("vmul_14") (Fin.cases ("vmul_15") (Fin.cases ("vmul_16") (Fin.cases ("vmul_17") (Fin.cases ("vmul_18") (Fin.cases ("vdiv_6") (Fin.cases ("vdiv_7") (Fin.cases ("vneg") (Fin.cases ("vmul_19") (Fin.cases ("vdiv_8") (Fin.cases ("vsum_13") (Fin.cases ("vadd_3") (Fin.cases ("vsigmoid_backward") (Fin.cases ("vto_copy_6") (Fin.cases ("candidate_literal_nan0") (Fin.cases ("candidate_literal_nan1") (Fin.cases ("candidate_vge") (Fin.cases ("candidate_vge_1") (Fin.cases ("candidate_vto_copy") (Fin.cases ("candidate_vsigmoid") (Fin.cases ("candidate_vsum_1") (Fin.cases ("candidate_vadd") (Fin.cases ("candidate_vdiv") (Fin.cases ("candidate_visnan") (Fin.cases ("candidate_vwhere") (Fin.cases ("candidate_vsort_value") (Fin.cases ("candidate_vsort_index") (Fin.cases ("candidate_vto_copy_1") (Fin.cases ("candidate_vscatter") (Fin.cases ("candidate_vto_copy_2") (Fin.cases ("candidate_vbitwise_not") (Fin.cases ("candidate_vmul") (Fin.cases ("candidate_vsum_2") (Fin.cases ("candidate_vto_copy_3") (Fin.cases ("candidate_vsigmoid_1") (Fin.cases ("candidate_vsum_7") (Fin.cases ("candidate_vadd_1") (Fin.cases ("candidate_vdiv_1") (Fin.cases ("candidate_visnan_1") (Fin.cases ("candidate_vwhere_1") (Fin.cases ("candidate_vsort_1_value") (Fin.cases ("candidate_vsort_1_index") (Fin.cases ("candidate_vto_copy_4") (Fin.cases ("candidate_vscatter_1") (Fin.cases ("candidate_vto_copy_5") (Fin.cases ("candidate_vbitwise_not_1") (Fin.cases ("candidate_vmul_7") (Fin.cases ("candidate_vmul_8") (Fin.cases ("candidate_vsum_8") (Fin.cases ("candidate_vsum_9") (Fin.cases ("candidate_vfloor_divide_2") (Fin.cases ("candidate_vclamp_min_2") (Fin.cases ("candidate_vadd_2") (Fin.cases ("candidate_vsum_10") (Fin.cases ("candidate_vfloor_divide_3") (Fin.cases ("candidate_vclamp_min_3") (Fin.cases ("candidate_vsum_11") (Fin.cases ("candidate_vmul_9") (Fin.cases ("candidate_vsum_12") (Fin.cases ("candidate_vmul_10") (Fin.cases ("candidate_vmul_11") (Fin.cases ("candidate_vreciprocal_1") (Fin.cases ("candidate_vmul_12") (Fin.cases ("candidate_vmul_13") (Fin.cases ("candidate_vmul_14") (Fin.cases ("candidate_vmul_15") (Fin.cases ("candidate_vmul_16") (Fin.cases ("candidate_vdiv_2") (Fin.cases ("candidate_vdiv_3") (Fin.cases ("candidate_vneg") (Fin.cases ("candidate_vmul_17") (Fin.cases ("candidate_vdiv_4") (Fin.cases ("candidate_vsum_13") (Fin.cases ("candidate_vadd_3") (Fin.cases ("candidate_vsigmoid_backward") (Fin.cases ("candidate_vto_copy_6") (Fin.elim0))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
  iwin := fun _ _ _ _ _ j => j.val
  imask := fun _ _ _ _ _ _ => True
  owin := fun _ _ _ _ _ j => j.val
  omask := fun _ _ _ _ _ _ => True
  swin := fun _ _ _ _ _ j => j.val
  smask := fun _ _ _ _ _ _ => True

def inputs : core.Ctx ≃ Inputs where
  toFun := fun v => ⟨constantf1ebabb642Equiv (v (0 : Fin 3)), constant0d12534efcEquiv (v (1 : Fin 3)), input0Equiv (v (2 : Fin 3))⟩
  invFun := fun v => Fin.cases (constantf1ebabb642Equiv.symm v.constantf1ebabb642) (Fin.cases (constant0d12534efcEquiv.symm v.constant0d12534efc) (Fin.cases (input0Equiv.symm v.input0) ((fun i => Fin.elim0 i))))
  left_inv := by
    intro v
    funext i
    fin_cases i <;> exact Equiv.symm_apply_apply _ _
  right_inv := by
    intro v
    change (⟨constantf1ebabb642Equiv (constantf1ebabb642Equiv.symm v.constantf1ebabb642), constant0d12534efcEquiv (constant0d12534efcEquiv.symm v.constant0d12534efc), input0Equiv (input0Equiv.symm v.input0)⟩ : Inputs) = v
    simp only [Equiv.apply_symm_apply]

def outputCoords : Fin (32*16) ≃ OutputIndex :=
  (finProdFinEquiv (m := 32) (n := 16)).symm

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
  Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.bf16) (Fin.elim0)))

def scratchStorage : Fin core.nScr → TileDType :=
  Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bool) (Fin.cases (.bool) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bool) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.bool) (Fin.cases (.bool) (Fin.cases (.bool) (Fin.cases (.int) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bool) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.bool) (Fin.cases (.bool) (Fin.cases (.bool) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bool) (Fin.cases (.bool) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bool) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.bool) (Fin.cases (.bool) (Fin.cases (.bool) (Fin.cases (.int) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bool) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.bool) (Fin.cases (.bool) (Fin.cases (.bool) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.elim0))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

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
  (∀ j : Fin 4, vals (0 : Fin 3) (j, PUnit.unit) = ([1, 3, 0, 6] : List Int).getD j.val 0) ∧
    (∀ j : Fin 4, vals (1 : Fin 3) (j, PUnit.unit) = ([5, 2, 0, 1] : List Int).getD j.val 0)

end FP
end IO
end
end VeriTile.Bench.PretrainKernels.local_masked_moe_global_aux_dx
