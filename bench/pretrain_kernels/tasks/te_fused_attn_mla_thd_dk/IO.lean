import bench.pretrain_kernels.tasks.te_fused_attn_mla_thd_dk.Kernels
import VeriTile.Triton.Memory.LogicalScheduledKernelIO
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Float.GuardedProgramIO
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Tactic.FinCases

namespace VeriTile.Bench.PretrainKernels.te_fused_attn_mla_thd_dk
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 2000000
namespace IO

structure Inputs where
  input0 : Fin 128 → Fin 4 → Fin 192 → ℝ
  input1 : Fin 128 → Fin 4 → Fin 192 → ℝ
  input2 : Fin 128 → Fin 4 → Fin 128 → ℝ
  input3 : Fin 128 → Fin 4 → Fin 128 → ℝ

abbrev OutputIndex := Fin 128 × Fin 4 × Fin 192

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

def input0Equiv : (Fin (128*4*192) → ℝ) ≃ (Fin 128 → Fin 4 → Fin 192 → ℝ) :=
  ((matrixEquiv 128 (4*192) ℝ).trans (Equiv.piCongrRight fun _ => ((matrixEquiv 4 (192) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 192 → ℝ))))))

def input1Equiv : (Fin (128*4*192) → ℝ) ≃ (Fin 128 → Fin 4 → Fin 192 → ℝ) :=
  ((matrixEquiv 128 (4*192) ℝ).trans (Equiv.piCongrRight fun _ => ((matrixEquiv 4 (192) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 192 → ℝ))))))

def input2Equiv : (Fin (128*4*128) → ℝ) ≃ (Fin 128 → Fin 4 → Fin 128 → ℝ) :=
  ((matrixEquiv 128 (4*128) ℝ).trans (Equiv.piCongrRight fun _ => ((matrixEquiv 4 (128) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 128 → ℝ))))))

def input3Equiv : (Fin (128*4*128) → ℝ) ≃ (Fin 128 → Fin 4 → Fin 128 → ℝ) :=
  ((matrixEquiv 128 (4*128) ℝ).trans (Equiv.piCongrRight fun _ => ((matrixEquiv 4 (128) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 128 → ℝ))))))

def core : UKernelIO where
  kernel := Kernels.original.eraseDType
  projection := by
    have h : Kernels.original.toAlgorithm? = .ok Kernels.original.toAlgKernel := rfl
    simp only [ComputeKernel.eraseDType, h]
    simp [ComputeKernel.toAlgKernel]
  nIn := 4
  nOut := 1
  nScr := 48
  bufs := ["input0", "input1", "input2", "input3", "out", "vfill", "vslice_scatter", "vfill_1", "vslice_scatter_1", "vfill_2", "vslice_scatter_2", "vfill_3", "vslice_scatter_3", "veq", "vge", "vbitwise_and", "vbitwise_not", "vmasked_fill", "vbmm", "vmul", "vadd", "vto_copy", "vsoftmax", "vbmm_2", "vto_copy_2", "vsoftmax_backward_data", "vto_copy_3", "vmul_1", "vbmm_3", "candidate_vfill", "candidate_vslice_scatter", "candidate_vfill_1", "candidate_vslice_scatter_1", "candidate_vfill_2", "candidate_vslice_scatter_2", "candidate_vfill_3", "candidate_vslice_scatter_3", "candidate_veq", "candidate_vge", "candidate_vbitwise_and", "candidate_vbitwise_not", "candidate_vmasked_fill", "candidate_vbmm", "candidate_vmul", "candidate_vadd", "candidate_vto_copy", "candidate_vsoftmax", "candidate_vbmm_2", "candidate_vto_copy_2", "candidate_vsoftmax_backward_data", "candidate_vto_copy_3", "candidate_vmul_1", "candidate_vbmm_3"]
  ity := Fin.cases (.float) (Fin.cases (.float) (Fin.cases (.float) (Fin.cases (.float) (Fin.elim0))))
  iarity := Fin.cases (128*4*192) (Fin.cases (128*4*192) (Fin.cases (128*4*128) (Fin.cases (128*4*128) (Fin.elim0))))
  ibuf := Fin.cases ("input0") (Fin.cases ("input1") (Fin.cases ("input2") (Fin.cases ("input3") (Fin.elim0))))
  oty := fun _ => .float
  oarity := fun _ => 128*4*192
  obuf := fun _ => "out"
  obuf_mem := fun _ => by simp
  sarity := Fin.cases (37) (Fin.cases (128) (Fin.cases (27) (Fin.cases (128) (Fin.cases (27) (Fin.cases (128) (Fin.cases (37) (Fin.cases (128) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (4*128*128) (Fin.cases (1*4*128*128) (Fin.cases (1*4*128*128) (Fin.cases (1*4*128*128) (Fin.cases (1*4*128*128) (Fin.cases (4*128*128) (Fin.cases (1*4*128*128) (Fin.cases (1*4*128*128) (Fin.cases (1*4*128*128) (Fin.cases (1*4*128*128) (Fin.cases (4*192*128) (Fin.cases (37) (Fin.cases (128) (Fin.cases (27) (Fin.cases (128) (Fin.cases (27) (Fin.cases (128) (Fin.cases (37) (Fin.cases (128) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (128*128) (Fin.cases (4*128*128) (Fin.cases (1*4*128*128) (Fin.cases (1*4*128*128) (Fin.cases (1*4*128*128) (Fin.cases (1*4*128*128) (Fin.cases (4*128*128) (Fin.cases (1*4*128*128) (Fin.cases (1*4*128*128) (Fin.cases (1*4*128*128) (Fin.cases (1*4*128*128) (Fin.cases (4*192*128) (Fin.elim0))))))))))))))))))))))))))))))))))))))))))))))))
  sbuf := Fin.cases ("vfill") (Fin.cases ("vslice_scatter") (Fin.cases ("vfill_1") (Fin.cases ("vslice_scatter_1") (Fin.cases ("vfill_2") (Fin.cases ("vslice_scatter_2") (Fin.cases ("vfill_3") (Fin.cases ("vslice_scatter_3") (Fin.cases ("veq") (Fin.cases ("vge") (Fin.cases ("vbitwise_and") (Fin.cases ("vbitwise_not") (Fin.cases ("vmasked_fill") (Fin.cases ("vbmm") (Fin.cases ("vmul") (Fin.cases ("vadd") (Fin.cases ("vto_copy") (Fin.cases ("vsoftmax") (Fin.cases ("vbmm_2") (Fin.cases ("vto_copy_2") (Fin.cases ("vsoftmax_backward_data") (Fin.cases ("vto_copy_3") (Fin.cases ("vmul_1") (Fin.cases ("vbmm_3") (Fin.cases ("candidate_vfill") (Fin.cases ("candidate_vslice_scatter") (Fin.cases ("candidate_vfill_1") (Fin.cases ("candidate_vslice_scatter_1") (Fin.cases ("candidate_vfill_2") (Fin.cases ("candidate_vslice_scatter_2") (Fin.cases ("candidate_vfill_3") (Fin.cases ("candidate_vslice_scatter_3") (Fin.cases ("candidate_veq") (Fin.cases ("candidate_vge") (Fin.cases ("candidate_vbitwise_and") (Fin.cases ("candidate_vbitwise_not") (Fin.cases ("candidate_vmasked_fill") (Fin.cases ("candidate_vbmm") (Fin.cases ("candidate_vmul") (Fin.cases ("candidate_vadd") (Fin.cases ("candidate_vto_copy") (Fin.cases ("candidate_vsoftmax") (Fin.cases ("candidate_vbmm_2") (Fin.cases ("candidate_vto_copy_2") (Fin.cases ("candidate_vsoftmax_backward_data") (Fin.cases ("candidate_vto_copy_3") (Fin.cases ("candidate_vmul_1") (Fin.cases ("candidate_vbmm_3") (Fin.elim0))))))))))))))))))))))))))))))))))))))))))))))))
  iwin := fun _ _ _ _ _ j => j.val
  imask := fun _ _ _ _ _ _ => True
  owin := fun _ _ _ _ _ j => j.val
  omask := fun _ _ _ _ _ _ => True
  swin := fun _ _ _ _ _ j => j.val
  smask := fun _ _ _ _ _ _ => True

def inputs : core.Ctx ≃ Inputs where
  toFun := fun v => ⟨input0Equiv (v (0 : Fin 4)), input1Equiv (v (1 : Fin 4)), input2Equiv (v (2 : Fin 4)), input3Equiv (v (3 : Fin 4))⟩
  invFun := fun v => Fin.cases (input0Equiv.symm v.input0) (Fin.cases (input1Equiv.symm v.input1) (Fin.cases (input2Equiv.symm v.input2) (Fin.cases (input3Equiv.symm v.input3) ((fun i => Fin.elim0 i)))))
  left_inv := by
    intro v
    funext i
    fin_cases i <;> exact Equiv.symm_apply_apply _ _
  right_inv := by
    intro v
    change (⟨input0Equiv (input0Equiv.symm v.input0), input1Equiv (input1Equiv.symm v.input1), input2Equiv (input2Equiv.symm v.input2), input3Equiv (input3Equiv.symm v.input3)⟩ : Inputs) = v
    simp only [Equiv.apply_symm_apply]

def outputCoords : Fin (128*4*192) ≃ OutputIndex :=
  (finProdFinEquiv (m := 128) (n := 4*192)).symm.trans (Equiv.prodCongr (Equiv.refl _) ((finProdFinEquiv (m := 4) (n := 192)).symm))

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
  Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.elim0))))

def scratchStorage : Fin core.nScr → TileDType :=
  Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.bool) (Fin.cases (.bool) (Fin.cases (.bool) (Fin.cases (.bool) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.bool) (Fin.cases (.bool) (Fin.cases (.bool) (Fin.cases (.bool) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.elim0))))))))))))))))))))))))))))))))))))))))))))))))

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
end VeriTile.Bench.PretrainKernels.te_fused_attn_mla_thd_dk
