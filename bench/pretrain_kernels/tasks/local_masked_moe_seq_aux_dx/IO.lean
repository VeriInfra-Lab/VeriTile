import bench.pretrain_kernels.tasks.local_masked_moe_seq_aux_dx.Kernels
import VeriTile.Triton.Memory.LogicalScheduledKernelIO
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Float.GuardedProgramIO
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Tactic.FinCases

namespace VeriTile.Bench.PretrainKernels.local_masked_moe_seq_aux_dx
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 2000000
namespace IO

structure Inputs where
  constantf1ebabb642 : Fin 4 → Int
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

def input0Equiv : (Fin (32*16) → ℝ) ≃ (Fin 32 → Fin 16 → ℝ) :=
  ((matrixEquiv 32 (16) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 16 → ℝ))))

def core : UKernelIO where
  kernel := Kernels.realOriginal.eraseDType
  projection := by
    have h : Kernels.realOriginal.toAlgorithm? = .ok Kernels.realOriginal.toAlgKernel := rfl
    simp only [ComputeKernel.eraseDType, h]
    simp [ComputeKernel.toAlgKernel]
  nIn := 2
  nOut := 1
  nScr := 88
  bufs := ["constantf1ebabb642", "input0", "out", "literal_nan", "vge", "vto_copy", "vsigmoid", "vsum_1", "vadd", "vdiv", "visnan", "vwhere", "vsort_value", "vsort_index", "vto_copy_1", "vscatter", "vto_copy_2", "vbitwise_not", "vbitwise_and", "vmul", "vsum_2", "vsum_3", "vto_copy_3", "vdiv_1", "vclamp_min", "vreciprocal", "vmul_1", "vpow_1", "vto_copy_4", "vmul_2", "vsum_4", "vmul_3", "vsum_5", "vmul_4", "vdiv_2", "vdiv_3", "vmul_5", "vmul_6", "vmul_7", "vdiv_4", "vdiv_5", "vneg", "vmul_8", "vdiv_6", "vsum_6", "vadd_1", "vsigmoid_backward", "vto_copy_5", "candidate_literal_nan", "candidate_vge", "candidate_vto_copy", "candidate_vsigmoid", "candidate_vsum_1", "candidate_vadd", "candidate_vdiv", "candidate_visnan", "candidate_vwhere", "candidate_vsort_value", "candidate_vsort_index", "candidate_vto_copy_1", "candidate_vscatter", "candidate_vto_copy_2", "candidate_vbitwise_not", "candidate_vmul", "candidate_vmul_1", "candidate_vsum_2", "candidate_vsum_3", "candidate_vsum_4", "candidate_vfloor_divide", "candidate_vclamp_min", "candidate_vmul_2", "candidate_vsum_5", "candidate_vpow_1", "candidate_vmul_3", "candidate_vdiv_1", "candidate_vmean", "candidate_vmul_4", "candidate_vmul_5", "candidate_vdiv_2", "candidate_vdiv_3", "candidate_vmul_6", "candidate_vmul_7", "candidate_vdiv_4", "candidate_vdiv_5", "candidate_vneg", "candidate_vmul_8", "candidate_vdiv_6", "candidate_vsum_6", "candidate_vadd_1", "candidate_vsigmoid_backward", "candidate_vto_copy_3"]
  ity := Fin.cases (.int) (Fin.cases (.float) (Fin.elim0))
  iarity := Fin.cases (4) (Fin.cases (32*16) (Fin.elim0))
  ibuf := Fin.cases ("constantf1ebabb642") (Fin.cases ("input0") (Fin.elim0))
  oty := fun _ => .float
  oarity := fun _ => 32*16
  obuf := fun _ => "out"
  obuf_mem := fun _ => by simp
  sarity := Fin.cases (1) (Fin.cases (4*8) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*1) (Fin.cases (32*1) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (4*16) (Fin.cases (4) (Fin.cases (4) (Fin.cases (4) (Fin.cases (4) (Fin.cases (4) (Fin.cases (4) (Fin.cases (4) (Fin.cases (4) (Fin.cases (4*16) (Fin.cases (64) (Fin.cases (64) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (64) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*1) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (1) (Fin.cases (4*8) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*1) (Fin.cases (32*1) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (4*16) (Fin.cases (4*16) (Fin.cases (4) (Fin.cases (4) (Fin.cases (4) (Fin.cases (4*16) (Fin.cases (4) (Fin.cases (4) (Fin.cases (4) (Fin.cases (4) (Fin.cases (1) (Fin.cases (1) (Fin.cases (1) (Fin.cases (4) (Fin.cases (4) (Fin.cases (4*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*1) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.cases (32*16) (Fin.elim0))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
  sbuf := Fin.cases ("literal_nan") (Fin.cases ("vge") (Fin.cases ("vto_copy") (Fin.cases ("vsigmoid") (Fin.cases ("vsum_1") (Fin.cases ("vadd") (Fin.cases ("vdiv") (Fin.cases ("visnan") (Fin.cases ("vwhere") (Fin.cases ("vsort_value") (Fin.cases ("vsort_index") (Fin.cases ("vto_copy_1") (Fin.cases ("vscatter") (Fin.cases ("vto_copy_2") (Fin.cases ("vbitwise_not") (Fin.cases ("vbitwise_and") (Fin.cases ("vmul") (Fin.cases ("vsum_2") (Fin.cases ("vsum_3") (Fin.cases ("vto_copy_3") (Fin.cases ("vdiv_1") (Fin.cases ("vclamp_min") (Fin.cases ("vreciprocal") (Fin.cases ("vmul_1") (Fin.cases ("vpow_1") (Fin.cases ("vto_copy_4") (Fin.cases ("vmul_2") (Fin.cases ("vsum_4") (Fin.cases ("vmul_3") (Fin.cases ("vsum_5") (Fin.cases ("vmul_4") (Fin.cases ("vdiv_2") (Fin.cases ("vdiv_3") (Fin.cases ("vmul_5") (Fin.cases ("vmul_6") (Fin.cases ("vmul_7") (Fin.cases ("vdiv_4") (Fin.cases ("vdiv_5") (Fin.cases ("vneg") (Fin.cases ("vmul_8") (Fin.cases ("vdiv_6") (Fin.cases ("vsum_6") (Fin.cases ("vadd_1") (Fin.cases ("vsigmoid_backward") (Fin.cases ("vto_copy_5") (Fin.cases ("candidate_literal_nan") (Fin.cases ("candidate_vge") (Fin.cases ("candidate_vto_copy") (Fin.cases ("candidate_vsigmoid") (Fin.cases ("candidate_vsum_1") (Fin.cases ("candidate_vadd") (Fin.cases ("candidate_vdiv") (Fin.cases ("candidate_visnan") (Fin.cases ("candidate_vwhere") (Fin.cases ("candidate_vsort_value") (Fin.cases ("candidate_vsort_index") (Fin.cases ("candidate_vto_copy_1") (Fin.cases ("candidate_vscatter") (Fin.cases ("candidate_vto_copy_2") (Fin.cases ("candidate_vbitwise_not") (Fin.cases ("candidate_vmul") (Fin.cases ("candidate_vmul_1") (Fin.cases ("candidate_vsum_2") (Fin.cases ("candidate_vsum_3") (Fin.cases ("candidate_vsum_4") (Fin.cases ("candidate_vfloor_divide") (Fin.cases ("candidate_vclamp_min") (Fin.cases ("candidate_vmul_2") (Fin.cases ("candidate_vsum_5") (Fin.cases ("candidate_vpow_1") (Fin.cases ("candidate_vmul_3") (Fin.cases ("candidate_vdiv_1") (Fin.cases ("candidate_vmean") (Fin.cases ("candidate_vmul_4") (Fin.cases ("candidate_vmul_5") (Fin.cases ("candidate_vdiv_2") (Fin.cases ("candidate_vdiv_3") (Fin.cases ("candidate_vmul_6") (Fin.cases ("candidate_vmul_7") (Fin.cases ("candidate_vdiv_4") (Fin.cases ("candidate_vdiv_5") (Fin.cases ("candidate_vneg") (Fin.cases ("candidate_vmul_8") (Fin.cases ("candidate_vdiv_6") (Fin.cases ("candidate_vsum_6") (Fin.cases ("candidate_vadd_1") (Fin.cases ("candidate_vsigmoid_backward") (Fin.cases ("candidate_vto_copy_3") (Fin.elim0))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
  iwin := fun _ _ _ _ _ j => j.val
  imask := fun _ _ _ _ _ _ => True
  owin := fun _ _ _ _ _ j => j.val
  omask := fun _ _ _ _ _ _ => True
  swin := fun _ _ _ _ _ j => j.val
  smask := fun _ _ _ _ _ _ => True

def inputs : core.Ctx ≃ Inputs where
  toFun := fun v => ⟨constantf1ebabb642Equiv (v (0 : Fin 2)), input0Equiv (v (1 : Fin 2))⟩
  invFun := fun v => Fin.cases (constantf1ebabb642Equiv.symm v.constantf1ebabb642) (Fin.cases (input0Equiv.symm v.input0) ((fun i => Fin.elim0 i)))
  left_inv := by
    intro v
    funext i
    fin_cases i <;> exact Equiv.symm_apply_apply _ _
  right_inv := by
    intro v
    change (⟨constantf1ebabb642Equiv (constantf1ebabb642Equiv.symm v.constantf1ebabb642), input0Equiv (input0Equiv.symm v.input0)⟩ : Inputs) = v
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
  Fin.cases (.int) (Fin.cases (.bf16) (Fin.elim0))

def scratchStorage : Fin core.nScr → TileDType :=
  Fin.cases (.real) (Fin.cases (.bool) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bool) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.bool) (Fin.cases (.bool) (Fin.cases (.bool) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.bool) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bool) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.bool) (Fin.cases (.bool) (Fin.cases (.bool) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.elim0))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

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
  (∀ j : Fin 4, vals (0 : Fin 2) (j, PUnit.unit) = ([1, 3, 0, 6] : List Int).getD j.val 0)

end FP
end IO
end
end VeriTile.Bench.PretrainKernels.local_masked_moe_seq_aux_dx
