import bench.pretrain_kernels.tasks.local_fused_hc_pre_dx.Kernels
import VeriTile.Triton.Memory.LogicalScheduledKernelIO
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Float.GuardedProgramIO
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Tactic.FinCases

namespace VeriTile.Bench.PretrainKernels.local_fused_hc_pre_dx
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 2000000
namespace IO

structure Inputs where
  input0 : Fin 2 → Fin 4096 → Fin 4 → Fin 2048 → ℝ
  input1 : Fin 8 → Fin 8192 → ℝ
  input2 : Fin 2 → ℝ
  input3 : Fin 8 → ℝ
  input4 : Fin 2 → Fin 4096 → Fin 2048 → ℝ
  input5 : Fin 2 → Fin 4096 → Fin 4 → ℝ

abbrev OutputIndex := Fin 2 × Fin 4096 × Fin 4 × Fin 2048

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

def input0Equiv : (Fin (2*4096*4*2048) → ℝ) ≃ (Fin 2 → Fin 4096 → Fin 4 → Fin 2048 → ℝ) :=
  ((matrixEquiv 2 (4096*4*2048) ℝ).trans (Equiv.piCongrRight fun _ => ((matrixEquiv 4096 (4*2048) ℝ).trans (Equiv.piCongrRight fun _ => ((matrixEquiv 4 (2048) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 2048 → ℝ))))))))

def input1Equiv : (Fin (8*8192) → ℝ) ≃ (Fin 8 → Fin 8192 → ℝ) :=
  ((matrixEquiv 8 (8192) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 8192 → ℝ))))

def input2Equiv : (Fin (2) → ℝ) ≃ (Fin 2 → ℝ) :=
  (Equiv.refl (Fin 2 → ℝ))

def input3Equiv : (Fin (8) → ℝ) ≃ (Fin 8 → ℝ) :=
  (Equiv.refl (Fin 8 → ℝ))

def input4Equiv : (Fin (2*4096*2048) → ℝ) ≃ (Fin 2 → Fin 4096 → Fin 2048 → ℝ) :=
  ((matrixEquiv 2 (4096*2048) ℝ).trans (Equiv.piCongrRight fun _ => ((matrixEquiv 4096 (2048) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 2048 → ℝ))))))

def input5Equiv : (Fin (2*4096*4) → ℝ) ≃ (Fin 2 → Fin 4096 → Fin 4 → ℝ) :=
  ((matrixEquiv 2 (4096*4) ℝ).trans (Equiv.piCongrRight fun _ => ((matrixEquiv 4096 (4) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 4 → ℝ))))))

def core : UKernelIO where
  kernel := Kernels.original.eraseDType
  projection := by
    have h : Kernels.original.toAlgorithm? = .ok Kernels.original.toAlgKernel := rfl
    simp only [ComputeKernel.eraseDType, h]
    simp [ComputeKernel.toAlgKernel]
  nIn := 6
  nOut := 1
  nScr := 80
  bufs := ["input0", "input1", "input2", "input3", "input4", "input5", "out", "vto_copy", "vpow_1", "vmean", "vadd", "vrsqrt", "vmm", "vmul", "vmul_1", "vadd_1", "vsigmoid", "vadd_2", "vmul_2", "vadd_3", "vsigmoid_1", "vto_copy_2", "vmul_5", "vmul_6", "vsum_2", "vmul_7", "vsigmoid_backward", "vmul_8", "vslice_backward", "vsigmoid_backward_1", "vmul_9", "vslice_backward_1", "vadd_5", "vmul_10", "vmul_11", "vsum_3", "vmm_1", "vadd_6", "vpow_2", "vmul_12", "vmul_13", "vdiv", "vpow_3", "vmul_14", "vmul_15", "vadd_7", "vto_copy_3", "candidate_vto_copy", "candidate_vpow_1", "candidate_vmean", "candidate_vadd", "candidate_vrsqrt", "candidate_vmm", "candidate_vmul", "candidate_vmul_1", "candidate_vadd_1", "candidate_vsigmoid", "candidate_vadd_2", "candidate_vmul_2", "candidate_vadd_3", "candidate_vsigmoid_1", "candidate_vto_copy_2", "candidate_vmul_5", "candidate_vmul_6", "candidate_vsum_2", "candidate_vmul_7", "candidate_vsigmoid_backward", "candidate_vmul_8", "candidate_vslice_backward", "candidate_vsigmoid_backward_1", "candidate_vmul_9", "candidate_vslice_backward_1", "candidate_vadd_5", "candidate_vmul_10", "candidate_vmul_11", "candidate_vsum_3", "candidate_vmm_1", "candidate_vadd_6", "candidate_vpow_2", "candidate_vmul_12", "candidate_vmul_13", "candidate_vdiv", "candidate_vpow_3", "candidate_vmul_14", "candidate_vmul_15", "candidate_vadd_7", "candidate_vto_copy_3"]
  ity := Fin.cases (.float) (Fin.cases (.float) (Fin.cases (.float) (Fin.cases (.float) (Fin.cases (.float) (Fin.cases (.float) (Fin.elim0))))))
  iarity := Fin.cases (2*4096*4*2048) (Fin.cases (8*8192) (Fin.cases (2) (Fin.cases (8) (Fin.cases (2*4096*2048) (Fin.cases (2*4096*4) (Fin.elim0))))))
  ibuf := Fin.cases ("input0") (Fin.cases ("input1") (Fin.cases ("input2") (Fin.cases ("input3") (Fin.cases ("input4") (Fin.cases ("input5") (Fin.elim0))))))
  oty := fun _ => .float
  oarity := fun _ => 2*4096*4*2048
  obuf := fun _ => "out"
  obuf_mem := fun _ => by simp
  sarity := Fin.cases (2*4096*8192) (Fin.cases (2*4096*8192) (Fin.cases (2*4096*1) (Fin.cases (2*4096*1) (Fin.cases (2*4096*1) (Fin.cases (8192*8) (Fin.cases (2*4096*8) (Fin.cases (2*4096*4) (Fin.cases (2*4096*4) (Fin.cases (2*4096*4) (Fin.cases (2*4096*4) (Fin.cases (2*4096*4) (Fin.cases (2*4096*4) (Fin.cases (2*4096*4) (Fin.cases (2*4096*2048) (Fin.cases (2*4096*4*2048) (Fin.cases (2*4096*4*2048) (Fin.cases (2*4096*4*1) (Fin.cases (2*4096*4) (Fin.cases (2*4096*4) (Fin.cases (2*4096*4) (Fin.cases (2*4096*8) (Fin.cases (2*4096*4) (Fin.cases (2*4096*4) (Fin.cases (2*4096*8) (Fin.cases (2*4096*8) (Fin.cases (2*4096*8) (Fin.cases (2*4096*8) (Fin.cases (2*4096*1) (Fin.cases (8192*8192) (Fin.cases (2*4096*8192) (Fin.cases (2*4096*1) (Fin.cases (2*4096*1) (Fin.cases (2*4096*1) (Fin.cases (2*4096*8192) (Fin.cases (2*4096*8192) (Fin.cases (2*4096*8192) (Fin.cases (2*4096*8192) (Fin.cases (2*4096*8192) (Fin.cases (2*4096*8192) (Fin.cases (2*4096*8192) (Fin.cases (2*4096*8192) (Fin.cases (2*4096*1) (Fin.cases (2*4096*1) (Fin.cases (2*4096*1) (Fin.cases (8192*8) (Fin.cases (2*4096*8) (Fin.cases (2*4096*4) (Fin.cases (2*4096*4) (Fin.cases (2*4096*4) (Fin.cases (2*4096*4) (Fin.cases (2*4096*4) (Fin.cases (2*4096*4) (Fin.cases (2*4096*4) (Fin.cases (2*4096*2048) (Fin.cases (2*4096*4*2048) (Fin.cases (2*4096*4*2048) (Fin.cases (2*4096*4*1) (Fin.cases (2*4096*4) (Fin.cases (2*4096*4) (Fin.cases (2*4096*4) (Fin.cases (2*4096*8) (Fin.cases (2*4096*4) (Fin.cases (2*4096*4) (Fin.cases (2*4096*8) (Fin.cases (2*4096*8) (Fin.cases (2*4096*8) (Fin.cases (2*4096*8) (Fin.cases (2*4096*1) (Fin.cases (8192*8192) (Fin.cases (2*4096*8192) (Fin.cases (2*4096*1) (Fin.cases (2*4096*1) (Fin.cases (2*4096*1) (Fin.cases (2*4096*8192) (Fin.cases (2*4096*8192) (Fin.cases (2*4096*8192) (Fin.cases (2*4096*8192) (Fin.cases (2*4096*8192) (Fin.cases (2*4096*8192) (Fin.elim0))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
  sbuf := Fin.cases ("vto_copy") (Fin.cases ("vpow_1") (Fin.cases ("vmean") (Fin.cases ("vadd") (Fin.cases ("vrsqrt") (Fin.cases ("vmm") (Fin.cases ("vmul") (Fin.cases ("vmul_1") (Fin.cases ("vadd_1") (Fin.cases ("vsigmoid") (Fin.cases ("vadd_2") (Fin.cases ("vmul_2") (Fin.cases ("vadd_3") (Fin.cases ("vsigmoid_1") (Fin.cases ("vto_copy_2") (Fin.cases ("vmul_5") (Fin.cases ("vmul_6") (Fin.cases ("vsum_2") (Fin.cases ("vmul_7") (Fin.cases ("vsigmoid_backward") (Fin.cases ("vmul_8") (Fin.cases ("vslice_backward") (Fin.cases ("vsigmoid_backward_1") (Fin.cases ("vmul_9") (Fin.cases ("vslice_backward_1") (Fin.cases ("vadd_5") (Fin.cases ("vmul_10") (Fin.cases ("vmul_11") (Fin.cases ("vsum_3") (Fin.cases ("vmm_1") (Fin.cases ("vadd_6") (Fin.cases ("vpow_2") (Fin.cases ("vmul_12") (Fin.cases ("vmul_13") (Fin.cases ("vdiv") (Fin.cases ("vpow_3") (Fin.cases ("vmul_14") (Fin.cases ("vmul_15") (Fin.cases ("vadd_7") (Fin.cases ("vto_copy_3") (Fin.cases ("candidate_vto_copy") (Fin.cases ("candidate_vpow_1") (Fin.cases ("candidate_vmean") (Fin.cases ("candidate_vadd") (Fin.cases ("candidate_vrsqrt") (Fin.cases ("candidate_vmm") (Fin.cases ("candidate_vmul") (Fin.cases ("candidate_vmul_1") (Fin.cases ("candidate_vadd_1") (Fin.cases ("candidate_vsigmoid") (Fin.cases ("candidate_vadd_2") (Fin.cases ("candidate_vmul_2") (Fin.cases ("candidate_vadd_3") (Fin.cases ("candidate_vsigmoid_1") (Fin.cases ("candidate_vto_copy_2") (Fin.cases ("candidate_vmul_5") (Fin.cases ("candidate_vmul_6") (Fin.cases ("candidate_vsum_2") (Fin.cases ("candidate_vmul_7") (Fin.cases ("candidate_vsigmoid_backward") (Fin.cases ("candidate_vmul_8") (Fin.cases ("candidate_vslice_backward") (Fin.cases ("candidate_vsigmoid_backward_1") (Fin.cases ("candidate_vmul_9") (Fin.cases ("candidate_vslice_backward_1") (Fin.cases ("candidate_vadd_5") (Fin.cases ("candidate_vmul_10") (Fin.cases ("candidate_vmul_11") (Fin.cases ("candidate_vsum_3") (Fin.cases ("candidate_vmm_1") (Fin.cases ("candidate_vadd_6") (Fin.cases ("candidate_vpow_2") (Fin.cases ("candidate_vmul_12") (Fin.cases ("candidate_vmul_13") (Fin.cases ("candidate_vdiv") (Fin.cases ("candidate_vpow_3") (Fin.cases ("candidate_vmul_14") (Fin.cases ("candidate_vmul_15") (Fin.cases ("candidate_vadd_7") (Fin.cases ("candidate_vto_copy_3") (Fin.elim0))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
  iwin := fun _ _ _ _ _ j => j.val
  imask := fun _ _ _ _ _ _ => True
  owin := fun _ _ _ _ _ j => j.val
  omask := fun _ _ _ _ _ _ => True
  swin := fun _ _ _ _ _ j => j.val
  smask := fun _ _ _ _ _ _ => True

def inputs : core.Ctx ≃ Inputs where
  toFun := fun v => ⟨input0Equiv (v (0 : Fin 6)), input1Equiv (v (1 : Fin 6)), input2Equiv (v (2 : Fin 6)), input3Equiv (v (3 : Fin 6)), input4Equiv (v (4 : Fin 6)), input5Equiv (v (5 : Fin 6))⟩
  invFun := fun v => Fin.cases (input0Equiv.symm v.input0) (Fin.cases (input1Equiv.symm v.input1) (Fin.cases (input2Equiv.symm v.input2) (Fin.cases (input3Equiv.symm v.input3) (Fin.cases (input4Equiv.symm v.input4) (Fin.cases (input5Equiv.symm v.input5) ((fun i => Fin.elim0 i)))))))
  left_inv := by
    intro v
    funext i
    fin_cases i <;> exact Equiv.symm_apply_apply _ _
  right_inv := by
    intro v
    change (⟨input0Equiv (input0Equiv.symm v.input0), input1Equiv (input1Equiv.symm v.input1), input2Equiv (input2Equiv.symm v.input2), input3Equiv (input3Equiv.symm v.input3), input4Equiv (input4Equiv.symm v.input4), input5Equiv (input5Equiv.symm v.input5)⟩ : Inputs) = v
    simp only [Equiv.apply_symm_apply]

def outputCoords : Fin (2*4096*4*2048) ≃ OutputIndex :=
  (finProdFinEquiv (m := 2) (n := 4096*4*2048)).symm.trans (Equiv.prodCongr (Equiv.refl _) ((finProdFinEquiv (m := 4096) (n := 4*2048)).symm.trans (Equiv.prodCongr (Equiv.refl _) ((finProdFinEquiv (m := 4) (n := 2048)).symm))))

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
  Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.elim0))))))

def scratchStorage : Fin core.nScr → TileDType :=
  Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.elim0))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

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
end VeriTile.Bench.PretrainKernels.local_fused_hc_pre_dx
