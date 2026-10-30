import bench.pretrain_kernels.tasks.local_situ_glu_b30x30_linear_dx.Kernels
import VeriTile.Triton.Memory.LogicalScheduledKernelIO
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Float.GuardedProgramIO
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Tactic.FinCases

namespace VeriTile.Bench.PretrainKernels.local_situ_glu_b30x30_linear_dx
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 2000000
namespace IO

structure Inputs where
  input0 : Fin 2 → Fin 128 → Fin 12288 → ℝ
  input1 : Fin 2 → Fin 128 → Fin 6144 → ℝ

abbrev OutputIndex := Fin 2 × Fin 128 × Fin 12288

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

def input0Equiv : (Fin (2*128*12288) → ℝ) ≃ (Fin 2 → Fin 128 → Fin 12288 → ℝ) :=
  ((matrixEquiv 2 (128*12288) ℝ).trans (Equiv.piCongrRight fun _ => ((matrixEquiv 128 (12288) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 12288 → ℝ))))))

def input1Equiv : (Fin (2*128*6144) → ℝ) ≃ (Fin 2 → Fin 128 → Fin 6144 → ℝ) :=
  ((matrixEquiv 2 (128*6144) ℝ).trans (Equiv.piCongrRight fun _ => ((matrixEquiv 128 (6144) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 6144 → ℝ))))))

def core : UKernelIO where
  kernel := Kernels.original.eraseDType
  projection := by
    have h : Kernels.original.toAlgorithm? = .ok Kernels.original.toAlgKernel := rfl
    simp only [ComputeKernel.eraseDType, h]
    simp [ComputeKernel.toAlgKernel]
  nIn := 2
  nOut := 1
  nScr := 52
  bufs := ["input0", "input1", "out", "vto_copy", "vto_copy_1", "vdiv", "vtanh", "vmul", "vsigmoid", "vmul_1", "vdiv_1", "vtanh_1", "vmul_2", "vto_copy_3", "vmul_4", "vmul_5", "vmul_6", "vtanh_backward", "vdiv_2", "vmul_7", "vmul_8", "vsigmoid_backward", "vmul_9", "vtanh_backward_1", "vdiv_3", "vadd", "vto_copy_4", "vto_copy_5", "vcat", "candidate_vto_copy", "candidate_vto_copy_1", "candidate_vdiv", "candidate_vtanh", "candidate_vmul", "candidate_vsigmoid", "candidate_vmul_1", "candidate_vdiv_1", "candidate_vtanh_1", "candidate_vmul_2", "candidate_vto_copy_3", "candidate_vmul_4", "candidate_vmul_5", "candidate_vmul_6", "candidate_vtanh_backward", "candidate_vdiv_2", "candidate_vmul_7", "candidate_vmul_8", "candidate_vsigmoid_backward", "candidate_vmul_9", "candidate_vtanh_backward_1", "candidate_vdiv_3", "candidate_vadd", "candidate_vto_copy_4", "candidate_vto_copy_5", "candidate_vcat"]
  ity := Fin.cases (.float) (Fin.cases (.float) (Fin.elim0))
  iarity := Fin.cases (2*128*12288) (Fin.cases (2*128*6144) (Fin.elim0))
  ibuf := Fin.cases ("input0") (Fin.cases ("input1") (Fin.elim0))
  oty := fun _ => .float
  oarity := fun _ => 2*128*12288
  obuf := fun _ => "out"
  obuf_mem := fun _ => by simp
  sarity := Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*12288) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*12288) (Fin.elim0))))))))))))))))))))))))))))))))))))))))))))))))))))
  sbuf := Fin.cases ("vto_copy") (Fin.cases ("vto_copy_1") (Fin.cases ("vdiv") (Fin.cases ("vtanh") (Fin.cases ("vmul") (Fin.cases ("vsigmoid") (Fin.cases ("vmul_1") (Fin.cases ("vdiv_1") (Fin.cases ("vtanh_1") (Fin.cases ("vmul_2") (Fin.cases ("vto_copy_3") (Fin.cases ("vmul_4") (Fin.cases ("vmul_5") (Fin.cases ("vmul_6") (Fin.cases ("vtanh_backward") (Fin.cases ("vdiv_2") (Fin.cases ("vmul_7") (Fin.cases ("vmul_8") (Fin.cases ("vsigmoid_backward") (Fin.cases ("vmul_9") (Fin.cases ("vtanh_backward_1") (Fin.cases ("vdiv_3") (Fin.cases ("vadd") (Fin.cases ("vto_copy_4") (Fin.cases ("vto_copy_5") (Fin.cases ("vcat") (Fin.cases ("candidate_vto_copy") (Fin.cases ("candidate_vto_copy_1") (Fin.cases ("candidate_vdiv") (Fin.cases ("candidate_vtanh") (Fin.cases ("candidate_vmul") (Fin.cases ("candidate_vsigmoid") (Fin.cases ("candidate_vmul_1") (Fin.cases ("candidate_vdiv_1") (Fin.cases ("candidate_vtanh_1") (Fin.cases ("candidate_vmul_2") (Fin.cases ("candidate_vto_copy_3") (Fin.cases ("candidate_vmul_4") (Fin.cases ("candidate_vmul_5") (Fin.cases ("candidate_vmul_6") (Fin.cases ("candidate_vtanh_backward") (Fin.cases ("candidate_vdiv_2") (Fin.cases ("candidate_vmul_7") (Fin.cases ("candidate_vmul_8") (Fin.cases ("candidate_vsigmoid_backward") (Fin.cases ("candidate_vmul_9") (Fin.cases ("candidate_vtanh_backward_1") (Fin.cases ("candidate_vdiv_3") (Fin.cases ("candidate_vadd") (Fin.cases ("candidate_vto_copy_4") (Fin.cases ("candidate_vto_copy_5") (Fin.cases ("candidate_vcat") (Fin.elim0))))))))))))))))))))))))))))))))))))))))))))))))))))
  iwin := fun _ _ _ _ _ j => j.val
  imask := fun _ _ _ _ _ _ => True
  owin := fun _ _ _ _ _ j => j.val
  omask := fun _ _ _ _ _ _ => True
  swin := fun _ _ _ _ _ j => j.val
  smask := fun _ _ _ _ _ _ => True

def inputs : core.Ctx ≃ Inputs where
  toFun := fun v => ⟨input0Equiv (v (0 : Fin 2)), input1Equiv (v (1 : Fin 2))⟩
  invFun := fun v => Fin.cases (input0Equiv.symm v.input0) (Fin.cases (input1Equiv.symm v.input1) ((fun i => Fin.elim0 i)))
  left_inv := by
    intro v
    funext i
    fin_cases i <;> exact Equiv.symm_apply_apply _ _
  right_inv := by
    intro v
    change (⟨input0Equiv (input0Equiv.symm v.input0), input1Equiv (input1Equiv.symm v.input1)⟩ : Inputs) = v
    simp only [Equiv.apply_symm_apply]

def outputCoords : Fin (2*128*12288) ≃ OutputIndex :=
  (finProdFinEquiv (m := 2) (n := 128*12288)).symm.trans (Equiv.prodCongr (Equiv.refl _) ((finProdFinEquiv (m := 128) (n := 12288)).symm))

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
  Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.elim0))))))))))))))))))))))))))))))))))))))))))))))))))))

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
end VeriTile.Bench.PretrainKernels.local_situ_glu_b30x30_linear_dx
