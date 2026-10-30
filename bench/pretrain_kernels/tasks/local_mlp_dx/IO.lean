import bench.pretrain_kernels.tasks.local_mlp_dx.Kernels
import VeriTile.Triton.Memory.LogicalScheduledKernelIO
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Float.GuardedProgramIO
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Tactic.FinCases

namespace VeriTile.Bench.PretrainKernels.local_mlp_dx
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 2000000
namespace IO

structure Inputs where
  parameter0 : Fin 6144 → Fin 2048 → ℝ
  parameter1 : Fin 6144 → Fin 2048 → ℝ
  parameter2 : Fin 2048 → Fin 6144 → ℝ
  input0 : Fin 2 → Fin 128 → Fin 2048 → ℝ
  input1 : Fin 2 → Fin 128 → Fin 2048 → ℝ

abbrev OutputIndex := Fin 2 × Fin 128 × Fin 2048

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

def parameter0Equiv : (Fin (6144*2048) → ℝ) ≃ (Fin 6144 → Fin 2048 → ℝ) :=
  ((matrixEquiv 6144 (2048) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 2048 → ℝ))))

def parameter1Equiv : (Fin (6144*2048) → ℝ) ≃ (Fin 6144 → Fin 2048 → ℝ) :=
  ((matrixEquiv 6144 (2048) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 2048 → ℝ))))

def parameter2Equiv : (Fin (2048*6144) → ℝ) ≃ (Fin 2048 → Fin 6144 → ℝ) :=
  ((matrixEquiv 2048 (6144) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 6144 → ℝ))))

def input0Equiv : (Fin (2*128*2048) → ℝ) ≃ (Fin 2 → Fin 128 → Fin 2048 → ℝ) :=
  ((matrixEquiv 2 (128*2048) ℝ).trans (Equiv.piCongrRight fun _ => ((matrixEquiv 128 (2048) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 2048 → ℝ))))))

def input1Equiv : (Fin (2*128*2048) → ℝ) ≃ (Fin 2 → Fin 128 → Fin 2048 → ℝ) :=
  ((matrixEquiv 2 (128*2048) ℝ).trans (Equiv.piCongrRight fun _ => ((matrixEquiv 128 (2048) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 2048 → ℝ))))))

def core : UKernelIO where
  kernel := Kernels.original.eraseDType
  projection := by
    have h : Kernels.original.toAlgorithm? = .ok Kernels.original.toAlgKernel := rfl
    simp only [ComputeKernel.eraseDType, h]
    simp [ComputeKernel.toAlgKernel]
  nIn := 5
  nOut := 1
  nScr := 35
  bufs := ["parameter0", "parameter1", "parameter2", "input0", "input1", "out", "vmul", "vmul_1", "vmul_2", "vmm", "vmm_1", "vto_copy", "vsilu", "vto_copy_1", "vmm_3", "vto_copy_3", "vmul_4", "vmul_5", "vto_copy_4", "vsilu_backward", "vto_copy_5", "vmm_4", "vmm_5", "vadd", "candidate_vmul", "candidate_vmul_1", "candidate_vmul_2", "candidate_vcat", "candidate_vmm", "candidate_vto_copy", "candidate_vsilu", "candidate_vto_copy_1", "candidate_vmm_2", "candidate_vto_copy_3", "candidate_vmul_4", "candidate_vmul_5", "candidate_vto_copy_4", "candidate_vsilu_backward", "candidate_vto_copy_5", "candidate_vcat_1", "candidate_vmm_3"]
  ity := Fin.cases (.float) (Fin.cases (.float) (Fin.cases (.float) (Fin.cases (.float) (Fin.cases (.float) (Fin.elim0)))))
  iarity := Fin.cases (6144*2048) (Fin.cases (6144*2048) (Fin.cases (2048*6144) (Fin.cases (2*128*2048) (Fin.cases (2*128*2048) (Fin.elim0)))))
  ibuf := Fin.cases ("parameter0") (Fin.cases ("parameter1") (Fin.cases ("parameter2") (Fin.cases ("input0") (Fin.cases ("input1") (Fin.elim0)))))
  oty := fun _ => .float
  oarity := fun _ => 2*128*2048
  obuf := fun _ => "out"
  obuf_mem := fun _ => by simp
  sarity := Fin.cases (6144*2048) (Fin.cases (6144*2048) (Fin.cases (2048*6144) (Fin.cases (256*6144) (Fin.cases (256*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (256*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (256*2048) (Fin.cases (256*2048) (Fin.cases (2*128*2048) (Fin.cases (6144*2048) (Fin.cases (6144*2048) (Fin.cases (2048*6144) (Fin.cases (12288*2048) (Fin.cases (256*12288) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (256*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*6144) (Fin.cases (2*128*12288) (Fin.cases (256*2048) (Fin.elim0)))))))))))))))))))))))))))))))))))
  sbuf := Fin.cases ("vmul") (Fin.cases ("vmul_1") (Fin.cases ("vmul_2") (Fin.cases ("vmm") (Fin.cases ("vmm_1") (Fin.cases ("vto_copy") (Fin.cases ("vsilu") (Fin.cases ("vto_copy_1") (Fin.cases ("vmm_3") (Fin.cases ("vto_copy_3") (Fin.cases ("vmul_4") (Fin.cases ("vmul_5") (Fin.cases ("vto_copy_4") (Fin.cases ("vsilu_backward") (Fin.cases ("vto_copy_5") (Fin.cases ("vmm_4") (Fin.cases ("vmm_5") (Fin.cases ("vadd") (Fin.cases ("candidate_vmul") (Fin.cases ("candidate_vmul_1") (Fin.cases ("candidate_vmul_2") (Fin.cases ("candidate_vcat") (Fin.cases ("candidate_vmm") (Fin.cases ("candidate_vto_copy") (Fin.cases ("candidate_vsilu") (Fin.cases ("candidate_vto_copy_1") (Fin.cases ("candidate_vmm_2") (Fin.cases ("candidate_vto_copy_3") (Fin.cases ("candidate_vmul_4") (Fin.cases ("candidate_vmul_5") (Fin.cases ("candidate_vto_copy_4") (Fin.cases ("candidate_vsilu_backward") (Fin.cases ("candidate_vto_copy_5") (Fin.cases ("candidate_vcat_1") (Fin.cases ("candidate_vmm_3") (Fin.elim0)))))))))))))))))))))))))))))))))))
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

def outputCoords : Fin (2*128*2048) ≃ OutputIndex :=
  (finProdFinEquiv (m := 2) (n := 128*2048)).symm.trans (Equiv.prodCongr (Equiv.refl _) ((finProdFinEquiv (m := 128) (n := 2048)).symm))

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
  Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.elim0)))))

def scratchStorage : Fin core.nScr → TileDType :=
  Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.elim0)))))))))))))))))))))))))))))))))))

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
end VeriTile.Bench.PretrainKernels.local_mlp_dx
