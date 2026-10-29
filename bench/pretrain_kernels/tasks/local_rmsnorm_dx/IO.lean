import bench.pretrain_kernels.tasks.local_rmsnorm_dx.Kernels
import VeriTile.Triton.Memory.LogicalScheduledKernelIO
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Float.GuardedProgramIO
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Tactic.FinCases

namespace VeriTile.Bench.PretrainKernels.local_rmsnorm_dx
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 2000000
namespace IO

structure Inputs where
  parameter0 : Fin 2048 → ℝ
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

def parameter0Equiv : (Fin (2048) → ℝ) ≃ (Fin 2048 → ℝ) :=
  (Equiv.refl (Fin 2048 → ℝ))

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
  nIn := 3
  nOut := 1
  nScr := 38
  bufs := ["parameter0", "input0", "input1", "out", "vto_copy", "vpow_1", "vmean", "vadd", "vrsqrt", "vto_copy_2", "vmul_2", "vmul_3", "vmul_4", "vsum_1", "vpow_2", "vmul_5", "vmul_6", "vdiv", "vpow_3", "vmul_7", "vmul_8", "vadd_1", "vto_copy_3", "candidate_vto_copy", "candidate_vpow_1", "candidate_vmean", "candidate_vadd", "candidate_vrsqrt", "candidate_vto_copy_2", "candidate_vmul_2", "candidate_vmul_3", "candidate_vmul_4", "candidate_vsum_1", "candidate_vpow_2", "candidate_vmul_5", "candidate_vmul_6", "candidate_vdiv", "candidate_vpow_3", "candidate_vmul_7", "candidate_vmul_8", "candidate_vadd_1", "candidate_vto_copy_3"]
  ity := Fin.cases (.float) (Fin.cases (.float) (Fin.cases (.float) (Fin.elim0)))
  iarity := Fin.cases (2048) (Fin.cases (2*128*2048) (Fin.cases (2*128*2048) (Fin.elim0)))
  ibuf := Fin.cases ("parameter0") (Fin.cases ("input0") (Fin.cases ("input1") (Fin.elim0)))
  oty := fun _ => .float
  oarity := fun _ => 2*128*2048
  obuf := fun _ => "out"
  obuf_mem := fun _ => by simp
  sarity := Fin.cases (2*128*2048) (Fin.cases (2*128*2048) (Fin.cases (2*128*1) (Fin.cases (2*128*1) (Fin.cases (2*128*1) (Fin.cases (2*128*2048) (Fin.cases (2*128*2048) (Fin.cases (2*128*2048) (Fin.cases (2*128*2048) (Fin.cases (2*128*1) (Fin.cases (2*128*1) (Fin.cases (2*128*1) (Fin.cases (2*128*1) (Fin.cases (2*128*2048) (Fin.cases (2*128*2048) (Fin.cases (2*128*2048) (Fin.cases (2*128*2048) (Fin.cases (2*128*2048) (Fin.cases (2*128*2048) (Fin.cases (2*128*2048) (Fin.cases (2*128*2048) (Fin.cases (2*128*1) (Fin.cases (2*128*1) (Fin.cases (2*128*1) (Fin.cases (2*128*2048) (Fin.cases (2*128*2048) (Fin.cases (2*128*2048) (Fin.cases (2*128*2048) (Fin.cases (2*128*1) (Fin.cases (2*128*1) (Fin.cases (2*128*1) (Fin.cases (2*128*1) (Fin.cases (2*128*2048) (Fin.cases (2*128*2048) (Fin.cases (2*128*2048) (Fin.cases (2*128*2048) (Fin.cases (2*128*2048) (Fin.cases (2*128*2048) (Fin.elim0))))))))))))))))))))))))))))))))))))))
  sbuf := Fin.cases ("vto_copy") (Fin.cases ("vpow_1") (Fin.cases ("vmean") (Fin.cases ("vadd") (Fin.cases ("vrsqrt") (Fin.cases ("vto_copy_2") (Fin.cases ("vmul_2") (Fin.cases ("vmul_3") (Fin.cases ("vmul_4") (Fin.cases ("vsum_1") (Fin.cases ("vpow_2") (Fin.cases ("vmul_5") (Fin.cases ("vmul_6") (Fin.cases ("vdiv") (Fin.cases ("vpow_3") (Fin.cases ("vmul_7") (Fin.cases ("vmul_8") (Fin.cases ("vadd_1") (Fin.cases ("vto_copy_3") (Fin.cases ("candidate_vto_copy") (Fin.cases ("candidate_vpow_1") (Fin.cases ("candidate_vmean") (Fin.cases ("candidate_vadd") (Fin.cases ("candidate_vrsqrt") (Fin.cases ("candidate_vto_copy_2") (Fin.cases ("candidate_vmul_2") (Fin.cases ("candidate_vmul_3") (Fin.cases ("candidate_vmul_4") (Fin.cases ("candidate_vsum_1") (Fin.cases ("candidate_vpow_2") (Fin.cases ("candidate_vmul_5") (Fin.cases ("candidate_vmul_6") (Fin.cases ("candidate_vdiv") (Fin.cases ("candidate_vpow_3") (Fin.cases ("candidate_vmul_7") (Fin.cases ("candidate_vmul_8") (Fin.cases ("candidate_vadd_1") (Fin.cases ("candidate_vto_copy_3") (Fin.elim0))))))))))))))))))))))))))))))))))))))
  iwin := fun _ _ _ _ _ j => j.val
  imask := fun _ _ _ _ _ _ => True
  owin := fun _ _ _ _ _ j => j.val
  omask := fun _ _ _ _ _ _ => True
  swin := fun _ _ _ _ _ j => j.val
  smask := fun _ _ _ _ _ _ => True

def inputs : core.Ctx ≃ Inputs where
  toFun := fun v => ⟨parameter0Equiv (v (0 : Fin 3)), input0Equiv (v (1 : Fin 3)), input1Equiv (v (2 : Fin 3))⟩
  invFun := fun v => Fin.cases (parameter0Equiv.symm v.parameter0) (Fin.cases (input0Equiv.symm v.input0) (Fin.cases (input1Equiv.symm v.input1) ((fun i => Fin.elim0 i))))
  left_inv := by
    intro v
    funext i
    fin_cases i <;> exact Equiv.symm_apply_apply _ _
  right_inv := by
    intro v
    change (⟨parameter0Equiv (parameter0Equiv.symm v.parameter0), input0Equiv (input0Equiv.symm v.input0), input1Equiv (input1Equiv.symm v.input1)⟩ : Inputs) = v
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
  Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.elim0)))

def scratchStorage : Fin core.nScr → TileDType :=
  Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.elim0))))))))))))))))))))))))))))))))))))))

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
end VeriTile.Bench.PretrainKernels.local_rmsnorm_dx
