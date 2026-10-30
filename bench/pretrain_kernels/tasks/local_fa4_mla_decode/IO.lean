import bench.pretrain_kernels.tasks.local_fa4_mla_decode.Kernels
import VeriTile.Triton.Memory.LogicalScheduledKernelIO
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Float.GuardedProgramIO
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Tactic.FinCases

namespace VeriTile.Bench.PretrainKernels.local_fa4_mla_decode
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 2000000
namespace IO

structure Inputs where
  input0 : Fin 4 → Fin 1 → Fin 8 → Fin 64 → ℝ
  input1 : Fin 4 → Fin 256 → Fin 1 → Fin 64 → ℝ
  input2 : Fin 4 → Fin 1 → Fin 8 → Fin 512 → ℝ
  input3 : Fin 4 → Fin 256 → Fin 1 → Fin 512 → ℝ

abbrev OutputIndex := Fin 4 × Fin 1 × Fin 8 × Fin 512

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

def input0Equiv : (Fin (4*1*8*64) → ℝ) ≃ (Fin 4 → Fin 1 → Fin 8 → Fin 64 → ℝ) :=
  ((matrixEquiv 4 (1*8*64) ℝ).trans (Equiv.piCongrRight fun _ => ((matrixEquiv 1 (8*64) ℝ).trans (Equiv.piCongrRight fun _ => ((matrixEquiv 8 (64) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 64 → ℝ))))))))

def input1Equiv : (Fin (4*256*1*64) → ℝ) ≃ (Fin 4 → Fin 256 → Fin 1 → Fin 64 → ℝ) :=
  ((matrixEquiv 4 (256*1*64) ℝ).trans (Equiv.piCongrRight fun _ => ((matrixEquiv 256 (1*64) ℝ).trans (Equiv.piCongrRight fun _ => ((matrixEquiv 1 (64) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 64 → ℝ))))))))

def input2Equiv : (Fin (4*1*8*512) → ℝ) ≃ (Fin 4 → Fin 1 → Fin 8 → Fin 512 → ℝ) :=
  ((matrixEquiv 4 (1*8*512) ℝ).trans (Equiv.piCongrRight fun _ => ((matrixEquiv 1 (8*512) ℝ).trans (Equiv.piCongrRight fun _ => ((matrixEquiv 8 (512) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 512 → ℝ))))))))

def input3Equiv : (Fin (4*256*1*512) → ℝ) ≃ (Fin 4 → Fin 256 → Fin 1 → Fin 512 → ℝ) :=
  ((matrixEquiv 4 (256*1*512) ℝ).trans (Equiv.piCongrRight fun _ => ((matrixEquiv 256 (1*512) ℝ).trans (Equiv.piCongrRight fun _ => ((matrixEquiv 1 (512) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 512 → ℝ))))))))

def core : UKernelIO where
  kernel := Kernels.original.eraseDType
  projection := by
    have h : Kernels.original.toAlgorithm? = .ok Kernels.original.toAlgKernel := rfl
    simp only [ComputeKernel.eraseDType, h]
    simp [ComputeKernel.toAlgKernel]
  nIn := 4
  nOut := 1
  nScr := 69
  bufs := ["input0", "input1", "input2", "input3", "out", "vcat", "vcat_1", "vbmm", "vmul", "vto_copy", "vsoftmax", "vto_copy_1", "vbmm_1", "candidate_vto_copy", "candidate_vto_copy_1", "candidate_vbmm", "candidate_vto_copy_2", "candidate_vto_copy_3", "candidate_vbmm_1", "candidate_vadd", "candidate_vmul", "candidate_vadd_1", "candidate_vsub", "candidate_vgt", "candidate_vmasked_fill", "candidate_vsoftmax", "candidate_vto_copy_4", "candidate_vmm", "candidate_vto_copy_5", "candidate_vto_copy_6", "candidate_vbmm_2", "candidate_vto_copy_7", "candidate_vto_copy_8", "candidate_vbmm_3", "candidate_vadd_2", "candidate_vmul_1", "candidate_vadd_3", "candidate_vsub_1", "candidate_vgt_1", "candidate_vmasked_fill_1", "candidate_vsoftmax_1", "candidate_vto_copy_9", "candidate_vmm_1", "candidate_vto_copy_10", "candidate_vto_copy_11", "candidate_vbmm_4", "candidate_vto_copy_12", "candidate_vto_copy_13", "candidate_vbmm_5", "candidate_vadd_4", "candidate_vmul_2", "candidate_vadd_5", "candidate_vsub_2", "candidate_vgt_2", "candidate_vmasked_fill_2", "candidate_vsoftmax_2", "candidate_vto_copy_14", "candidate_vmm_2", "candidate_vto_copy_15", "candidate_vto_copy_16", "candidate_vbmm_6", "candidate_vto_copy_17", "candidate_vto_copy_18", "candidate_vbmm_7", "candidate_vadd_6", "candidate_vmul_3", "candidate_vadd_7", "candidate_vsub_3", "candidate_vgt_3", "candidate_vmasked_fill_3", "candidate_vsoftmax_3", "candidate_vto_copy_19", "candidate_vmm_3", "candidate_vcat"]
  ity := Fin.cases (.float) (Fin.cases (.float) (Fin.cases (.float) (Fin.cases (.float) (Fin.elim0))))
  iarity := Fin.cases (4*1*8*64) (Fin.cases (4*256*1*64) (Fin.cases (4*1*8*512) (Fin.cases (4*256*1*512) (Fin.elim0))))
  ibuf := Fin.cases ("input0") (Fin.cases ("input1") (Fin.cases ("input2") (Fin.cases ("input3") (Fin.elim0))))
  oty := fun _ => .float
  oarity := fun _ => 4*1*8*512
  obuf := fun _ => "out"
  obuf_mem := fun _ => by simp
  sarity := Fin.cases (4*1*8*576) (Fin.cases (4*256*1*576) (Fin.cases (32*1*256) (Fin.cases (4*8*1*256) (Fin.cases (4*8*1*256) (Fin.cases (4*8*1*256) (Fin.cases (4*8*1*256) (Fin.cases (32*1*512) (Fin.cases (8*1*64) (Fin.cases (1*256*64) (Fin.cases (8*1*256) (Fin.cases (8*1*512) (Fin.cases (1*256*512) (Fin.cases (8*1*256) (Fin.cases (8*1*256) (Fin.cases (8*1*256) (Fin.cases (1*1) (Fin.cases (1*1) (Fin.cases (1*256) (Fin.cases (8*1*256) (Fin.cases (8*1*256) (Fin.cases (8*1*256) (Fin.cases (8*512) (Fin.cases (8*1*64) (Fin.cases (1*256*64) (Fin.cases (8*1*256) (Fin.cases (8*1*512) (Fin.cases (1*256*512) (Fin.cases (8*1*256) (Fin.cases (8*1*256) (Fin.cases (8*1*256) (Fin.cases (1*1) (Fin.cases (1*1) (Fin.cases (1*256) (Fin.cases (8*1*256) (Fin.cases (8*1*256) (Fin.cases (8*1*256) (Fin.cases (8*512) (Fin.cases (8*1*64) (Fin.cases (1*256*64) (Fin.cases (8*1*256) (Fin.cases (8*1*512) (Fin.cases (1*256*512) (Fin.cases (8*1*256) (Fin.cases (8*1*256) (Fin.cases (8*1*256) (Fin.cases (1*1) (Fin.cases (1*1) (Fin.cases (1*256) (Fin.cases (8*1*256) (Fin.cases (8*1*256) (Fin.cases (8*1*256) (Fin.cases (8*512) (Fin.cases (8*1*64) (Fin.cases (1*256*64) (Fin.cases (8*1*256) (Fin.cases (8*1*512) (Fin.cases (1*256*512) (Fin.cases (8*1*256) (Fin.cases (8*1*256) (Fin.cases (8*1*256) (Fin.cases (1*1) (Fin.cases (1*1) (Fin.cases (1*256) (Fin.cases (8*1*256) (Fin.cases (8*1*256) (Fin.cases (8*1*256) (Fin.cases (8*512) (Fin.cases (4*8*512) (Fin.elim0)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
  sbuf := Fin.cases ("vcat") (Fin.cases ("vcat_1") (Fin.cases ("vbmm") (Fin.cases ("vmul") (Fin.cases ("vto_copy") (Fin.cases ("vsoftmax") (Fin.cases ("vto_copy_1") (Fin.cases ("vbmm_1") (Fin.cases ("candidate_vto_copy") (Fin.cases ("candidate_vto_copy_1") (Fin.cases ("candidate_vbmm") (Fin.cases ("candidate_vto_copy_2") (Fin.cases ("candidate_vto_copy_3") (Fin.cases ("candidate_vbmm_1") (Fin.cases ("candidate_vadd") (Fin.cases ("candidate_vmul") (Fin.cases ("candidate_vadd_1") (Fin.cases ("candidate_vsub") (Fin.cases ("candidate_vgt") (Fin.cases ("candidate_vmasked_fill") (Fin.cases ("candidate_vsoftmax") (Fin.cases ("candidate_vto_copy_4") (Fin.cases ("candidate_vmm") (Fin.cases ("candidate_vto_copy_5") (Fin.cases ("candidate_vto_copy_6") (Fin.cases ("candidate_vbmm_2") (Fin.cases ("candidate_vto_copy_7") (Fin.cases ("candidate_vto_copy_8") (Fin.cases ("candidate_vbmm_3") (Fin.cases ("candidate_vadd_2") (Fin.cases ("candidate_vmul_1") (Fin.cases ("candidate_vadd_3") (Fin.cases ("candidate_vsub_1") (Fin.cases ("candidate_vgt_1") (Fin.cases ("candidate_vmasked_fill_1") (Fin.cases ("candidate_vsoftmax_1") (Fin.cases ("candidate_vto_copy_9") (Fin.cases ("candidate_vmm_1") (Fin.cases ("candidate_vto_copy_10") (Fin.cases ("candidate_vto_copy_11") (Fin.cases ("candidate_vbmm_4") (Fin.cases ("candidate_vto_copy_12") (Fin.cases ("candidate_vto_copy_13") (Fin.cases ("candidate_vbmm_5") (Fin.cases ("candidate_vadd_4") (Fin.cases ("candidate_vmul_2") (Fin.cases ("candidate_vadd_5") (Fin.cases ("candidate_vsub_2") (Fin.cases ("candidate_vgt_2") (Fin.cases ("candidate_vmasked_fill_2") (Fin.cases ("candidate_vsoftmax_2") (Fin.cases ("candidate_vto_copy_14") (Fin.cases ("candidate_vmm_2") (Fin.cases ("candidate_vto_copy_15") (Fin.cases ("candidate_vto_copy_16") (Fin.cases ("candidate_vbmm_6") (Fin.cases ("candidate_vto_copy_17") (Fin.cases ("candidate_vto_copy_18") (Fin.cases ("candidate_vbmm_7") (Fin.cases ("candidate_vadd_6") (Fin.cases ("candidate_vmul_3") (Fin.cases ("candidate_vadd_7") (Fin.cases ("candidate_vsub_3") (Fin.cases ("candidate_vgt_3") (Fin.cases ("candidate_vmasked_fill_3") (Fin.cases ("candidate_vsoftmax_3") (Fin.cases ("candidate_vto_copy_19") (Fin.cases ("candidate_vmm_3") (Fin.cases ("candidate_vcat") (Fin.elim0)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
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

def outputCoords : Fin (4*1*8*512) ≃ OutputIndex :=
  (finProdFinEquiv (m := 4) (n := 1*8*512)).symm.trans (Equiv.prodCongr (Equiv.refl _) ((finProdFinEquiv (m := 1) (n := 8*512)).symm.trans (Equiv.prodCongr (Equiv.refl _) ((finProdFinEquiv (m := 8) (n := 512)).symm))))

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
  Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.bool) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.bool) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.bool) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.bool) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.elim0)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

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
end VeriTile.Bench.PretrainKernels.local_fa4_mla_decode
