import bench.pretrain_kernels.tasks.local_mla_rope_fused.Kernels
import VeriTile.Triton.Memory.LogicalScheduledKernelIO
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Float.GuardedProgramIO
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Tactic.FinCases

namespace VeriTile.Bench.PretrainKernels.local_mla_rope_fused
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 2000000
namespace IO

structure Inputs where
  input0 : Fin 2 → Fin 12 → Fin 128 → Fin 24 → ℝ
  input1 : Fin 2 → Fin 12 → Fin 128 → Fin 32 → ℝ
  input2 : Fin 2 → Fin 128 → Fin 8 → ℝ

abbrev OutputIndex := Fin 2 × Fin 12 × Fin 128 × Fin 64

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

def input0Equiv : (Fin (2*12*128*24) → ℝ) ≃ (Fin 2 → Fin 12 → Fin 128 → Fin 24 → ℝ) :=
  ((matrixEquiv 2 (12*128*24) ℝ).trans (Equiv.piCongrRight fun _ => ((matrixEquiv 12 (128*24) ℝ).trans (Equiv.piCongrRight fun _ => ((matrixEquiv 128 (24) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 24 → ℝ))))))))

def input1Equiv : (Fin (2*12*128*32) → ℝ) ≃ (Fin 2 → Fin 12 → Fin 128 → Fin 32 → ℝ) :=
  ((matrixEquiv 2 (12*128*32) ℝ).trans (Equiv.piCongrRight fun _ => ((matrixEquiv 12 (128*32) ℝ).trans (Equiv.piCongrRight fun _ => ((matrixEquiv 128 (32) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 32 → ℝ))))))))

def input2Equiv : (Fin (2*128*8) → ℝ) ≃ (Fin 2 → Fin 128 → Fin 8 → ℝ) :=
  ((matrixEquiv 2 (128*8) ℝ).trans (Equiv.piCongrRight fun _ => ((matrixEquiv 128 (8) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 8 → ℝ))))))

def core : UKernelIO where
  kernel := Kernels.original.eraseDType
  projection := by
    have h : Kernels.original.toAlgorithm? = .ok Kernels.original.toAlgKernel := rfl
    simp only [ComputeKernel.eraseDType, h]
    simp [ComputeKernel.toAlgKernel]
  nIn := 3
  nOut := 1
  nScr := 62
  bufs := ["input0", "input1", "input2", "out", "vto_copy", "vdiv", "vpow_1", "vreciprocal", "vmul", "vto_copy_4", "vbmm", "vcat_2", "vcos_1", "vmul_5", "vsin_1", "vmul_6", "vto_copy_5", "vto_copy_6", "vmul_7", "vmul_8", "vsub", "vmul_9", "vmul_10", "vadd", "vcat_3", "vmul_11", "vmul_12", "vsub_1", "vmul_13", "vmul_14", "vadd_1", "vcat_4", "vcat_5", "vcat_6", "vcat_7", "candidate_vto_copy", "candidate_vdiv", "candidate_vpow_1", "candidate_vreciprocal", "candidate_vmul", "candidate_vmul_1", "candidate_vcat", "candidate_vcos", "candidate_vmul_2", "candidate_vto_copy_1", "candidate_vcat_1", "candidate_vsin", "candidate_vmul_3", "candidate_vto_copy_2", "candidate_vmul_5", "candidate_vmul_6", "candidate_vsub", "candidate_vmul_7", "candidate_vmul_8", "candidate_vadd", "candidate_vcat_2", "candidate_vcat_3", "candidate_vmul_9", "candidate_vmul_10", "candidate_vsub_1", "candidate_vmul_11", "candidate_vmul_12", "candidate_vadd_1", "candidate_vcat_4", "candidate_vcat_5", "candidate_vcat_6"]
  ity := Fin.cases (.float) (Fin.cases (.float) (Fin.cases (.float) (Fin.elim0)))
  iarity := Fin.cases (2*12*128*24) (Fin.cases (2*12*128*32) (Fin.cases (2*128*8) (Fin.elim0)))
  ibuf := Fin.cases ("input0") (Fin.cases ("input1") (Fin.cases ("input2") (Fin.elim0)))
  oty := fun _ => .float
  oarity := fun _ => 2*12*128*64
  obuf := fun _ => "out"
  obuf_mem := fun _ => by simp
  sarity := Fin.cases (4) (Fin.cases (4) (Fin.cases (4) (Fin.cases (4) (Fin.cases (4) (Fin.cases (1*1*128) (Fin.cases (1*4*128) (Fin.cases (1*128*8) (Fin.cases (1*128*8) (Fin.cases (1*128*8) (Fin.cases (1*128*8) (Fin.cases (1*128*8) (Fin.cases (1*128*8) (Fin.cases (1*128*8) (Fin.cases (2*12*128*4) (Fin.cases (2*12*128*4) (Fin.cases (2*12*128*4) (Fin.cases (2*12*128*4) (Fin.cases (2*12*128*4) (Fin.cases (2*12*128*4) (Fin.cases (2*12*128*8) (Fin.cases (2*1*128*4) (Fin.cases (2*1*128*4) (Fin.cases (2*1*128*4) (Fin.cases (2*1*128*4) (Fin.cases (2*1*128*4) (Fin.cases (2*1*128*4) (Fin.cases (2*1*128*8) (Fin.cases (2*12*128*24) (Fin.cases (2*12*128*24) (Fin.cases (2*12*128*64) (Fin.cases (4) (Fin.cases (4) (Fin.cases (4) (Fin.cases (4) (Fin.cases (4) (Fin.cases (128*4) (Fin.cases (128*8) (Fin.cases (128*8) (Fin.cases (128*8) (Fin.cases (128*8) (Fin.cases (128*8) (Fin.cases (128*8) (Fin.cases (128*8) (Fin.cases (128*8) (Fin.cases (128*2*12*4) (Fin.cases (128*2*12*4) (Fin.cases (128*2*12*4) (Fin.cases (128*2*12*4) (Fin.cases (128*2*12*4) (Fin.cases (128*2*12*4) (Fin.cases (128*2*12*8) (Fin.cases (128*2*12*24) (Fin.cases (128*2*1*4) (Fin.cases (128*2*1*4) (Fin.cases (128*2*1*4) (Fin.cases (128*2*1*4) (Fin.cases (128*2*1*4) (Fin.cases (128*2*1*4) (Fin.cases (128*2*1*8) (Fin.cases (128*2*12*24) (Fin.cases (128*2*12*64) (Fin.elim0))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
  sbuf := Fin.cases ("vto_copy") (Fin.cases ("vdiv") (Fin.cases ("vpow_1") (Fin.cases ("vreciprocal") (Fin.cases ("vmul") (Fin.cases ("vto_copy_4") (Fin.cases ("vbmm") (Fin.cases ("vcat_2") (Fin.cases ("vcos_1") (Fin.cases ("vmul_5") (Fin.cases ("vsin_1") (Fin.cases ("vmul_6") (Fin.cases ("vto_copy_5") (Fin.cases ("vto_copy_6") (Fin.cases ("vmul_7") (Fin.cases ("vmul_8") (Fin.cases ("vsub") (Fin.cases ("vmul_9") (Fin.cases ("vmul_10") (Fin.cases ("vadd") (Fin.cases ("vcat_3") (Fin.cases ("vmul_11") (Fin.cases ("vmul_12") (Fin.cases ("vsub_1") (Fin.cases ("vmul_13") (Fin.cases ("vmul_14") (Fin.cases ("vadd_1") (Fin.cases ("vcat_4") (Fin.cases ("vcat_5") (Fin.cases ("vcat_6") (Fin.cases ("vcat_7") (Fin.cases ("candidate_vto_copy") (Fin.cases ("candidate_vdiv") (Fin.cases ("candidate_vpow_1") (Fin.cases ("candidate_vreciprocal") (Fin.cases ("candidate_vmul") (Fin.cases ("candidate_vmul_1") (Fin.cases ("candidate_vcat") (Fin.cases ("candidate_vcos") (Fin.cases ("candidate_vmul_2") (Fin.cases ("candidate_vto_copy_1") (Fin.cases ("candidate_vcat_1") (Fin.cases ("candidate_vsin") (Fin.cases ("candidate_vmul_3") (Fin.cases ("candidate_vto_copy_2") (Fin.cases ("candidate_vmul_5") (Fin.cases ("candidate_vmul_6") (Fin.cases ("candidate_vsub") (Fin.cases ("candidate_vmul_7") (Fin.cases ("candidate_vmul_8") (Fin.cases ("candidate_vadd") (Fin.cases ("candidate_vcat_2") (Fin.cases ("candidate_vcat_3") (Fin.cases ("candidate_vmul_9") (Fin.cases ("candidate_vmul_10") (Fin.cases ("candidate_vsub_1") (Fin.cases ("candidate_vmul_11") (Fin.cases ("candidate_vmul_12") (Fin.cases ("candidate_vadd_1") (Fin.cases ("candidate_vcat_4") (Fin.cases ("candidate_vcat_5") (Fin.cases ("candidate_vcat_6") (Fin.elim0))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
  iwin := fun _ _ _ _ _ j => j.val
  imask := fun _ _ _ _ _ _ => True
  owin := fun _ _ _ _ _ j => j.val
  omask := fun _ _ _ _ _ _ => True
  swin := fun _ _ _ _ _ j => j.val
  smask := fun _ _ _ _ _ _ => True

def inputs : core.Ctx ≃ Inputs where
  toFun := fun v => ⟨input0Equiv (v (0 : Fin 3)), input1Equiv (v (1 : Fin 3)), input2Equiv (v (2 : Fin 3))⟩
  invFun := fun v => Fin.cases (input0Equiv.symm v.input0) (Fin.cases (input1Equiv.symm v.input1) (Fin.cases (input2Equiv.symm v.input2) ((fun i => Fin.elim0 i))))
  left_inv := by
    intro v
    funext i
    fin_cases i <;> exact Equiv.symm_apply_apply _ _
  right_inv := by
    intro v
    change (⟨input0Equiv (input0Equiv.symm v.input0), input1Equiv (input1Equiv.symm v.input1), input2Equiv (input2Equiv.symm v.input2)⟩ : Inputs) = v
    simp only [Equiv.apply_symm_apply]

def outputCoords : Fin (2*12*128*64) ≃ OutputIndex :=
  (finProdFinEquiv (m := 2) (n := 12*128*64)).symm.trans (Equiv.prodCongr (Equiv.refl _) ((finProdFinEquiv (m := 12) (n := 128*64)).symm.trans (Equiv.prodCongr (Equiv.refl _) ((finProdFinEquiv (m := 128) (n := 64)).symm))))

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
  Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.real) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.elim0))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

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
end VeriTile.Bench.PretrainKernels.local_mla_rope_fused
