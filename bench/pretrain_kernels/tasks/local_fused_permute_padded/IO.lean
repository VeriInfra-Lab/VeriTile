import bench.pretrain_kernels.tasks.local_fused_permute_padded.Kernels
import VeriTile.Triton.Memory.LogicalScheduledKernelIO
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Float.GuardedProgramIO
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Tactic.FinCases

namespace VeriTile.Bench.PretrainKernels.local_fused_permute_padded
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 2000000
namespace IO

structure Inputs where
  parameter0 : Fin 256 → Fin 128 → ℝ
  input0 : Fin 256 → Fin 2048 → ℝ

abbrev OutputIndex := Fin 1856 × Fin 2048

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

def parameter0Equiv : (Fin (256*128) → ℝ) ≃ (Fin 256 → Fin 128 → ℝ) :=
  ((matrixEquiv 256 (128) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 128 → ℝ))))

def input0Equiv : (Fin (256*2048) → ℝ) ≃ (Fin 256 → Fin 2048 → ℝ) :=
  ((matrixEquiv 256 (2048) ℝ).trans (Equiv.piCongrRight fun _ => (Equiv.refl (Fin 2048 → ℝ))))

def core : UKernelIO where
  kernel := Kernels.original.eraseDType
  projection := by
    have h : Kernels.original.toAlgorithm? = .ok Kernels.original.toAlgKernel := rfl
    simp only [ComputeKernel.eraseDType, h]
    simp [ComputeKernel.toAlgKernel]
  nIn := 2
  nOut := 1
  nScr := 45
  bufs := ["parameter0", "input0", "out", "vtopk_value", "vtopk_index", "vscatter", "vto_copy_5", "vto_copy_6", "vsum_3", "vadd_2", "vsub_3", "vfloor_divide_1", "vmul_1", "vcumsum_2", "vsub_4", "vto_copy_7", "vcumsum_3", "vsub_5", "vadd_3", "vwhere_4", "veq", "vwhere_5", "vsum_4", "vany_2", "vindex", "vwhere_6", "candidate_vtopk_value", "candidate_vtopk_index", "candidate_vscatter", "candidate_vto_copy_5", "candidate_vto_copy_6", "candidate_vto_copy_7", "candidate_vto_copy_8", "candidate_vsum_3", "candidate_vadd_2", "candidate_vsub_3", "candidate_vfloor_divide_1", "candidate_vmul_1", "candidate_vcumsum_2", "candidate_vsub_4", "candidate_vto_copy_9", "candidate_vcumsum_3", "candidate_vsub_5", "candidate_vadd_3", "candidate_vwhere_4", "candidate_vwhere_5", "candidate_vwhere_6", "candidate_vscatter_add_2"]
  ity := Fin.cases (.float) (Fin.cases (.float) (Fin.elim0))
  iarity := Fin.cases (256*128) (Fin.cases (256*2048) (Fin.elim0))
  ibuf := Fin.cases ("parameter0") (Fin.cases ("input0") (Fin.elim0))
  oty := fun _ => .float
  oarity := fun _ => 1856*2048
  obuf := fun _ => "out"
  obuf_mem := fun _ => by simp
  sarity := Fin.cases (256*1) (Fin.cases (256*1) (Fin.cases (256*128) (Fin.cases (256*2048) (Fin.cases (256*128) (Fin.cases (128) (Fin.cases (128) (Fin.cases (128) (Fin.cases (128) (Fin.cases (128) (Fin.cases (128) (Fin.cases (128) (Fin.cases (256*128) (Fin.cases (256*128) (Fin.cases (256*128) (Fin.cases (256*128) (Fin.cases (256*128) (Fin.cases (1856*32768) (Fin.cases (1856*32768) (Fin.cases (1856) (Fin.cases (1856) (Fin.cases (1856*2048) (Fin.cases (1856*2048) (Fin.cases (256*1) (Fin.cases (256*1) (Fin.cases (256*128) (Fin.cases (256*2048) (Fin.cases (256*128) (Fin.cases (256*128) (Fin.cases (256*128) (Fin.cases (128) (Fin.cases (128) (Fin.cases (128) (Fin.cases (128) (Fin.cases (128) (Fin.cases (128) (Fin.cases (128) (Fin.cases (256*128) (Fin.cases (256*128) (Fin.cases (256*128) (Fin.cases (256*128) (Fin.cases (256*128) (Fin.cases (256*128) (Fin.cases (256*128*2048) (Fin.cases (1857*2048) (Fin.elim0)))))))))))))))))))))))))))))))))))))))))))))
  sbuf := Fin.cases ("vtopk_value") (Fin.cases ("vtopk_index") (Fin.cases ("vscatter") (Fin.cases ("vto_copy_5") (Fin.cases ("vto_copy_6") (Fin.cases ("vsum_3") (Fin.cases ("vadd_2") (Fin.cases ("vsub_3") (Fin.cases ("vfloor_divide_1") (Fin.cases ("vmul_1") (Fin.cases ("vcumsum_2") (Fin.cases ("vsub_4") (Fin.cases ("vto_copy_7") (Fin.cases ("vcumsum_3") (Fin.cases ("vsub_5") (Fin.cases ("vadd_3") (Fin.cases ("vwhere_4") (Fin.cases ("veq") (Fin.cases ("vwhere_5") (Fin.cases ("vsum_4") (Fin.cases ("vany_2") (Fin.cases ("vindex") (Fin.cases ("vwhere_6") (Fin.cases ("candidate_vtopk_value") (Fin.cases ("candidate_vtopk_index") (Fin.cases ("candidate_vscatter") (Fin.cases ("candidate_vto_copy_5") (Fin.cases ("candidate_vto_copy_6") (Fin.cases ("candidate_vto_copy_7") (Fin.cases ("candidate_vto_copy_8") (Fin.cases ("candidate_vsum_3") (Fin.cases ("candidate_vadd_2") (Fin.cases ("candidate_vsub_3") (Fin.cases ("candidate_vfloor_divide_1") (Fin.cases ("candidate_vmul_1") (Fin.cases ("candidate_vcumsum_2") (Fin.cases ("candidate_vsub_4") (Fin.cases ("candidate_vto_copy_9") (Fin.cases ("candidate_vcumsum_3") (Fin.cases ("candidate_vsub_5") (Fin.cases ("candidate_vadd_3") (Fin.cases ("candidate_vwhere_4") (Fin.cases ("candidate_vwhere_5") (Fin.cases ("candidate_vwhere_6") (Fin.cases ("candidate_vscatter_add_2") (Fin.elim0)))))))))))))))))))))))))))))))))))))))))))))
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

def outputCoords : Fin (1856*2048) ≃ OutputIndex :=
  (finProdFinEquiv (m := 1856) (n := 2048)).symm

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
  Fin.cases (.real) (Fin.cases (.real) (Fin.elim0))

def scratchStorage : Fin core.nScr → TileDType :=
  Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.bool) (Fin.cases (.bf16) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.bool) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.bool) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.cases (.real) (Fin.cases (.int) (Fin.cases (.bool) (Fin.cases (.bf16) (Fin.cases (.int) (Fin.cases (.bool) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.int) (Fin.cases (.bf16) (Fin.cases (.bf16) (Fin.elim0)))))))))))))))))))))))))))))))))))))))))))))

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

/-- Capacity in the public mathematical metadata view. The decoder must be
bound to the selected numerical models; this is not an implicit conversion. -/
def metadataAdmissible {α : Type} (decode : α → ℝ) (vals : (io α).Inputs) : Prop :=
  (let parameter0 := parameter0Equiv (fun j : Fin 32768 => decode (vals (0 : Fin 2) (j, PUnit.unit)));
let vtopk_index : Fin 256 → Fin 1 → Int := fun i0 i1 => ((((List.finRange 128).mergeSort (fun left right => decide (parameter0 i0 ⟨(left.val) % 128, Nat.mod_lt _ (by decide)⟩ > parameter0 i0 ⟨(right.val) % 128, Nat.mod_lt _ (by decide)⟩ ∨ (parameter0 i0 ⟨(left.val) % 128, Nat.mod_lt _ (by decide)⟩ = parameter0 i0 ⟨(right.val) % 128, Nat.mod_lt _ (by decide)⟩ ∧ left.val ≤ right.val)))).getD i1.val 0).val : Int);
let vscatter : Fin 256 → Fin 128 → Bool := fun i0 i1 => ((List.finRange 1).foldl (fun value lane => if vtopk_index i0 ⟨(lane.val) % 1, Nat.mod_lt _ (by decide)⟩ = (i1.val : Int) then Bool.true else value) (Bool.false));
let vto_copy_6 : Fin 256 → Fin 128 → Int := fun i0 i1 => (if vscatter i0 i1 then 1 else 0);
let vsum_3 : Fin 128 → Int := fun i0 => (∑ lane : Fin 256, vto_copy_6 lane i0);
let candidate_vtopk_index : Fin 256 → Fin 1 → Int := fun i0 i1 => ((((List.finRange 128).mergeSort (fun left right => decide (parameter0 i0 ⟨(left.val) % 128, Nat.mod_lt _ (by decide)⟩ > parameter0 i0 ⟨(right.val) % 128, Nat.mod_lt _ (by decide)⟩ ∨ (parameter0 i0 ⟨(left.val) % 128, Nat.mod_lt _ (by decide)⟩ = parameter0 i0 ⟨(right.val) % 128, Nat.mod_lt _ (by decide)⟩ ∧ left.val ≤ right.val)))).getD i1.val 0).val : Int);
let candidate_vscatter : Fin 256 → Fin 128 → Bool := fun i0 i1 => ((List.finRange 1).foldl (fun value lane => if candidate_vtopk_index i0 ⟨(lane.val) % 1, Nat.mod_lt _ (by decide)⟩ = (i1.val : Int) then Bool.true else value) (Bool.false));
let candidate_vto_copy_6 : Fin 256 → Fin 128 → Int := fun i0 i1 => (if candidate_vscatter i0 i1 then 1 else 0);
let candidate_vto_copy_7 : Fin 256 → Fin 128 → Bool := fun i0 i1 => decide (candidate_vto_copy_6 i0 i1 ≠ 0);
let candidate_vto_copy_8 : Fin 256 → Fin 128 → Int := fun i0 i1 => (if candidate_vto_copy_7 i0 i1 then 1 else 0);
let candidate_vsum_3 : Fin 128 → Int := fun i0 => (∑ lane : Fin 256, candidate_vto_copy_8 lane i0);
(∑ expert : Fin 128, ((vsum_3 ⟨(expert.val) % 128, Nat.mod_lt _ (by decide)⟩ + 15) / 16) * 16) = 1856 ∧ (∑ expert : Fin 128, ((candidate_vsum_3 ⟨(expert.val) % 128, Nat.mod_lt _ (by decide)⟩ + 15) / 16) * 16) = 1856)

end FP
end IO
end
end VeriTile.Bench.PretrainKernels.local_fused_permute_padded
