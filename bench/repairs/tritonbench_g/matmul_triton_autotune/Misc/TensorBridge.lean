import bench.repairs.tritonbench_g.matmul_triton_autotune.Proofs.KernelComplete
import bench.tritonbench_g.matmul_triton_autotune.Proofs.Real
import bench.repairs.tritonbench_g.matmul_triton_autotune.Proofs.KernelNative
import bench.repairs.tritonbench_g.matmul_leakyrelu.KernelOnly.Misc.TensorBridge

/-! The independent mathematical GEMM and optional leaky-ReLU formula.
Only the kernel grid is proved. FP16 casts are identity in the ideal Real
interpretation; actual rounding and output bits belong to FPEquiv. -/
namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Kernel.TensorBridge
open VeriTile.Triton FP.Structural Whole
open VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly
open VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly.TensorBridge (Byte read32 read16 halfBits halfByte actual_half_result)
noncomputable section
set_option maxHeartbeats 3000000

def mathematical (a b : Matrix) (activation : String) (read : Nat → ℝ)
    (i : Fin a.rows) (j : Fin b.columns) : ℝ :=
  let value := ∑ k : Fin a.columns, logical read a i.val k.val * logical read b k.val j.val
  if activation == "leaky_relu" then if 0 ≤ value then value else (1/100 : ℝ)*value else value

theorem mathematical_kernel
    (a b : Matrix) (al : TensorBinding a) (bl : TensorBinding b) (h : InputContract a b)
    (activation : String) (selection : Selection) (c : OutputTensor)
    (cl : OutputBinding c a b) (fresh : Fresh a b c)
    (schedule : Schedule (request a b activation selection)) (read : Nat → ℝ) (scratch : State ℝ) :
    VeriTile.Spec.Real (∃ final,
      Executed (idealAlgebra id) idealDot (request a b activation selection) (initial a b read scratch)
        (tensorAPI a b al bl h activation selection c cl fresh read scratch) schedule.order final ∧
      TensorResult a b c (fun value _ => value) (mathematical a b activation read) read final ∧
      (∀ region offset, ¬Whole.Footprint (request a b activation selection) region offset →
        final.mem region offset = (initial a b read scratch).mem region offset) ∧
      Native.FullAccessSafety (request a b activation selection) (initial a b read scratch)
        (tensorAPI a b al bl h activation selection c cl fresh read scratch)) := by
  obtain ⟨final,he,result,safety⟩ := execute (idealAlgebra id) idealDot (idealNumerics id)
    (ideal_primitives id) a b al bl h activation selection c cl fresh schedule read scratch
  have value (i : Fin a.rows) (j : Fin b.columns) :
      final.mem "c" (Whole.outputOffset (request a b activation selection) i.val j.val) =
        Cell.mk .fp16 (mathematical a b activation read i j) := by
    have v := result.1 i.val i.isLt j.val j.isLt
    rw [ideal_reference id (request a b activation selection) (initial a b read scratch)
      (fun i k => logical read a i.val k.val) (fun k j => logical read b k.val j.val)
      (initial_a a b al.toMatrixLaws activation selection read scratch)
      (initial_b a b bl.toMatrixLaws (admitted a b al.toMatrixLaws bl.toMatrixLaws h).1
        activation selection read scratch) i j] at v
    exact v
  exact ⟨final,he,tensor_result a b activation selection c cl.toOutputLaws fresh
    (fun value _ => value) _ read final value,result.2,safety⟩

#specNonCircular mathematical avoiding [Source.typed, Repair.runPrograms, initial, writeback,
  Whole.reference, Whole.logicalReference]
#axiomsClean mathematical_kernel
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Kernel.TensorBridge

/-! Full-domain FP32 input words, source-ordered live-carry instruction graph,
left-side 0x3c23d70a slope multiplication and mandatory FP16 output bytes.
Fixed instruction arithmetic is the explicit model boundary. There is no
finite-input guard, primitive experiment premise, wrapper theorem, arbitrary
variable-size dot callback or native-to-Real equality. -/
namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Kernel.TensorBridge
open VeriTile.Triton FP.Structural Whole
open VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly
open VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly.TensorBridge (Byte read32 read16 halfBits halfByte actual_half_result)
open ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 3000000
local instance : Inhabited Floating.Word := ⟨FP.Value.zero .fp32⟩

/-- Logical input Tensor coordinates and the explicit per-configuration
padded instruction recurrence. No source state, AST or output memory appears. -/
def floating (instructions : Floating.Instructions) (a b : Matrix) (activation : String)
    (selection : Selection) (heap : Nat → Byte) (i : Fin a.rows) (j : Fin b.columns) : Floating.Word :=
  let value := foldRange (fun q acc => Floating.block instructions selection.config selection.config.bk
    (fun e => if q*selection.config.bk+e.val < a.columns then
      logical (read32 heap) a i.val (q*selection.config.bk+e.val) else Floating.zero)
    (fun e => if q*selection.config.bk+e.val < a.columns then
      logical (read32 heap) b (q*selection.config.bk+e.val) j.val else Floating.zero) acc)
    0 ((a.columns+selection.config.bk-1)/selection.config.bk) 1 Floating.zero
  Floating.narrow (if activation == "leaky_relu" then
    if Floating.nonnegative value then value else FP.mul .ieeeValue Floating.slope value
    else value)

theorem logical_floating (instructions : Floating.Instructions) (a b : Matrix)
    (activation : String) (selection : Selection) (heap : Nat → Byte)
    (i : Fin a.rows) (j : Fin b.columns) :
    Whole.logicalReference (Floating.numerics instructions selection.config)
      (request a b activation selection)
      (fun i k => logical (read32 heap) a i.val k.val)
      (fun k j => logical (read32 heap) b k.val j.val) i j =
      floating instructions a b activation selection heap i j := by
  rfl

theorem floating_kernel
    (instructions : Floating.Instructions)
    (a b : Matrix) (al : TensorBinding a) (bl : TensorBinding b) (h : InputContract a b)
    (activation : String) (selection : Selection) (c : OutputTensor)
    (cl : OutputBinding c a b) (fresh : Fresh a b c)
    (schedule : Schedule (request a b activation selection))
    (heap : Nat → Byte) (scratch : State Floating.Word) :
    ∃ final,
      Executed Floating.scalar (Floating.dot instructions selection.config)
        (request a b activation selection) (initial a b (read32 heap) scratch)
        (tensorAPI a b al bl h activation selection c cl fresh (read32 heap) scratch)
        schedule.order final ∧
      TensorResult a b c halfByte (floating instructions a b activation selection heap) heap final ∧
      (∀ i : Fin a.rows, ∀ j : Fin b.columns,
        read16 (writeback c halfByte heap final) (outputAddress c (i.val*b.columns+j.val)) =
          halfBits (floating instructions a b activation selection heap i j)) ∧
      (∀ region offset, ¬Whole.Footprint (request a b activation selection) region offset →
        final.mem region offset = (initial a b (read32 heap) scratch).mem region offset) ∧
      Native.FullAccessSafety (request a b activation selection) (initial a b (read32 heap) scratch)
        (tensorAPI a b al bl h activation selection c cl fresh (read32 heap) scratch) := by
  obtain ⟨final,he,result,safety⟩ := execute Floating.scalar (Floating.dot instructions selection.config)
    (Floating.numerics instructions selection.config) (Floating.primitives instructions selection.config)
    a b al bl h activation selection c cl fresh schedule (read32 heap) scratch
  have value (i : Fin a.rows) (j : Fin b.columns) :
      final.mem "c" (Whole.outputOffset (request a b activation selection) i.val j.val) =
        Cell.mk .fp16 (floating instructions a b activation selection heap i j) := by
    have v := result.1 i.val i.isLt j.val j.isLt
    rw [Whole.reference_logical (Floating.numerics instructions selection.config)
      (request a b activation selection) (initial a b (read32 heap) scratch)
      (fun i k => logical (read32 heap) a i.val k.val)
      (fun k j => logical (read32 heap) b k.val j.val)
      (initial_a a b al.toMatrixLaws activation selection (read32 heap) scratch)
      (initial_b a b bl.toMatrixLaws (admitted a b al.toMatrixLaws bl.toMatrixLaws h).1
        activation selection (read32 heap) scratch) i j,
      logical_floating instructions a b activation selection heap i j] at v
    exact v
  have actual := tensor_result a b activation selection c cl.toOutputLaws fresh halfByte
    (floating instructions a b activation selection heap) heap final value
  exact ⟨final,he,actual,actual_half_result a b c heap final _ actual,result.2,safety⟩

#specNonCircular floating avoiding [Source.typed, Repair.runPrograms, initial, writeback,
  Whole.reference, Whole.logicalReference]
#axiomsClean logical_floating
#axiomsClean floating_kernel
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Kernel.TensorBridge
