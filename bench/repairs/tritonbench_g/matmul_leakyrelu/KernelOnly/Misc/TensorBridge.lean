import bench.repairs.tritonbench_g.matmul_leakyrelu.KernelOnly.Misc.CandidateCorrect
import bench.repairs.tritonbench_g.matmul_leakyrelu.KernelOnly.Misc.NativeCorrect
import bench.repairs.tritonbench_g.matmul_leakyrelu.KernelOnly.FPEquiv
import bench.repairs.tritonbench_g.matmul_leakyrelu.KernelOnly.Memory

namespace VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly.TensorBridge
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatmulLeakyrelu
noncomputable section
set_option maxHeartbeats 3000000

/-- The checked Tensor request has precisely the raw kernel parameters. -/
theorem kernel_request (a b : Matrix) (al : MatrixLaws a) (bl : MatrixLaws b)
    (h : InputContract (metadata a b)) (activation : String) :
    request a b activation =
      (KernelInterface.Parameters.mk "a" "b" "c" a.rows a.columns b.columns
        a.strideRow a.strideColumn b.strideRow b.strideColumn activation).request := by
  have conditions := admitted a b al bl h activation
  simp only [request,KernelInterface.Parameters.request,conditions.2.1,conditions.2.2.1]

/-- Independent finite matrix product and optional slope 1/100. Identity
narrowing is the exact Real projection; this is not native FP16 equality. -/
def mathematical (a b : Matrix) (activation : String) (heap : Nat → ℝ)
    (i : Fin a.rows) (j : Fin b.columns) : ℝ :=
  Whole.mathematical (request a b activation)
    (fun i k => logical heap a i.val k.val) (fun k j => logical heap b k.val j.val) i j

theorem mathematical_kernel
    (a b : Matrix) (al : MatrixLaws a) (bl : MatrixLaws b)
    (h : InputContract (metadata a b)) (activation : String)
    (c : OutputTensor) (cl : OutputLaws c a b) (fresh : Fresh a b c)
    (output : outputCheck (metadata a b) (outputBase c) = Bool.true)
    (schedule : Schedule (request a b activation)) (heap : Nat → ℝ) (scratch : State ℝ) :
    VeriTile.Spec.Real (∃ final,
      run (Whole.idealAlgebra id) Whole.Contracted.idealDotAcc (request a b activation)
        schedule.order (initial a b heap scratch) = some final ∧
      (∀ i : Fin a.rows, ∀ j : Fin b.columns,
        final.mem "c" (Whole.outputOffset (request a b activation) i.val j.val) =
          Cell.mk .fp16 (mathematical a b activation heap i j)) ∧
      TensorResult a b c (fun x _ => x) (mathematical a b activation heap) heap final ∧
      (∀ region offset, ¬Whole.Footprint (request a b activation) region offset →
        final.mem region offset = (initial a b heap scratch).mem region offset) ∧
      Whole.SourceSafety (request a b activation) (initial a b heap scratch)
        (tensorContract a b al bl h activation c cl fresh output heap scratch)) := by
  obtain ⟨final, he, result, safety⟩ := execute (Whole.idealAlgebra id)
    Whole.Contracted.idealDotAcc (Whole.Contracted.idealNumerics id)
    (Whole.Contracted.ideal_primitives id) a b al bl h activation c cl fresh output schedule heap scratch
  have hv (i : Fin a.rows) (j : Fin b.columns) :
      final.mem "c" (Whole.outputOffset (request a b activation) i.val j.val) =
        Cell.mk .fp16 (mathematical a b activation heap i j) := by
    have v := result.1 i.val i.isLt j.val j.isLt
    rw [Whole.Contracted.ideal_reference id (request a b activation) (initial a b heap scratch)
      (fun i k => logical heap a i.val k.val) (fun k j => logical heap b k.val j.val)
      (initial_a a b al activation heap scratch)
      (initial_b a b bl (admitted a b al bl h activation).2.1 activation heap scratch) i j] at v
    exact v
  exact ⟨final, he, hv, tensor_result a b activation c cl fresh _ _ heap final hv,
    result.2, safety⟩

abbrev Byte := BitVec 8
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def read32 (heap : Nat → Byte) (p : Nat) : Word :=
  FP.Value.ofBits .fp32 ((heap p).toNat+256*(heap (p+1)).toNat+
    65536*(heap (p+2)).toNat+16777216*(heap (p+3)).toNat)
def halfBits (x : Word) : BitVec 16 := BitVec.ofNat 16 x.bits.toNat
def halfByte (x : Word) (byte : Fin 2) : Byte :=
  BitVec.ofNat 8 (x.bits.toNat / 256^byte.val)
def read16 (heap : Nat → Byte) (p : Nat) : BitVec 16 :=
  BitVec.ofNat 16 ((heap p).toNat+256*(heap (p+1)).toNat)

theorem half_reconstruct (x : Word) :
    BitVec.ofNat 16 ((halfByte x ⟨0,by decide⟩).toNat+
      256*(halfByte x ⟨1,by decide⟩).toNat) = halfBits x := by
  apply BitVec.eq_of_toNat_eq
  simp only [halfByte, halfBits, BitVec.toNat_ofNat]
  norm_num
  omega

/-- Independent logical matrices and the ordered K32 live-carry recurrence;
output cells carry the FP16 payload in their low sixteen bits. -/
def floating (Q : Whole.Contracted.Numerics Word) (a b : Matrix) (activation : String)
    (heap : Nat → Byte) (i : Fin a.rows) (j : Fin b.columns) : Word :=
  Whole.Contracted.logicalReference Q (request a b activation)
    (fun i k => logical (read32 heap) a i.val k.val)
    (fun k j => logical (read32 heap) b k.val j.val) i j

theorem actual_half_result (a b : Matrix) (c : OutputTensor)
    (heap : Nat → Byte) (final : State Word) (target : Fin a.rows → Fin b.columns → Word)
    (result : TensorResult a b c halfByte target heap final) (i : Fin a.rows) (j : Fin b.columns) :
    read16 (writeback c halfByte heap final) (outputAddress c (i.val*b.columns+j.val)) =
      halfBits (target i j) := by
  unfold read16
  have h0 := result.1 i j ⟨0, by decide⟩
  have h1 := result.1 i j ⟨1, by decide⟩
  simp only [Nat.add_zero] at h0
  rw [h0,h1]
  exact half_reconstruct _

theorem floating_kernel
    (M : Algebra Word) (D : FPFused.DotAcc Word) (Q : Whole.Contracted.Numerics Word)
    (law : Whole.Contracted.PrimitiveLaws M D Q)
    (a b : Matrix) (al : MatrixLaws a) (bl : MatrixLaws b)
    (h : InputContract (metadata a b)) (activation : String)
    (c : OutputTensor) (cl : OutputLaws c a b) (fresh : Fresh a b c)
    (output : outputCheck (metadata a b) (outputBase c) = Bool.true)
    (schedule : Schedule (request a b activation)) (heap : Nat → Byte) (scratch : State Word) :
    ∃ final,
      run M D (request a b activation) schedule.order (initial a b (read32 heap) scratch) = some final ∧
      TensorResult a b c halfByte (floating Q a b activation heap) heap final ∧
      (∀ i : Fin a.rows, ∀ j : Fin b.columns,
        read16 (writeback c halfByte heap final) (outputAddress c (i.val*b.columns+j.val)) =
          halfBits (floating Q a b activation heap i j)) ∧
      (∀ region offset, ¬Whole.Footprint (request a b activation) region offset →
        final.mem region offset = (initial a b (read32 heap) scratch).mem region offset) ∧
      Whole.SourceSafety (request a b activation) (initial a b (read32 heap) scratch)
        (tensorContract a b al bl h activation c cl fresh output (read32 heap) scratch) := by
  obtain ⟨final, he, result, safety⟩ := execute M D Q law a b al bl h activation c cl fresh output
    schedule (read32 heap) scratch
  have hv (i : Fin a.rows) (j : Fin b.columns) :
      final.mem "c" (Whole.outputOffset (request a b activation) i.val j.val) =
        Cell.mk .fp16 (floating Q a b activation heap i j) := by
    have v := result.1 i.val i.isLt j.val j.isLt
    rw [Whole.Contracted.reference_logical Q (request a b activation) (initial a b (read32 heap) scratch)
      (fun i k => logical (read32 heap) a i.val k.val)
      (fun k j => logical (read32 heap) b k.val j.val)
      (initial_a a b al activation (read32 heap) scratch)
      (initial_b a b bl (admitted a b al bl h activation).2.1 activation (read32 heap) scratch) i j] at v
    exact v
  have actual := tensor_result a b activation c cl fresh halfByte _ heap final hv
  exact ⟨final, he, actual, actual_half_result a b c heap final _ actual, result.2, safety⟩

#axiomsClean mathematical_kernel
#axiomsClean floating_kernel
#specNonCircular mathematical avoiding [run,dsl,initial,writeback]
#specNonCircular floating avoiding [run,dsl,initial,writeback]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly.TensorBridge
