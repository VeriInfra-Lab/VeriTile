import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8_scalar_cast.Misc.NativeCorrect
import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8_scalar_cast.Correct
import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8_scalar_cast.FPEquiv
import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8_scalar_cast.Memory

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.TensorBridge
open VeriTile.Triton
open MatmulLeakyrelu.KernelOnly MatmulLeakyreluFp8 MatmulLeakyreluFp8.Checked
open VeriTile.Bench.TritonBenchG.MatmulPersistentTriton
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 8000000

def mathematical (r : Request) (read : Nat → ℝ) (i j : Nat) : ℝ :=
  Real.activate (r.activation == "leaky_relu")
    (∑ k ∈ Finset.range r.A.columns, logical read r.A i k * logical read r.B k j)

theorem real_reference (args : Host.Arguments) (out : Tensor)
    (h : Layout args.problem ⟨args.A,args.B,out⟩)
    (memory : Memory ℝ) (i j : Nat) (hi : i < args.A.rows) (hj : j < args.B.columns) :
    reference realOps args memory i j = Real.formula args.problem ⟨args.A,args.B,out⟩ memory i j := by
  by_cases zero : args.A.columns = 0
  · simp [reference,zero,realOps,Real.formula,Real.activate,RealMath.formula,Host.Arguments.problem]
  · have source := grid_reference realOps args out h memory zero i j hi hj
    rw [source_run_real] at source
    exact source.symm.trans (Real.output args.problem ⟨args.A,args.B,out⟩ h memory i j hi hj)

theorem mathematical_reference (r : Request) (c : OutputTensor) (h : Contract r c)
    (read : Nat → ℝ) (scratch : Memory ℝ) (i j : Nat)
    (hi : i < r.A.rows) (hj : j < r.B.columns) :
    reference realOps (arguments r) (initial r read scratch) i j = mathematical r read i j := by
  rw [real_reference (arguments r) (outputDescriptor r c) (layout r c h) _ i j hi hj]
  unfold Real.formula RealMath.formula Real.inputs mathematical
  apply congrArg (Real.activate (r.activation == "leaky_relu"))
  apply Finset.sum_congr rfl
  intro k hk
  have hkk : k < r.A.columns := Finset.mem_range.mp hk
  change initial r read scratch ((arguments r).A.address i k) *
    initial r read scratch ((arguments r).B.address k j) = _
  rw [initial_a r h read scratch i k hi hkk,initial_b r h read scratch k j hkk hj]

/-- Complete event computation, exact independent finite-K formula, actual
output Tensor addresses and complement/full-input-backing frame. -/
theorem correct_event (r : Request) (c : OutputTensor) (h : Contract r c)
    (trials : Nat) (read : Nat → ℝ) (scratch : Memory ℝ) (before : Nat → ℝ) :
    ∃ final, eventRun realOps r c trials read scratch = ((),final) ∧
      TensorResult r c (fun value _ => value) (fun i j => mathematical r read i.val j.val) before final := by
  let final := (eventRun realOps r c trials read scratch).2
  refine ⟨final,?_,?_⟩
  · dsimp [final]
  · apply tensor_result r c h
    intro i j
    change (eventRun realOps r c trials read scratch).2 _ = _
    rw [event_value realOps r c h trials read scratch i.val j.val i.isLt j.isLt,
      ←reference_logical realOps r c h read scratch i.val j.val i.isLt j.isLt]
    exact mathematical_reference r c h read scratch i.val j.val i.isLt j.isLt

/-- Complete source computation, with actual Tensor addresses, the independent
finite-K Real formula, and complement/full-input-backing frame. -/
theorem correct (M : FP.Structural.Algebra ℝ) (law : Execution.ScalarLaws M realOps)
    (r : Request) (c : OutputTensor) (h : Contract r c) (trials : Nat)
    (read : Nat → ℝ) (scratch : Memory ℝ) (registers : FP.Structural.State ℝ) (before : Nat → ℝ) :
    ∃ final, Execution.computation M realOps r c trials registers (initial r read scratch) = some final ∧
      TensorResult r c (fun value _ => value) (fun i j => mathematical r read i.val j.val) before final := by
  obtain ⟨final,he,hr⟩ := correct_event r c h trials read scratch before
  refine ⟨final,?_,hr⟩
  rw [Execution.source_event M realOps law r c h,he]

def read32 (heap : Nat → Byte) (p : Nat) : Word :=
  BitVec.ofNat 32 ((heap p).toNat+256*(heap (p+1)).toNat+
    65536*(heap (p+2)).toNat+16777216*(heap (p+3)).toNat)
def read16 (heap : Nat → Byte) (p : Nat) : BitVec 16 :=
  BitVec.ofNat 16 ((heap p).toNat+256*(heap (p+1)).toNat)
def halfBits (x : Word) : BitVec 16 := BitVec.ofNat 16 x.toNat
def halfByte (x : Word) (byte : Fin 2) : Byte := wordByte x byte.val

theorem half_reconstruct (x : Word) :
    BitVec.ofNat 16 ((halfByte x ⟨0,by decide⟩).toNat+
      256*(halfByte x ⟨1,by decide⟩).toNat) = halfBits x := by
  apply BitVec.eq_of_toNat_eq
  simp only [halfByte,wordByte,halfBits,BitVec.toNat_ofNat]
  norm_num
  omega

def floating (instruction : InstructionProfile) (r : Request) (heap : Nat → Byte)
    (i : Fin r.A.rows) (j : Fin r.B.columns) : Word :=
  logicalReference (fpReference instruction) r (read32 heap) i.val j.val

theorem actual_half_result (r : Request) (c : OutputTensor)
    (heap : Nat → Byte) (final : Memory Word)
    (target : Fin r.A.rows → Fin r.B.columns → Word)
    (result : TensorResult r c halfByte target heap final)
    (i : Fin r.A.rows) (j : Fin r.B.columns) :
    read16 (writeback c halfByte heap final) (outputAddress c (i.val*r.B.columns+j.val)) =
      halfBits (target i j) := by
  unfold read16
  have h0 := result.1 i j ⟨0,by decide⟩
  have h1 := result.1 i j ⟨1,by decide⟩
  simp only [Nat.add_zero] at h0
  rw [h0,h1]
  exact half_reconstruct _

/-- Intermediate complete event execution theorem. The main source linkage is
separate; this theorem never accepts a completed-kernel or desired-result premise. -/
theorem fp_equiv_event (native : PrimitiveOps) (instruction : InstructionProfile)
    (ports : LocalPorts native instruction) (r : Request) (c : OutputTensor) (h : Contract r c)
    (trials : Nat) (heap : Nat → Byte) (scratch : Memory Word) :
    ∃ final, eventRun native.operations r c trials (read32 heap) scratch = ((),final) ∧
      TensorResult r c halfByte (floating instruction r heap) heap final ∧
      (∀ i : Fin r.A.rows, ∀ j : Fin r.B.columns,
        read16 (writeback c halfByte heap final) (outputAddress c (i.val*r.B.columns+j.val)) =
          halfBits (floating instruction r heap i j)) := by
  let final := (eventRun native.operations r c trials (read32 heap) scratch).2
  have result : TensorResult r c halfByte (floating instruction r heap) heap final := by
    apply tensor_result r c h
    intro i j
    change (eventRun native.operations r c trials (read32 heap) scratch).2 _ = _
    rw [event_value native.operations r c h trials (read32 heap) scratch i.val j.val i.isLt j.isLt,
      operations_eq ports]
    rfl
  refine ⟨final,?_,result,actual_half_result r c heap final _ result⟩
  dsimp [final]

/-- The complete shared typed source computes the selected fixed-K8 native
recurrence and writes its actual FP16 bytes, with all input backing preserved. -/
theorem fp_equiv (M : FP.Structural.Algebra Word) (native : PrimitiveOps)
    (instruction : InstructionProfile) (ports : LocalPorts native instruction)
    (law : Execution.ScalarLaws M native.operations)
    (r : Request) (c : OutputTensor) (h : Contract r c) (trials : Nat)
    (heap : Nat → Byte) (scratch : Memory Word) (registers : FP.Structural.State Word) :
    ∃ final, Execution.computation M native.operations r c trials registers
        (initial r (read32 heap) scratch) = some final ∧
      TensorResult r c halfByte (floating instruction r heap) heap final ∧
      (∀ i : Fin r.A.rows, ∀ j : Fin r.B.columns,
        read16 (writeback c halfByte heap final) (outputAddress c (i.val*r.B.columns+j.val)) =
          halfBits (floating instruction r heap i j)) := by
  obtain ⟨final,he,hr,hbits⟩ := fp_equiv_event native instruction ports r c h trials heap scratch
  refine ⟨final,?_,hr,hbits⟩
  rw [Execution.source_event M native.operations law r c h,he]

/-- The raw FP32 words are the logical values of the actual resolved input
Tensor, rather than detached mathematical arrays. -/
theorem logical_input_bits (heap : Nat → Byte) (x : Matrix) (xl : MatrixLaws x)
    (resolved : x.tensor.metadata.isNeg = Bool.false) (i j : Nat) :
    logical (read32 heap) x i j =
      (VeriTile.Torch.FP32.logicalValue (fun p => ⟨read32 heap p⟩) x.tensor xl.tensor
        (i*x.columns+j)).bits := by
  simp [logical,VeriTile.Torch.FP32.logicalValue,resolved]

/-- Erase only Tensor identity/device/API flags, retaining every kernel address field. -/
def kernelBuffer (t : MatmulLeakyreluFp8.Tensor) : KernelBuffers.Buffer :=
  ⟨t.storage,t.offset,t.rowStride,t.columnStride,t.allocationBytes,t.allocationBase,t.pointer⟩

def kernelGeometry (r : Request) : KernelBuffers.Geometry :=
  ⟨r.A.rows,r.B.columns,r.A.columns,r.activation == "leaky_relu"⟩

def kernelBuffers (r : Request) (c : OutputTensor) : KernelBuffers.Buffers :=
  ⟨kernelBuffer (arguments r).A,kernelBuffer (arguments r).B,kernelBuffer (outputDescriptor r c)⟩

theorem kernel_source (r : Request) (c : OutputTensor) :
    Execution.source (kernelGeometry r).problem ((kernelBuffers r c).views (kernelGeometry r)) =
      Execution.source (arguments r).problem ⟨(arguments r).A,(arguments r).B,outputDescriptor r c⟩ := rfl

theorem kernel_layout (r : Request) (c : OutputTensor) (h : Contract r c) :
    KernelBuffers.Layout (kernelGeometry r) (kernelBuffers r c) := by
  have l := layout r c h
  refine ⟨⟨rfl,rfl⟩,⟨rfl,rfl⟩,⟨rfl,rfl⟩,rfl,rfl,rfl,?_,?_,?_,l.outputStrides,l.aSeparate,l.bSeparate⟩
  · exact ⟨l.aBound.rows,l.aBound.columns,l.aBound.rowStride,l.aBound.columnStride,l.aBound.offset,
      l.aBound.span,l.aBound.capacity,l.aBound.physical,l.aBound.pointer⟩
  · change (KernelBuffers.Buffer.view (kernelBuffer (arguments r).B) r.A.columns r.B.columns .fp32).Bounded
    rw [h.dimensions]
    exact ⟨l.bBound.rows,l.bBound.columns,l.bBound.rowStride,l.bBound.columnStride,l.bBound.offset,
      l.bBound.span,l.bBound.capacity,l.bBound.physical,l.bBound.pointer⟩
  · exact ⟨l.cBound.rows,l.cBound.columns,l.cBound.rowStride,l.cBound.columnStride,l.cBound.offset,
      l.cBound.span,l.cBound.capacity,l.cBound.physical,l.cBound.pointer⟩

#axiomsClean kernel_source
#axiomsClean kernel_layout
#axiomsClean correct
#axiomsClean fp_equiv
#axiomsClean logical_input_bits
#specNonCircular mathematical avoiding [eventCode,eventRun,writeback,logicalReference]
#specNonCircular floating avoiding [eventCode,eventRun,writeback,resultMemory]
#specNonCircular LocalPorts avoiding [eventCode,eventRun,floating,TensorResult]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.TensorBridge
