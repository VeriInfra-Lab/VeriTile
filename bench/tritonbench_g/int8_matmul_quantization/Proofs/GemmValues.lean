import bench.tritonbench_g.int8_matmul_quantization.Source
import bench.tritonbench_g.int8_matmul_quantization.Proofs.KernelGemmEpilogue
import bench.tritonbench_g.int8_matmul_quantization.Proofs.IntegerBounds

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.Gemm
open VeriTile.Triton FP.Structural KernelExecution
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false
variable {I F H : Type} [Inhabited F]
abbrev Parameters := KernelSource.GemmParameters

/-- Original input words only. Row scales retain their declared storage
representation until the explicit source widening below. -/
structure Input (F : Type) where
  A : Nat → Nat → Byte
  B : Nat → Nat → Byte
  aScale : Nat → F
  bScale : Nat → F

/-- Local reads at the caller's actual strides. No output, accumulator,
successful execution, or result equation appears in this input relation. -/
structure Reads (p : Parameters) (rowScale : Precision) (input : Input F) (s : State F) : Prop where
  A : ∀ r k, r < p.rows → k < p.inner →
    (s.mem p.input (r*p.inputRowStride+k*p.inputInnerStride)).read .int = (input.A r k).toInt
  B : ∀ k n, k < p.inner → n < p.columns →
    (s.mem p.weight (k*p.weightInnerStride+n*p.weightColumnStride)).read .int = (input.B k n).toInt
  aScale : ∀ r, r < p.rows →
    match rowScale with
    | .half => (s.mem p.inputScale (r*p.inputScaleStride)).read .fp16 = input.aScale r
    | .single => (s.mem p.inputScale (r*p.inputScaleStride)).read .real = input.aScale r
  bScale : ∀ n, n < p.columns →
    (s.mem p.weightScale (n*p.weightScaleStride)).read .real = input.bScale n

def inner (p : Parameters) (t z lane : Nat) := t*(p.blockInner*p.split)+z*p.blockInner+lane

/-- Independent ordered native integer graph, selected only by source
coordinates. Both fused carry and nonfused block plans remain explicit. -/
def accumulator (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (plans : Plans) (p : Parameters) (input : Input F) (r n z : Nat) : Nat → Int
  | 0 => 0
  | t+1 => carrier.decode ((plans p.blockInner r n z t).eval ops
      (fun lane => if inner p t z lane < p.inner then input.A r (inner p t z lane) else 0)
      (fun lane => if inner p t z lane < p.inner then input.B (inner p t z lane) n else 0)
      (carrier.encode (accumulator carrier ops plans p input r n z t)))

def rowWord (M : Algebra F) (kind : Precision) (input : Input F) (r : Nat) : F :=
  match kind with | .half => input.aScale r | .single => M.fp32Load (input.aScale r)

/-- Exact source precision: signed accumulator -> FP32; row scale then
column scale; mandatory FP16 partial. No real-arithmetic rewrite is used. -/
def partialWord (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (plans : Plans) (rowScale : Precision) (p : Parameters) (input : Input F) (r n z : Nat) : F :=
  M.cast none .real .fp16 (M.binary (some .fp32) .real .mul
    (M.binary (some .fp32) .real .mul
      (M.fromInt (some .fp32) (accumulator carrier ops plans p input r n z p.steps))
      (M.cast (some .fp32) (KernelGemm.rowScaleType rowScale) .real (rowWord M rowScale input r)))
    (M.fp32Load (input.bScale n)))

theorem input_row (p : Parameters) (s : State F) (i : KernelGemm.CIndex p)
    (active : KernelGemm.active p s i) :
    KernelGemm.inputRow p s i.1 = KernelGemm.rowBlock p s*p.blockRows+i.1.val := by
  exact Nat.mod_eq_of_lt active.1
theorem input_column (p : Parameters) (s : State F) (i : KernelGemm.CIndex p)
    (active : KernelGemm.active p s i) :
    KernelGemm.inputColumn p s i.2.1 = KernelGemm.columnBlock p s*p.blockColumns+i.2.1.val := by
  exact Nat.mod_eq_of_lt active.2

theorem a_operand (p : Parameters) (rowScale : Precision) (input : Input F) (s : State F)
    (reads : Reads p rowScale input s) (i : KernelGemm.CIndex p) (active : KernelGemm.active p s i)
    (t : Nat) (lane : Fin p.blockInner) :
    BitVec.ofInt 8 (KernelGemm.aValue p s t (i.1,lane,PUnit.unit)) =
      if inner p t (s.pids 1) lane < p.inner then
        input.A (KernelGemm.rowBlock p s*p.blockRows+i.1.val) (inner p t (s.pids 1) lane) else 0 := by
  have mask : (s.pids 1*p.blockInner+lane.val < p.inner-t*(p.blockInner*p.split)) ↔
      inner p t (s.pids 1) lane < p.inner := by unfold inner; omega
  unfold KernelGemm.aValue KernelGemm.inputInner
  simp only [Prod.fst, Prod.snd, mask]
  split_ifs with hk
  · have addr : KernelGemm.aOffset p s t (i.1,lane,PUnit.unit) =
        (KernelGemm.rowBlock p s*p.blockRows+i.1.val)*p.inputRowStride+
          inner p t (s.pids 1) lane*p.inputInnerStride := by
      simp only [KernelGemm.aOffset, input_row p s i active, KernelGemm.inputInner, inner]
      ring
    rw [addr, reads.A _ _ active.1 hk, BitVec.ofInt_toInt]
  · rfl

theorem b_operand (p : Parameters) (rowScale : Precision) (input : Input F) (s : State F)
    (reads : Reads p rowScale input s) (i : KernelGemm.CIndex p) (active : KernelGemm.active p s i)
    (t : Nat) (lane : Fin p.blockInner) :
    BitVec.ofInt 8 (KernelGemm.bValue p s t (lane,i.2.1,PUnit.unit)) =
      if inner p t (s.pids 1) lane < p.inner then
        input.B (inner p t (s.pids 1) lane) (KernelGemm.columnBlock p s*p.blockColumns+i.2.1.val) else 0 := by
  have mask : (s.pids 1*p.blockInner+lane.val < p.inner-t*(p.blockInner*p.split)) ↔
      inner p t (s.pids 1) lane < p.inner := by unfold inner; omega
  unfold KernelGemm.bValue KernelGemm.inputInner
  simp only [Prod.fst, Prod.snd, mask]
  split_ifs with hk
  · have addr : KernelGemm.bOffset p s t (lane,i.2.1,PUnit.unit) =
        inner p t (s.pids 1) lane*p.weightInnerStride+
          (KernelGemm.columnBlock p s*p.blockColumns+i.2.1.val)*p.weightColumnStride := by
      simp only [KernelGemm.bOffset, input_column p s i active, KernelGemm.inputInner, inner]
      ring
    rw [addr, reads.B _ _ hk active.2, BitVec.ofInt_toInt]
  · rfl

theorem accumulator_source (M : Algebra F) (carrier : IntegerCarrier I)
    (ops : Int8DequantMatmul.Ops I F H) (plans : Plans)
    (p : Parameters) (rowScale : Precision) (input : Input F) (s : State F)
    (reads : Reads p rowScale input s) (i : KernelGemm.CIndex p) (active : KernelGemm.active p s i) (t : Nat) :
    KernelGemm.accumulation carrier ops plans p s t i =
      accumulator carrier ops plans p input (KernelGemm.rowBlock p s*p.blockRows+i.1.val)
        (KernelGemm.columnBlock p s*p.blockColumns+i.2.1.val) (s.pids 1) t := by
  induction t with
  | zero => rfl
  | succ t ih =>
    simp only [KernelGemm.accumulation, accumulator, ih,
      a_operand p rowScale input s reads i active, b_operand p rowScale input s reads i active]

theorem partial_source (M : Algebra F) (carrier : IntegerCarrier I)
    (ops : Int8DequantMatmul.Ops I F H) (plans : Plans)
    (p : Parameters) (rowScale : Precision) (input : Input F) (s : State F)
    (reads : Reads p rowScale input s) (i : KernelGemm.CIndex p) (active : KernelGemm.active p s i) :
    KernelGemm.partialResult M carrier ops plans rowScale p s i =
      partialWord M carrier ops plans rowScale p input (KernelGemm.rowBlock p s*p.blockRows+i.1.val)
        (KernelGemm.columnBlock p s*p.blockColumns+i.2.1.val) (s.pids 1) := by
  unfold KernelGemm.partialResult partialWord
  rw [accumulator_source M carrier ops plans p rowScale input s reads i active]
  have ha : KernelGemm.aScale M rowScale p s i.1 =
      rowWord M rowScale input (KernelGemm.rowBlock p s*p.blockRows+i.1.val) := by
    cases rowScale <;> simp [KernelGemm.aScale, rowWord, input_row p s i active, active.1,
      reads.aScale _ active.1, KernelGemm.rowScaleType] at *
  have hb : KernelGemm.bScale M p s i.2.1 =
      M.fp32Load (input.bScale (KernelGemm.columnBlock p s*p.blockColumns+i.2.1.val)) := by
    simp [KernelGemm.bScale, input_column p s i active, active.2, reads.bScale _ active.2]
  rw [ha, hb]

#axiomsClean a_operand
#axiomsClean b_operand
#axiomsClean accumulator_source
#axiomsClean partial_source
#specNonCircular partialWord avoiding [KernelGemm.partialResult, KernelGemm.accumulation, KernelExecution.exec]
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.Gemm
