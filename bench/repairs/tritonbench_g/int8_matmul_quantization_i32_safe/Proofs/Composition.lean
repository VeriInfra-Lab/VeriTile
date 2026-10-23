import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.ProducerTargets
import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.GridResults

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.Composed
open VeriTile.Triton FP.Structural KernelExecution
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false
variable {F : Type} [Inhabited F]

def geometry (s : Shape) (width : Nat) (positive : 0<width) : KernelProducerGrid.Geometry :=
  ⟨s.M,s.K,width,positive⟩

/-- The only connection is storage renaming: actual producer q/scales
become GEMM A/AS. Weight/output regions retain the actual producer frame. -/
def attach (produced : State F) : State F :=
  { produced with mem := fun region i =>
      if region="A" then produced.mem "q" i
      else if region="AS" then produced.mem "scales" i else produced.mem region i }

structure WeightReads (s : Shape) (c : Config) (columnMajor : Bool) (B : Nat → Nat → Byte)
    (scales : Nat → F) (state : State F) : Prop where
  B : ∀ k n, k<s.K → n<s.N →
    (state.mem "B" (k*(Gemm.parameters s c columnMajor).weightInnerStride+
      n*(Gemm.parameters s c columnMajor).weightColumnStride)).read .int=(B k n).toInt
  scales : ∀ n, n<s.N → (state.mem "BS" n).read .real=scales n

/-- Stored byte/half values are read from the completed producer execution;
this lemma contains only pointwise storage representation facts. -/
theorem reads_produced (s : Shape) (c : Config) (columnMajor : Bool) (width : Nat)
    (covers : s.K≤width) (state produced : State F) (A B : Nat → Nat → Byte) (as bs : Nat → F)
    (data : ∀ r : Fin s.M, ∀ k : Fin width, k.val<s.K → produced.mem "q" (r.val*s.K+k.val)=.mk .int (A r k).toInt)
    (scales : ∀ r : Fin s.M, produced.mem "scales" r=.mk .fp16 (as r))
    (weights : WeightReads s c columnMajor B bs state)
    (frame : ∀ (region : RegionName) i, region≠"q" → region≠"scales" → produced.mem region i=state.mem region i) :
    Gemm.Reads (Gemm.parameters s c columnMajor) .half ⟨A,B,as,bs⟩ (attach produced) := by
  constructor
  · intro r k hr hk
    change k < s.K at hk
    have pos : 1 ≤ s.K := by omega
    have h:=data ⟨r,hr⟩ ⟨k,by omega⟩ hk
    simpa [Gemm.parameters,attach,max_eq_left pos,Nat.mul_one,h] using congrArg (FP.Structural.Cell.read (α:=F) .int) h
  · intro k n hk hn
    simp only [Gemm.parameters,attach,show ("B":RegionName)≠"A" by decide,
      show ("B":RegionName)≠"AS" by decide,ite_false]
    rw [frame _ _ (by decide) (by decide)]
    exact weights.B k n hk hn
  · intro r hr
    have h:=scales ⟨r,hr⟩
    simp [Gemm.parameters,attach,Nat.mul_one,h]
  · intro n hn
    simp only [Gemm.parameters,attach,Nat.mul_one,show ("BS":RegionName)≠"A" by decide,
      show ("BS":RegionName)≠"AS" by decide,ite_false]
    rw [frame _ _ (by decide) (by decide)]
    exact weights.scales n hn

local instance : Inhabited Word32 := ⟨FP.Value.zero .fp32⟩
def nativeInput (f : KernelNativeModel.FloatingInstructions) (p : KernelProducerGrid.Geometry)
    (selection : Producer.Selection p) (values : Nat → Nat → Word32) (B : Nat → Nat → Byte) (bs : Nat → Word32) :
    Gemm.Input Word32 :=
  ⟨fun r k => KernelProducerTargets.nativeByte f p selection (values r) k, B,
   fun r => KernelScalarProfile.pack16 (KernelProducerTargets.nativeStoredScale f p selection (values r)), bs⟩

theorem native_producer_to_gemm (f : KernelNativeModel.FloatingInstructions) (s : Shape) (c : Config)
    (columnMajor : Bool) (width : Nat) (positive : 0<width) (covers : s.K≤width)
    (selection : Producer.Selection (geometry s width positive)) (rest : Algebra Word32)
    (values : Nat → Nat → Word32) (B : Nat → Nat → Byte) (bs : Nat → Word32) (state : State Word32)
    (reads : KernelProducerTargets.Reads (geometry s width positive) values state)
    (weights : WeightReads s c columnMajor B bs state) (order : Producer.Schedule (geometry s width positive)) :
    ∃ produced, KernelProducerGrid.run (KernelScalarProfile.native f selection.tree rest) KernelNativeModel.integer8
      (geometry s width positive) order.order state=some produced ∧
      Gemm.Reads (Gemm.parameters s c columnMajor) .half
        (nativeInput f (geometry s width positive) selection values B bs) (attach produced) := by
  obtain ⟨produced,he,hq,hs,hf⟩ := Producer.native_complete f (geometry s width positive) selection covers rest values state reads order
  refine ⟨produced,he,reads_produced s c columnMajor width covers state produced _ _ _ _ hq hs weights ?_⟩
  intro region i hq hs
  exact hf region i (Or.inl hq) (Or.inl hs)

def realInput (s : Shape) (values : Nat → Nat → ℝ) (B : Nat → Nat → Byte) (bs : Nat → ℝ) : Gemm.Input ℝ :=
  ⟨fun r k => KernelProducerTargets.mathematicalByte (values r) s.K k, B,
    fun r => KernelProducerTargets.mathematicalScale (values r) s.K, bs⟩

theorem real_producer_to_gemm (s : Shape) (c : Config) (columnMajor : Bool) (width : Nat)
    (positive : 0<width) (covers : s.K≤width) (selection : Producer.Selection (geometry s width positive))
    (rest : Algebra ℝ) (values : Nat → Nat → ℝ) (B : Nat → Nat → Byte) (bs : Nat → ℝ) (state : State ℝ)
    (reads : KernelProducerTargets.Reads (geometry s width positive) values state)
    (weights : WeightReads s c columnMajor B bs state) (order : Producer.Schedule (geometry s width positive)) :
    ∃ produced, KernelProducerGrid.run (KernelScalarProfile.mathematical selection.tree rest) realInteger8
      (geometry s width positive) order.order state=some produced ∧
      Gemm.Reads (Gemm.parameters s c columnMajor) .half (realInput s values B bs) (attach produced) := by
  obtain ⟨produced,he,hq,hs,hf⟩ := Producer.mathematical_complete (geometry s width positive) selection covers rest values state reads order
  refine ⟨produced,he,reads_produced s c columnMajor width covers state produced _ _ _ _ hq hs weights ?_⟩
  intro region i hq hs
  exact hf region i (Or.inl hq) (Or.inl hs)

/-- Independent dense Real formula over the producer's derived stored
quantized bytes/scales. The producer execution, then GEMM, is explicit. -/
theorem real_complete (s : Shape) (c : Config) (columnMajor : Bool) (width : Nat)
    (positive : 0<width) (covers : s.K≤width) (selection : Producer.Selection (geometry s width positive))
    (rest : Algebra ℝ) (plans : Plans) (output : Precision) (values : Nat → Nat → ℝ)
    (B : Nat → Nat → Byte) (bs : Nat → ℝ) (state : State ℝ)
    (reads : KernelProducerTargets.Reads (geometry s width positive) values state)
    (weights : WeightReads s c columnMajor B bs state) (order : Producer.Schedule (geometry s width positive))
    (schedule : Int8MatmulQuantization.Schedule s c) :
    ∃ produced, KernelProducerGrid.run (KernelScalarProfile.mathematical selection.tree rest) realInteger8
      (geometry s width positive) order.order state=some produced ∧
      Gemm.Reads (Gemm.parameters s c columnMajor) .half (realInput s values B bs) (attach produced) ∧
      (∀ cell : MatrixCell s,
        (Grid.launch s c (Grid.realAction selection.tree rest plans .half output s c columnMajor (realInput s values B bs))
          schedule (Grid.zeroCell output 0) (attach produced)).mem "C" (Grid.outputOffset s cell) =
            Grid.zeroCell output (target s (Gemm.realInput s (realInput s values B bs)) cell.1 cell.2)) := by
  obtain ⟨produced,he,hr⟩ := real_producer_to_gemm s c columnMajor width positive covers selection rest values B bs state reads weights order
  exact ⟨produced,he,hr,fun cell => Grid.real_complete selection.tree rest plans .half output s c columnMajor
    (realInput s values B bs) schedule (attach produced) cell⟩

theorem native_complete (f : KernelNativeModel.FloatingInstructions) (s : Shape) (c : Config)
    (columnMajor : Bool) (width : Nat) (positive : 0<width) (covers : s.K≤width)
    (selection : Producer.Selection (geometry s width positive)) (rest : Algebra Word32) (plans : Plans)
    (output : Precision) (values : Nat → Nat → Word32) (B : Nat → Nat → Byte) (bs : Nat → Word32)
    (state : State Word32) (reads : KernelProducerTargets.Reads (geometry s width positive) values state)
    (weights : WeightReads s c columnMajor B bs state) (order : Producer.Schedule (geometry s width positive))
    (schedule : Int8MatmulQuantization.Schedule s c) :
    ∃ produced, KernelProducerGrid.run (KernelScalarProfile.native f selection.tree rest) KernelNativeModel.integer8
      (geometry s width positive) order.order state=some produced ∧
      Gemm.Reads (Gemm.parameters s c columnMajor) .half
        (nativeInput f (geometry s width positive) selection values B bs) (attach produced) ∧
      (∀ cell : MatrixCell s,
        (Grid.launch s c (Grid.nativeAction f selection.tree rest plans .half output s c columnMajor
          (nativeInput f (geometry s width positive) selection values B bs)) schedule
          (Grid.zeroCell output (FP.Value.zero .fp32)) (attach produced)).mem "C" (Grid.outputOffset s cell) =
            Grid.nativeWord f selection.tree rest plans .half output s c columnMajor
              (nativeInput f (geometry s width positive) selection values B bs) schedule cell) := by
  obtain ⟨produced,he,hr⟩ := native_producer_to_gemm f s c columnMajor width positive covers selection rest values B bs state reads weights order
  exact ⟨produced,he,hr,fun cell => Grid.native_complete f selection.tree rest plans .half output s c columnMajor
    (nativeInput f (geometry s width positive) selection values B bs) schedule (attach produced) cell⟩

#axiomsClean reads_produced
#axiomsClean native_producer_to_gemm
#axiomsClean real_producer_to_gemm
#axiomsClean real_complete
#axiomsClean native_complete
#specNonCircular nativeInput avoiding [KernelProducerGrid.run, Grid.launch, KernelExecution.exec]
#specNonCircular realInput avoiding [KernelProducerGrid.run, Grid.launch, KernelExecution.exec]
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.Composed
