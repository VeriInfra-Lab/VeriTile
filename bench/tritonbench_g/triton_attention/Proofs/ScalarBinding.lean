import bench.tritonbench_g.triton_attention.Proofs.NativeTarget
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Native
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section

/-- The source staging uses fixed words for zero and one. Other scalar leaves
use the selected scalar conversion, independently of arithmetic instructions. -/
def scalarLiteralWord (encode : ℝ → Word) (value : ℝ) : Word :=
  if value=0 then 0 else if value=1 then 0x3f800000 else encode value

/-- Ordinary local scalar-input binding. This constrains only the source scale
leaf and the raw integer-one specialization, not an operation or kernel result.
The inherited ABI record already ties `word` to `marshal raw`. -/
structure ScalarBinding (s : Checked.ScalarABI) (encode : ℝ → Word) : Prop where
  word : scalarLiteralWord encode (s.project s.raw)=s.word
  constexprOne : s.raw=.integer 1 → s.project s.raw=1

theorem literal_word (one : Bool) (kind : Kind) (I : Prior.Instructions)
    (encode : ℝ → Word) (value : ℝ) :
    (lowerScalar one kind (literal value)).eval I encode=scalarLiteralWord encode value := by
  by_cases h0 : value=0
  · simp [literal,scalarLiteralWord,h0,lowerScalar,Graph.eval]
  · by_cases h1 : value=1
    · simp [literal,scalarLiteralWord,h1,lowerScalar,Graph.eval]
    · simp [literal,scalarLiteralWord,h0,h1,lowerScalar,Graph.eval]

/-- The very literal port used by the shared forward/backward source is bound
to the supplied raw scalar's FP32 word for every physical owner. -/
theorem ScalarBinding.source_scale_word (s : Checked.ScalarABI) (encode : ℝ → Word)
    (binding : ScalarBinding s encode) (I : Prior.Instructions) (kind : Kind) (owner : Nat) :
    (lowerScalar (constexprScale s) kind
      ((modelScalar (constexprScale s) kind).literal (some .fp32) .real
        (s.project s.raw) owner)).eval I encode=s.word := by
  change (lowerScalar (constexprScale s) kind (literal (s.project s.raw))).eval I encode=s.word
  rw [literal_word]
  exact binding.word

theorem ScalarBinding.source_scale_marshaled (s : Checked.ScalarABI) (encode : ℝ → Word)
    (binding : ScalarBinding s encode) (I : Prior.Instructions) (kind : Kind) (owner : Nat) :
    (lowerScalar (constexprScale s) kind
      ((modelScalar (constexprScale s) kind).literal (some .fp32) .real
        (s.project s.raw) owner)).eval I encode=s.marshal s.raw :=
  (binding.source_scale_word s encode I kind owner).trans s.marshalled

theorem ScalarBinding.target_scale_word (s : Checked.ScalarABI) (encode : ℝ → Word)
    (binding : ScalarBinding s encode) (I : Prior.Instructions) :
    (Target.scaleWord (s.project s.raw)).eval I encode=s.word := by
  have leaf : (Target.scaleWord (s.project s.raw)).eval I encode=
      scalarLiteralWord encode (s.project s.raw) := by
    by_cases h0 : s.project s.raw=0
    · simp [Target.scaleWord,scalarLiteralWord,h0,Graph.eval]
    · by_cases h1 : s.project s.raw=1
      · simp [Target.scaleWord,scalarLiteralWord,h1,Graph.eval]
      · simp [Target.scaleWord,scalarLiteralWord,h0,h1,Graph.eval]
  exact leaf.trans binding.word

theorem ScalarBinding.runtime_literal (s : Checked.ScalarABI) (encode : ℝ → Word)
    (binding : ScalarBinding s encode) (h0 : s.project s.raw≠0) (h1 : s.project s.raw≠1) :
    encode (s.project s.raw)=s.word := by
  simpa [scalarLiteralWord,h0,h1] using binding.word

/-- A nonvacuous constructor covering both integer-one and runtime-float-one
observations, for every unrelated literal conversion. -/
theorem one_binding (s : Checked.ScalarABI) (encode : ℝ → Word)
    (projected : s.project s.raw=1) (word : s.word=0x3f800000) : ScalarBinding s encode :=
  ⟨by simp [scalarLiteralWord,projected,word],fun _=>projected⟩

theorem ScalarBinding.constexpr_literal (s : Checked.ScalarABI) (encode : ℝ → Word)
    (binding : ScalarBinding s encode) (raw : s.raw=.integer 1) :
    literal (s.project s.raw)=.one := by
  simp [binding.constexprOne raw,literal]

theorem ScalarBinding.constexpr_word (s : Checked.ScalarABI) (encode : ℝ → Word)
    (binding : ScalarBinding s encode) (raw : s.raw=.integer 1) : s.word=0x3f800000 := by
  simpa [scalarLiteralWord,binding.constexprOne raw] using binding.word.symm

theorem ScalarBinding.constexpr_source_mul (s : Checked.ScalarABI) (encode : ℝ → Word)
    (binding : ScalarBinding s encode) (raw : s.raw=.integer 1) (kind : Kind) (a : Stage) :
    lowerScalar (constexprScale s) kind (.mul a (literal (s.project s.raw)))=
      lowerScalar Bool.true kind a := by
  rw [binding.constexpr_literal s encode raw,raw_integer_one s raw]
  rfl

theorem ScalarBinding.constexpr_source_backward_sub (s : Checked.ScalarABI) (encode : ℝ → Word)
    (binding : ScalarBinding s encode) (raw : s.raw=.integer 1) (a m : Stage) :
    lowerScalar (constexprScale s) .backward (.sub (.mul a (literal (s.project s.raw))) m)=
      .sub (lowerScalar Bool.true .backward a) (lowerScalar Bool.true .backward m) := by
  rw [binding.constexpr_literal s encode raw,raw_integer_one s raw]
  rfl

/-- Float one retains its runtime operations; equality of its word with the
one constant does not enable raw-integer specialization. -/
theorem runtime_one_source_mul (s : Checked.ScalarABI) (bits : BitVec 64)
    (raw : s.raw=.float64 bits) (projected : s.project s.raw=1) (kind : Kind) (a : Stage) :
    lowerScalar (constexprScale s) kind (.mul a (literal (s.project s.raw)))=
      .mul (.word 0x3f800000) (lowerScalar Bool.false kind a) := by
  rw [projected,raw_float_runtime s bits raw]
  simp [literal,lowerScalar]

theorem runtime_one_source_backward_fma (s : Checked.ScalarABI) (bits : BitVec 64)
    (raw : s.raw=.float64 bits) (projected : s.project s.raw=1) (a m : Stage) :
    lowerScalar (constexprScale s) .backward (.sub (.mul a (literal (s.project s.raw))) m)=
      .fma (.word 0x3f800000) (lowerScalar Bool.false .backward a)
        (.neg (lowerScalar Bool.false .backward m)) := by
  rw [projected,raw_float_runtime s bits raw]
  simp [literal,lowerScalar]

theorem loaded_one_is_not_constexpr (kind : Kind) (a : Stage) :
    lowerScalar Bool.true kind (.mul a (.ready (.word 0x3f800000)))=
      .mul (lowerScalar Bool.true kind a) (.word 0x3f800000) := rfl

/-- The inconsistent raw-one/project-two witness from the v1 review cannot
satisfy the bound public interface. -/
theorem ScalarBinding.excludes_project_two (s : Checked.ScalarABI) (encode : ℝ → Word)
    (binding : ScalarBinding s encode) (raw : s.raw=.integer 1) : s.project s.raw≠2 := by
  rw [binding.constexprOne raw]
  norm_num

#axiomsClean literal_word
#axiomsClean ScalarBinding.source_scale_word
#axiomsClean ScalarBinding.source_scale_marshaled
#axiomsClean ScalarBinding.target_scale_word
#axiomsClean ScalarBinding.runtime_literal
#axiomsClean one_binding
#axiomsClean ScalarBinding.constexpr_literal
#axiomsClean ScalarBinding.constexpr_word
#axiomsClean ScalarBinding.constexpr_source_mul
#axiomsClean ScalarBinding.constexpr_source_backward_sub
#axiomsClean runtime_one_source_mul
#axiomsClean runtime_one_source_backward_fma
#axiomsClean loaded_one_is_not_constexpr
#axiomsClean ScalarBinding.excludes_project_two
#specNonCircular ScalarBinding avoiding [Target.forward,Target.prepared,FP.Structural.exec]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Native
