import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.Whole
import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.HostBindings
import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.NativeCorrespondence
import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.CheckedNative
import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.ByteHost
namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open VeriTile.Triton FP.Structural Host
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- The logical output plus frame determine the complete observed memory. -/
theorem outputPost_unique (x : Arguments) (initial a b : State Word) (value : Index x → Int)
    (ha : OutputPost x initial a value) (hb : OutputPost x initial b value) : a.mem=b.mem := by
  classical
  funext r o
  by_cases hr : r=x.c.region
  · subst r
    by_cases ho : ∃ i : Index x, o=Host.address x i
    · obtain ⟨i,rfl⟩ := ho
      exact (ha.1 i).trans (hb.1 i).symm
    · have hf : x.c.region≠x.c.region ∨ ∀ i : Index x, o≠Host.address x i :=
        Or.inr (fun i h => ho ⟨i,h⟩)
      exact (ha.2 _ _ hf).trans (hb.2 _ _ hf).symm
  · exact (ha.2 _ _ (Or.inl hr)).trans (hb.2 _ _ (Or.inl hr)).symm

/-- Arbitrary backend callbacks outside the selected operations, and arbitrary
fallbacks for inactive undefined casts, cannot change any returned memory. -/
theorem scaled_backend_independent (rest₁ rest₂ : Algebra Word) (cfg : FP.Config)
    (u v : Nat → Word → Int) (x : Arguments) (scale : Tensor) (initial a b : State Word)
    (h : Valid x) (hs : ScaleValid x scale)
    (ha : Int8Storage x.a initial) (hb : Int8Storage x.b initial) (hscale : FP32Storage scale initial)
    (hd : ScaledWord.LogicalDefined cfg (scaledParameters x scale) initial)
    (ea : Host.int_scaled_matmul_kernel rest₁ cfg u x scale initial=some (x.c,a))
    (eb : Host.int_scaled_matmul_kernel rest₂ cfg v x scale initial=some (x.c,b)) : a.mem=b.mem := by
  obtain ⟨ta,eta,hpa⟩ := int_scaled_matmul_correct rest₁ cfg u x scale initial h hs ha hb hscale hd
  obtain ⟨tb,etb,hpb⟩ := int_scaled_matmul_correct rest₂ cfg v x scale initial h hs ha hb hscale hd
  have hta : ta=a := (Prod.mk.inj (Option.some.inj (eta.symm.trans ea))).2
  have htb : tb=b := (Prod.mk.inj (Option.some.inj (etb.symm.trans eb))).2
  subst ta; subst tb
  exact outputPost_unique x initial a b _ hpa hpb

#axiomsClean outputPost_unique
#axiomsClean scaled_backend_independent

/-- Complete repaired source call under the three local native FP primitive
correspondences, at the pinned nearest-even/no-flush FP32 reference profile.
The active defined-conversion condition is solely about the independent
logical input product and scale. Output coverage, frame, execution success
and exact returned C identity are derived from the existing whole-call proof. -/
specification int_scaled_matmul_native_correct (rest : Algebra Word) (native : Native.Primitives)
    (numeric : Native.PrimitiveCorrespondence native)
    (x : Arguments) (scale : Tensor) (initial : State Word)
    (h : Valid x) (hs : ScaleValid x scale)
    (ha : Int8Storage x.a initial) (hb : Int8Storage x.b initial) (hscale : FP32Storage scale initial)
    (hd : ScaledWord.LogicalDefined Native.sourceConfig (scaledParameters x scale) initial) :
    ∃ final, Native.call rest native x scale initial=some (x.c,final) ∧
      OutputPost x initial final (fun i => ScaledWord.logicalResult Native.sourceConfig
        (scaledParameters x scale) initial i.1.val i.2.val) := by
  rw [Native.call_eq rest native numeric]
  exact int_scaled_matmul_correct rest Native.sourceConfig (Native.fallback native)
    x scale initial h hs ha hb hscale hd

#axiomsClean int_scaled_matmul_native_correct
end
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001

namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.FPEquiv
open VeriTile.Triton FP.Structural
open Contract TensorBinding InputAdapter FiniteExecution
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Every admitted scale word is covered. Source signed32 overflow, FP32
promotion and multiplication, and the output conversion remain explicit.
Undefined software conversions use the actual local conversion port. -/
specification int_scaled_matmul_checked_fp32 (rest : Algebra Word) (cfg : FP.Config)
    (u : Nat → Word → Int) (read : Read) (a b c scale : TensorBinding.Tensor)
    (la : Laws .int8 a) (lb : Laws .int8 b) (ls : Laws .fp32 scale)
    (p : Contract.Config) (h : InputContract Bool.true a.metadata b.metadata c.metadata scale.metadata p)
    (initial : State Word) (bound : Inputs read a b scale Bool.true initial) :
    ∃ final,CheckedHost.call Bool.true cfg u a b c scale p initial=some (c,final) ∧
      CheckedHost.OutputPost a b c initial final
        (fun i => LogicalInputs.scaledExpected cfg (u 32) read a b scale la lb ls i.1.val i.2.val) := by
  obtain ⟨final,he,metadata,values,frame⟩ := CheckedHost.correct rest Bool.true cfg u a b c scale p h initial
  refine ⟨final,he,metadata,?_,frame⟩
  intro i
  simpa only [CheckedHost.reference,↓reduceIte,
    LogicalInputs.scaled_reference cfg u read a b c scale la lb ls p h initial bound i] using values i

/-- Complete finite-index native source call, conditional only on local FP32
promotion, multiplication and defined signed32 conversion laws. No full-dot,
tile, reduction, execution or output equality is a numerical premise. -/
theorem int_scaled_matmul_checked_fp32_logical_if_local (rest : Algebra Word) (native : CheckedNative.Instructions)
    (numeric : Native.PrimitiveCorrespondence (CheckedNative.ports native)) (read : Read)
    (a b c scale : TensorBinding.Tensor) (la : Laws .int8 a) (lb : Laws .int8 b)
    (ls : Laws .fp32 scale) (p : Contract.Config)
    (h : InputContract Bool.true a.metadata b.metadata c.metadata scale.metadata p)
    (initial : State Word) (bound : Inputs read a b scale Bool.true initial) :
    ∃ final,CheckedNative.typedCall native a b c scale p initial=some (c,final) ∧
      CheckedHost.OutputPost a b c initial final (fun i =>
        LogicalInputs.scaledExpected Native.sourceConfig (fun x => (native.signed32 x).toInt)
          read a b scale la lb ls i.1.val i.2.val) :=
  CheckedNative.correct rest (CheckedNative.ports native) numeric read a b c scale la lb ls p h initial bound

/-- Complete native instruction and physical-byte result. All integer MMA
and add.u32 structure, finite indexing, stores, grid coverage, returned C
identity, storage offsets and exterior bytes are proved. Only the three
local FP32 instruction laws remain numerical assumptions. -/
specification int_scaled_matmul_checked_fp32_if_local (rest : Algebra Word)
    (native : CheckedNative.Instructions)
    (numeric : Native.PrimitiveCorrespondence (CheckedNative.ports native)) (read : Read)
    (a b c scale : TensorBinding.Tensor) (la : Laws .int8 a) (lb : Laws .int8 b)
    (lc : Laws .int32 c) (ls : Laws .fp32 scale) (p : Contract.Config)
    (h : InputContract Bool.true a.metadata b.metadata c.metadata scale.metadata p)
    (initial : State Word) (bound : Inputs read a b scale Bool.true initial)
    (outputBound : OutputBytes.Bound read c initial) :
    ∃ final bytes,ByteHost.nativeCall native a b c scale p (initial,read)=some (c,(final,bytes)) ∧
      CheckedHost.OutputPost a b c initial final (fun i =>
        LogicalInputs.scaledExpected Native.sourceConfig (fun x => (native.signed32 x).toInt)
          read a b scale la lb ls i.1.val i.2.val) ∧
      ByteHost.Post read bytes a b c (fun i =>
        LogicalInputs.scaledExpected Native.sourceConfig (fun x => (native.signed32 x).toInt)
          read a b scale la lb ls i.1.val i.2.val) := by
  obtain ⟨final,he,post⟩ := int_scaled_matmul_checked_fp32_logical_if_local rest native numeric
    read a b c scale la lb ls p h initial bound
  refine ⟨final,OutputBytes.render read c final,?_,post,?_⟩
  · rw [ByteHost.nativeCall_render native a b c scale p h read initial outputBound,he]
    rfl
  · apply ByteHost.post Bool.true a b c scale lc p h read initial final _ post
    intro i
    unfold LogicalInputs.scaledExpected
    change (BitVec.ofInt 32 (FP.IntegerConversion.totalSigned (CheckedNative.ports native).signed32 32 _)).toInt=
      FP.IntegerConversion.totalSigned (CheckedNative.ports native).signed32 32 _
    rw [←Native.signed_eq_total (CheckedNative.ports native) numeric]
    simp only [CheckedNative.ports,BitVec.ofInt_toInt]

#axiomsClean int_scaled_matmul_checked_fp32
#axiomsClean int_scaled_matmul_checked_fp32_if_local
end
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.FPEquiv
