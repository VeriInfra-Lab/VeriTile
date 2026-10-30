import bench.repairs.tritonbench_g.int_scaled_matmul.Proofs.KernelModel
import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.ByteHost

namespace VeriTile.Bench.Repairs.IntScaledMatmul.KernelExecution
open VeriTile.Triton FP.Structural
open Revision001 Contract TensorBinding InputAdapter FiniteExecution
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Independent whole-K result, followed only in the scaled API by the source
FP32 promotion/multiplication and signed32 conversion. No execution or tiling. -/
def expected (scaled : Bool) (undefined : Word → BitVec 32) (read : Read)
    (a b scale : TensorBinding.Tensor) (la : Laws .int8 a) (lb : Laws .int8 b)
    (i : Index a b) : Int :=
  let integer := LogicalInputs.integerExpected read a b la lb i.1.val i.2.val
  if scaled then
    let raw := readFP32 read (physicalAddress .fp32 scale i.1.val)
    let rowScale := if scale.metadata.isNeg then raw.neg else raw
    (KernelModel.signed32 undefined
      (FP.mul Native.sourceConfig (FP.round .fp32 Native.sourceConfig (integer : Rat)) rowScale)).toInt
  else integer

def Post (a b : TensorBinding.Tensor) (initial final : State Word)
    (value : Index a b → Int) : Prop :=
  (∀ i,final.mem outputC (address a b i)=Cell.mk .int (value i)) ∧
  (∀ r o,(r≠outputC ∨ ∀ i : Index a b,o≠address a b i) → final.mem r o=initial.mem r o)

/-- Execute the complete shared DSL source over the prepared grid. Buffers
and metadata are already bound; assertions and wrapper objects are not run. -/
def source (rest : Algebra Word) (scaled : Bool) (undefined : Word → BitVec 32)
    (a b : TensorBinding.Tensor) (p : Contract.Config) (s : State Word) :=
  if scaled then ScaledWord.grid rest Native.sourceConfig (KernelModel.fallback undefined)
    (scaledParameters (launch a b p) scales) s
  else BlockWord.grid rest (blockParameters (launch a b p)) s

/-- Native finite-index/instruction execution. Empty prepared grids contain
no programs; nonempty grids use the proved zero-seeded MMA and add.u32 loop. -/
def native (scaled : Bool) (undefined : Word → BitVec 32)
    (a b : TensorBinding.Tensor) (p : Contract.Config) (s : State Word) :=
  if Empty a.metadata b.metadata then some s
  else if scaled then CheckedNative.grid (KernelModel.ports undefined) (launch a b p) scales s
    else IntegerInstructions.intGrid (launch a b p) s

theorem native_scaled_grid (undefined : Word → BitVec 32) (x : Launch)
    (scale : RegionName) (s : State Word) :
    CheckedNative.grid (KernelModel.ports undefined) x scale s=
      IntegerInstructions.scaledGrid Native.sourceConfig (KernelModel.fallback undefined) x scale s := by
  have one : CheckedNative.one (KernelModel.ports undefined) x scale=
      IntegerInstructions.scaledOne Native.sourceConfig (KernelModel.fallback undefined) x scale := by
    funext pid current
    simp only [CheckedNative.one,CheckedNative.program,IntegerInstructions.scaledOne,
      CheckedNative.suffix_eq (KernelModel.ports undefined) (KernelModel.correspondence undefined)]
    rfl
  simp only [CheckedNative.grid,IntegerInstructions.scaledGrid,one]

theorem reference_eq (scaled : Bool) (undefined : Word → BitVec 32) (read : Read)
    (a b c scale : TensorBinding.Tensor) (la : Laws .int8 a) (lb : Laws .int8 b)
    (ls : scaled=Bool.true → Laws .fp32 scale) (p : Contract.Config)
    (h : InputContract scaled a.metadata b.metadata c.metadata scale.metadata p)
    (initial : State Word) (bound : Inputs read a b scale scaled initial) :
    CheckedHost.reference scaled Native.sourceConfig (KernelModel.fallback undefined) a b p initial=
      expected scaled undefined read a b scale la lb := by
  funext i
  cases scaled
  · simpa only [CheckedHost.reference,expected,Bool.false_eq_true,↓reduceIte] using
      LogicalInputs.integer_reference read Bool.false a b c scale la lb p h initial bound i
  · have same := LogicalInputs.scaled_reference Native.sourceConfig (KernelModel.fallback undefined)
      read a b c scale la lb (ls rfl) p h initial bound i
    simp only [CheckedHost.reference,expected,↓reduceIte]
    rw [same]
    unfold LogicalInputs.scaledExpected fp32Value
    exact (KernelModel.conversion undefined _).symm

theorem source_correct (rest : Algebra Word) (scaled : Bool) (undefined : Word → BitVec 32)
    (read : Read) (a b c scale : TensorBinding.Tensor) (la : Laws .int8 a) (lb : Laws .int8 b)
    (ls : scaled=Bool.true → Laws .fp32 scale) (p : Contract.Config)
    (h : InputContract scaled a.metadata b.metadata c.metadata scale.metadata p)
    (initial : State Word) (bound : Inputs read a b scale scaled initial) :
    ∃ final,source rest scaled undefined a b p initial=some final ∧
      Post a b initial final (expected scaled undefined read a b scale la lb) := by
  rw [←reference_eq scaled undefined read a b c scale la lb ls p h initial bound]
  have ha : (launch a b p).a≠(launch a b p).c := by simp [launch,inputA,outputC]
  have hb : (launch a b p).b≠(launch a b p).c := by simp [launch,inputB,outputC]
  cases scaled
  · obtain ⟨final,he,values,frame⟩ := BlockWord.grid_correct rest (blockParameters (launch a b p))
      (int_geometry a b p h.config) (Contract.positive_tiles p h.config).2.2.1 ha hb initial
    refine ⟨final,he,?_,?_⟩
    · simpa only [CheckedHost.reference,Bool.false_eq_true,↓reduceIte,Geometry.address,
        BlockWord.geometry,blockParameters,launch,address,Nat.mul_one] using (fun i => values i trivial)
    · intro r o outside
      apply frame r o
      simpa only [Geometry.address,BlockWord.geometry,blockParameters,launch,address,Nat.mul_one] using
        outside.imp id (fun h i _ => h i)
  · have hs : scales≠(launch a b p).c := by simp [launch,scales,outputC]
    obtain ⟨final,he,values,frame⟩ := TotalScaled.grid_correct rest Native.sourceConfig
      (KernelModel.fallback undefined) (scaledParameters (launch a b p) scales)
      (scaled_geometry a b p h.config) (FiniteGrid.even_exact (launch a b p) scales h.config)
      (LogicalSum.ceil_cover _ _ (Contract.positive_tiles p h.config).2.2.1) ha hb hs initial
    refine ⟨final,he,?_,?_⟩
    · simpa only [CheckedHost.reference,↓reduceIte,Geometry.address,ScaledWord.geometry,
        scaledParameters,launch,address,Nat.mul_one] using (fun i => values i trivial)
    · intro r o outside
      apply frame r o
      simpa only [Geometry.address,ScaledWord.geometry,scaledParameters,launch,address,Nat.mul_one] using
        outside.imp id (fun h i _ => h i)

theorem native_correct (rest : Algebra Word) (scaled : Bool) (undefined : Word → BitVec 32)
    (read : Read) (a b c scale : TensorBinding.Tensor) (la : Laws .int8 a) (lb : Laws .int8 b)
    (ls : scaled=Bool.true → Laws .fp32 scale) (p : Contract.Config)
    (h : InputContract scaled a.metadata b.metadata c.metadata scale.metadata p)
    (initial : State Word) (bound : Inputs read a b scale scaled initial) :
    ∃ final,native scaled undefined a b p initial=some final ∧
      Post a b initial final (expected scaled undefined read a b scale la lb) := by
  have checked := (checkTensorInputs_iff scaled a.metadata b.metadata c.metadata scale.metadata p).mpr h
  obtain ⟨final,he,metadata,values,frame⟩ := CheckedHost.correct rest scaled Native.sourceConfig
    (KernelModel.fallback undefined) a b c scale p h initial
  have execution : (native scaled undefined a b p initial).map (fun final => (c,final))=some (c,final) := by
    by_cases empty : Empty a.metadata b.metadata
    · simpa only [CheckedHost.call,native,if_pos checked,if_pos empty,Option.map_some] using he
    · simpa only [CheckedHost.call,native,if_pos checked,if_neg empty,native_scaled_grid] using he
  have run : native scaled undefined a b p initial=some final := by
    cases found : native scaled undefined a b p initial with
    | none => simp [found] at execution
    | some value =>
      simp only [found,Option.map_some,Option.some.injEq,Prod.mk.injEq,true_and] at execution
      simpa only [execution] using found
  refine ⟨final,run,?_,frame⟩
  simpa only [reference_eq scaled undefined read a b c scale la lb ls p h initial bound] using values

theorem post_unique (a b : TensorBinding.Tensor) (initial left right : State Word)
    (value : Index a b → Int) (hl : Post a b initial left value) (hr : Post a b initial right value) :
    left.mem=right.mem := by
  classical
  funext region offset
  by_cases region_eq : region=outputC
  · subst region
    by_cases inside : ∃ i : Index a b,offset=address a b i
    · obtain ⟨i,rfl⟩ := inside
      exact (hl.1 i).trans (hr.1 i).symm
    · have outside : outputC≠outputC ∨ ∀ i : Index a b,offset≠address a b i :=
        Or.inr (fun i h => inside ⟨i,h⟩)
      exact (hl.2 _ _ outside).trans (hr.2 _ _ outside).symm
  · exact (hl.2 _ _ (Or.inl region_eq)).trans (hr.2 _ _ (Or.inl region_eq)).symm

theorem equivalent (rest : Algebra Word) (scaled : Bool) (undefined : Word → BitVec 32)
    (read : Read) (a b c scale : TensorBinding.Tensor) (la : Laws .int8 a) (lb : Laws .int8 b)
    (ls : scaled=Bool.true → Laws .fp32 scale) (p : Contract.Config)
    (h : InputContract scaled a.metadata b.metadata c.metadata scale.metadata p)
    (initial : State Word) (bound : Inputs read a b scale scaled initial) :
    ∃ original lowered,source rest scaled undefined a b p initial=some original ∧
      native scaled undefined a b p initial=some lowered ∧ original.mem=lowered.mem ∧
      Post a b initial original (expected scaled undefined read a b scale la lb) := by
  obtain ⟨original,hs,post⟩ := source_correct rest scaled undefined read a b c scale la lb ls p h initial bound
  obtain ⟨lowered,hn,npost⟩ := native_correct rest scaled undefined read a b c scale la lb ls p h initial bound
  exact ⟨original,lowered,hs,hn,post_unique a b initial original lowered _ post npost,post⟩

#axiomsClean reference_eq
#axiomsClean source_correct
#axiomsClean native_correct
#axiomsClean equivalent
#specNonCircular expected avoiding [source,native,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.IntScaledMatmul.KernelExecution
