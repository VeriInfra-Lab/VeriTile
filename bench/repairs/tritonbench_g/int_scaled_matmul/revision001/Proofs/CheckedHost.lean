import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.InputAdapter
import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.IntegerInstructions

namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.CheckedHost
open VeriTile.Triton FP.Structural
open Contract TensorBinding InputAdapter FiniteExecution
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def call (scaled : Bool) (cfg : FP.Config) (u : Nat → Word → Int)
    (a b c scale : TensorBinding.Tensor) (p : Contract.Config) (s : State Word) :=
  if Contract.checkTensorInputs scaled a.metadata b.metadata c.metadata scale.metadata p=.ok () then
    if Contract.Empty a.metadata b.metadata then some (c,s)
    else ((if scaled then IntegerInstructions.scaledGrid cfg u (launch a b p) scales s
      else IntegerInstructions.intGrid (launch a b p) s)).map (fun final => (c,final))
  else none

def Metadata (a b c : TensorBinding.Tensor) : Prop :=
  c.metadata.shape=[rows a.metadata,columns b.metadata] ∧ c.metadata.dtype=.other "int32" ∧
  c.metadata.device=a.metadata.device ∧ c.metadata.device.kind="cuda" ∧
  c.metadata.isContiguous=Bool.true ∧ c.metadata.isNeg=Bool.false

def OutputPost (a b c : TensorBinding.Tensor) (initial final : State Word) (value : Index a b → Int) : Prop :=
  Metadata a b c ∧
  (∀ i,final.mem outputC (address a b i)=Cell.mk .int (value i)) ∧
  (∀ r o,(r≠outputC ∨ ∀ i : Index a b,o≠address a b i) → final.mem r o=initial.mem r o)

def reference (scaled : Bool) (cfg : FP.Config) (u : Nat → Word → Int)
    (a b : TensorBinding.Tensor) (p : Contract.Config) (s : State Word) (i : Index a b) : Int :=
  if scaled then TotalScaled.logicalResult cfg u (scaledParameters (launch a b p) scales) s i.1.val i.2.val
  else BlockWord.logicalDot (blockParameters (launch a b p)) s i.1.val i.2.val

theorem correct (rest : Algebra Word) (scaled : Bool) (cfg : FP.Config) (u : Nat → Word → Int)
    (a b c scale : TensorBinding.Tensor) (p : Contract.Config)
    (h : InputContract scaled a.metadata b.metadata c.metadata scale.metadata p) (initial : State Word) :
    ∃ final,call scaled cfg u a b c scale p initial=some (c,final) ∧
      OutputPost a b c initial final (reference scaled cfg u a b p initial) := by
  have checked := (checkTensorInputs_iff scaled a.metadata b.metadata c.metadata scale.metadata p).mpr h
  have metadata := output_metadata scaled a b c scale p h
  by_cases empty : Empty a.metadata b.metadata
  · refine ⟨initial,?_,metadata,?_,?_⟩
    · simp [call,checked,empty]
    · intro i
      rcases empty with zero | zero
      · have hi := i.1.isLt; omega
      · have hi := i.2.isLt; omega
    · intro _ _ _; rfl
  · have env := (execution_of_nonempty scaled a.metadata b.metadata c.metadata scale.metadata p h empty).1
    have ha : (launch a b p).a≠(launch a b p).c := by simp [launch,inputA,outputC]
    have hb : (launch a b p).b≠(launch a b p).c := by simp [launch,inputB,outputC]
    cases scaled
    · obtain ⟨final,he,values,frame⟩ := FiniteGrid.int_correct rest (launch a b p) h.config env
        (int_geometry a b p h.config) ha hb initial
      refine ⟨final,?_,metadata,?_,?_⟩
      · simp [call,checked,empty,IntegerInstructions.intGrid_eq (launch a b p) h.config,
          IntegerInstructions.scaledGrid_eq cfg u (launch a b p) scales h.config,he]
      · simpa only [reference,Bool.false_eq_true,↓reduceIte,Geometry.address,BlockWord.geometry,
          blockParameters,launch,address,Nat.mul_one] using (fun i => values i trivial)
      · intro r o outside
        apply frame r o
        simpa only [Geometry.address,BlockWord.geometry,blockParameters,launch,address,Nat.mul_one] using
          outside.imp id (fun h i _ => h i)
    · have hs : scales≠(launch a b p).c := by simp [launch,scales,outputC]
      obtain ⟨final,he,values,frame⟩ := FiniteGrid.scaled_correct rest cfg u (launch a b p) scales h.config env
        (scaled_geometry a b p h.config) ha hb hs initial
      refine ⟨final,?_,metadata,?_,?_⟩
      · simp [call,checked,empty,IntegerInstructions.intGrid_eq (launch a b p) h.config,
          IntegerInstructions.scaledGrid_eq cfg u (launch a b p) scales h.config,he]
      · simpa only [reference,↓reduceIte,Geometry.address,ScaledWord.geometry,
          scaledParameters,launch,address,Nat.mul_one] using (fun i => values i trivial)
      · intro r o outside
        apply frame r o
        simpa only [Geometry.address,ScaledWord.geometry,scaledParameters,launch,address,Nat.mul_one] using
          outside.imp id (fun h i _ => h i)

theorem rejected (scaled : Bool) (cfg : FP.Config) (u : Nat → Word → Int)
    (a b c scale : TensorBinding.Tensor) (p : Contract.Config) (initial : State Word)
    (h : ¬InputContract scaled a.metadata b.metadata c.metadata scale.metadata p) :
    call scaled cfg u a b c scale p initial=none := by
  simp [call,(checkTensorInputs_iff scaled a.metadata b.metadata c.metadata scale.metadata p).not.mpr h]

#axiomsClean correct
#axiomsClean rejected
end
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.CheckedHost
