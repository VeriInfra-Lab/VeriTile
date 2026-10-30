import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.LogicalInputs
import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.NativeCorrespondence
import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.SaturatingTile

namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.CheckedNative
open VeriTile.Triton FP.Structural
open Contract TensorBinding InputAdapter FiniteExecution
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Native conversion returns a 32-bit word even when the numerical cast is
undefined. This type fact is independent of the three numerical laws. -/
structure Instructions where
  promote32 : Int → Word
  multiply32 : Word → Word → Word
  signed32 : Word → BitVec 32
def ports (native : Instructions) : Native.Primitives :=
  ⟨native.promote32,native.multiply32,fun x => (native.signed32 x).toInt⟩

def suffix (native : Native.Primitives) (x : Launch) (scale : RegionName)
    (s : State Word) : Suffix x := fun carry i =>
  native.signed32 (native.multiply32 (Native.promote native (carry i))
    ((s.mem scale (FiniteIndex.row x.M x.N x.config (s.pids 0) i.1.val).toNat).read .real))
def program (native : Native.Primitives) (x : Launch) (scale : RegionName) (s : State Word) :=
  IntegerInstructions.program x Bool.true s (suffix native x scale s)
def one (native : Native.Primitives) (x : Launch) (scale : RegionName) (pid : Nat) (s : State Word) :=
  program native x scale (GridExecution.atProgram s pid)
def grid (native : Native.Primitives) (x : Launch) (scale : RegionName) (s : State Word) :=
  GridExecution.sequence (one native x scale) (List.range (FiniteIndex.count x.M x.N x.config)) s
def call (native : Native.Primitives) (a b c scale : TensorBinding.Tensor)
    (p : Contract.Config) (s : State Word) :=
  if Contract.checkTensorInputs Bool.true a.metadata b.metadata c.metadata scale.metadata p=.ok () then
    if Contract.Empty a.metadata b.metadata then some (c,s)
    else (grid native (launch a b p) scales s).map (fun final => (c,final))
  else none

def typedCall (native : Instructions) (a b c scale : TensorBinding.Tensor)
    (p : Contract.Config) (s : State Word) := call (ports native) a b c scale p s

theorem conversion_range (native : Instructions) (x : Word) :
    FP.IntegerConversion.SignedRange 32 ((ports native).signed32 x) :=
  ⟨by decide,BitVec.le_toInt _,BitVec.toInt_lt⟩

/-- All source carries have signed INT32 range, including masked lanes. -/
theorem accumulator_domain (x : Launch) (s : State Word) (n : Nat) (i : CIndex x) :
    FP.IntegerConversion.SignedRange 32 (accumulation x Bool.true s n i) := by
  rw [FiniteSource.scaled_accumulation x scales s n]
  exact Native.accumulator_domain (scaledParameters x scales) s n i

/-- The native promotion port is used for every actual carry; its total
mathematical completion outside signed32 is never selected. -/
theorem suffix_actual (native : Native.Primitives) (x : Launch) (scale : RegionName)
    (s : State Word) (n : Nat) (i : CIndex x) :
    suffix native x scale s (accumulation x Bool.true s n) i=
      native.signed32 (native.multiply32 (native.promote32 (accumulation x Bool.true s n i))
        ((s.mem scale (FiniteIndex.row x.M x.N x.config (s.pids 0) i.1.val).toNat).read .real)) := by
  simp only [suffix,Native.promote,if_pos (accumulator_domain x s n i)]

theorem suffix_eq (native : Native.Primitives) (h : Native.PrimitiveCorrespondence native)
    (x : Launch) (scale : RegionName) (s : State Word) :
    suffix native x scale s=scaledSuffix Native.sourceConfig (Native.fallback native) x scale s := by
  funext carry i
  simp only [suffix,scaledSuffix,Native.promote_eq native h,h.multiplication,
    Native.signed_eq_total native h]
  rfl

theorem program_eq (native : Native.Primitives) (h : Native.PrimitiveCorrespondence native)
    (x : Launch) (scale : RegionName) (s : State Word) (legal : ConfigContract x.config) :
    program native x scale s=scaledProgram Native.sourceConfig (Native.fallback native) x scale s := by
  simp only [program,scaledProgram,suffix_eq native h,IntegerInstructions.program_eq x legal]

/-- Finite signed-index execution and the complete native typed source have
the same observable memory. Only the three scalar numerical laws are assumed. -/
theorem program_source (rest : Algebra Word) (native : Native.Primitives)
    (h : Native.PrimitiveCorrespondence native) (x : Launch) (scale : RegionName)
    (legal : ConfigContract x.config) (env : IndexEnvelope x.M x.N x.K x.config)
    (s : State Word) (hp : s.pids 0<FiniteIndex.count x.M x.N x.config) :
    ∃ actual typed,program native x scale s=some actual ∧
      FP.Structural.exec (Native.backend rest native) (ScaledWord.source (scaledParameters x scale)) s=some typed ∧
      actual.mem=typed.mem := by
  rw [program_eq native h x scale s legal,Native.backend_eq rest native h]
  exact FiniteSource.scaled_program_source rest Native.sourceConfig (Native.fallback native) x scale legal env s hp

theorem call_eq (native : Native.Primitives) (h : Native.PrimitiveCorrespondence native)
    (a b c scale : TensorBinding.Tensor) (p : Contract.Config) (s : State Word) :
    call native a b c scale p s=
      CheckedHost.call Bool.true Native.sourceConfig (Native.fallback native) a b c scale p s := by
  have programs : one native (launch a b p) scales=
      IntegerInstructions.scaledOne Native.sourceConfig (Native.fallback native) (launch a b p) scales := by
    funext pid current
    simp only [one,program,IntegerInstructions.scaledOne,suffix_eq native h]
  simp only [call,CheckedHost.call,grid,IntegerInstructions.scaledGrid,programs,↓reduceIte]

theorem correct (rest : Algebra Word) (native : Native.Primitives)
    (numeric : Native.PrimitiveCorrespondence native) (read : Read)
    (a b c scale : TensorBinding.Tensor) (la : Laws .int8 a) (lb : Laws .int8 b)
    (ls : Laws .fp32 scale) (p : Contract.Config)
    (h : InputContract Bool.true a.metadata b.metadata c.metadata scale.metadata p)
    (initial : State Word) (bound : Inputs read a b scale Bool.true initial) :
    ∃ final,call native a b c scale p initial=some (c,final) ∧
      CheckedHost.OutputPost a b c initial final (fun i =>
        LogicalInputs.scaledExpected Native.sourceConfig native.signed32 read a b scale la lb ls i.1.val i.2.val) := by
  obtain ⟨final,he,metadata,values,frame⟩ := CheckedHost.correct rest Bool.true Native.sourceConfig
    (Native.fallback native) a b c scale p h initial
  refine ⟨final,?_,metadata,?_,frame⟩
  · rw [call_eq native numeric]; exact he
  · intro i
    simpa only [CheckedHost.reference,↓reduceIte,
      LogicalInputs.scaled_reference Native.sourceConfig (Native.fallback native) read a b c scale la lb ls p h initial bound i,
      Native.fallback] using values i

#axiomsClean suffix_actual
#axiomsClean conversion_range
#axiomsClean program_source
#axiomsClean call_eq
#axiomsClean correct
end
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.CheckedNative
