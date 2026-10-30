import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.CheckedHost

namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.LogicalInputs
open VeriTile.Triton FP.Structural
open Contract TensorBinding InputAdapter FiniteExecution
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Independent full-K integer sum of the original signed INT8 tensor values,
projected to the source's signed INT32 result. No tile schedule appears here. -/
def integerExpected (read : Read) (a b : TensorBinding.Tensor)
    (la : Laws .int8 a) (lb : Laws .int8 b) (r c : Nat) : Int :=
  (BitVec.ofInt 32 (LogicalSum.sum (fun k =>
    int8Value read a la (r*columns a.metadata+k) *
    int8Value read b lb (k*columns b.metadata+c)) (columns a.metadata))).toInt

/-- Source suffix: signed INT32 to FP32, FP32 multiplication, then the total
signed32 conversion. Undefined software casts retain the local conversion port. -/
def scaledExpected (cfg : FP.Config) (undefined : Word → Int) (read : Read)
    (a b scale : TensorBinding.Tensor) (la : Laws .int8 a) (lb : Laws .int8 b)
    (ls : Laws .fp32 scale) (r c : Nat) : Int :=
  FP.IntegerConversion.totalSigned undefined 32
    (FP.mul cfg (FP.round .fp32 cfg (integerExpected read a b la lb r c : Rat))
      (fp32Value read scale ls r))

theorem wrap_int8 (read : Read) (a : TensorBinding.Tensor) (la : Laws .int8 a) (i : Nat) :
    BlockWord.wrap 8 (int8Value read a la i)=int8Value read a la i := by
  unfold int8Value BlockWord.wrap
  split <;> simp only [BitVec.ofInt_toInt]

theorem scalar_inputs (read : Read) (scaled : Bool) (a b c scale : TensorBinding.Tensor)
    (la : Laws .int8 a) (lb : Laws .int8 b) (p : Contract.Config)
    (h : InputContract scaled a.metadata b.metadata c.metadata scale.metadata p)
    (initial : State Word) (bound : Inputs read a b scale scaled initial)
    (r col k : Nat) (hr : r<rows a.metadata) (hc : col<columns b.metadata)
    (hk : k<columns a.metadata) :
    (initial.mem inputA (r*columns a.metadata+k)).read .int=int8Value read a la (r*columns a.metadata+k) ∧
    (initial.mem inputB (k*columns b.metadata+col)).read .int=int8Value read b lb (k*columns b.metadata+col) := by
  have ma := member_a scaled a.metadata b.metadata c.metadata scale.metadata
  have mb := member_b scaled a.metadata b.metadata c.metadata scale.metadata
  have ia : r*columns a.metadata+k<a.metadata.numel := by
    rw [matrix_numel a.metadata (h.ranks _ ma)]
    exact linear_bound _ _ _ _ hr hk
  have ib : k*columns b.metadata+col<b.metadata.numel := by
    rw [matrix_numel b.metadata (h.ranks _ mb),←h.shapes.1]
    exact linear_bound _ _ _ _ hk hc
  constructor
  · rw [bound.left _ ia]
    exact direct_int8 read a la (h.contiguous _ ma) (h.resolved _ ma) _ ia
  · rw [bound.right _ ib]
    exact direct_int8 read b lb (h.contiguous _ mb) (h.resolved _ mb) _ ib

theorem integer_reference (read : Read) (scaled : Bool) (a b c scale : TensorBinding.Tensor)
    (la : Laws .int8 a) (lb : Laws .int8 b) (p : Contract.Config)
    (h : InputContract scaled a.metadata b.metadata c.metadata scale.metadata p)
    (initial : State Word) (bound : Inputs read a b scale scaled initial) (i : Index a b) :
    BlockWord.logicalDot (blockParameters (launch a b p)) initial i.1.val i.2.val=
      integerExpected read a b la lb i.1.val i.2.val := by
  unfold BlockWord.logicalDot integerExpected BlockWord.wrap
  congr 2
  apply LogicalSum.sum_congr
  intro k hk
  have values := scalar_inputs read scaled a b c scale la lb p h initial bound
    i.1.val i.2.val k i.1.isLt i.2.isLt hk
  simpa only [BlockWord.logicalProduct,blockParameters,launch,Nat.mul_one,values.1,values.2,wrap_int8]

theorem scale_input (read : Read) (a b c scale : TensorBinding.Tensor)
    (ls : Laws .fp32 scale) (p : Contract.Config)
    (h : InputContract Bool.true a.metadata b.metadata c.metadata scale.metadata p)
    (initial : State Word) (bound : Inputs read a b scale Bool.true initial)
    (r : Nat) (hr : r<rows a.metadata) :
    (initial.mem scales r).read .real=fp32Value read scale ls r := by
  have ms := member_scale a.metadata b.metadata c.metadata scale.metadata
  have hi : r<scale.metadata.numel := by
    simpa [VeriTile.Torch.TensorMetadata.numel,h.shapes.2.2 rfl] using hr
  rw [bound.scale rfl r hi]
  exact direct_fp32 read scale ls (h.contiguous _ ms) (h.resolved _ ms) r hi

theorem scaled_reference (cfg : FP.Config) (u : Nat → Word → Int) (read : Read)
    (a b c scale : TensorBinding.Tensor) (la : Laws .int8 a) (lb : Laws .int8 b)
    (ls : Laws .fp32 scale) (p : Contract.Config)
    (h : InputContract Bool.true a.metadata b.metadata c.metadata scale.metadata p)
    (initial : State Word) (bound : Inputs read a b scale Bool.true initial) (i : Index a b) :
    TotalScaled.logicalResult cfg u (scaledParameters (launch a b p) scales) initial i.1.val i.2.val=
      scaledExpected cfg (u 32) read a b scale la lb ls i.1.val i.2.val := by
  have hi := integer_reference read Bool.true a b c scale la lb p h initial bound i
  have hs := scale_input read a b c scale ls p h initial bound i.1.val i.1.isLt
  change ScaledWord.logicalDot (scaledParameters (launch a b p) scales) initial i.1.val i.2.val=_ at hi
  unfold TotalScaled.logicalResult ScaledWord.logicalScaled scaledExpected
  rw [hi]
  simp only [scaledParameters,Nat.mul_one,hs]

#specNonCircular integerExpected avoiding [CheckedHost.call,BlockWord.source,ScaledWord.source,FP.Structural.exec]
#specNonCircular scaledExpected avoiding [CheckedHost.call,BlockWord.source,ScaledWord.source,FP.Structural.exec]
#axiomsClean integer_reference
#axiomsClean scaled_reference
end
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.LogicalInputs
