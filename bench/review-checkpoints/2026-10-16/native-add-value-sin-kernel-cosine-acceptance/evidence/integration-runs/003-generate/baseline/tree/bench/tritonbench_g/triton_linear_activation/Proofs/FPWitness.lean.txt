import bench.tritonbench_g.triton_linear_activation.Proofs.FPContract
import bench.tritonbench_g.triton_linear_activation.Proofs.FPSoftware
namespace VeriTile.Bench.TritonBenchG.TritonLinearActivation.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 16000000
set_option maxRecDepth 35000

def witnessParameters (name : String) : Parameters :=
  ⟨"C","Saved","X","W","Bias",4,4,6,1,7,1,6,1,6,1,2,2,2,2,2,Bool.true,Bool.true,name⟩
def witnessInitial (fmt : Format) (cancel : Bool := Bool.false) : State Word where
  mem := fun r o => outputCell fmt (castWord .ieeeValue .single fmt (FP.Value.ofBits .fp32
    (if r=("X":RegionName) then
      if o%6=0 then 0x3f800000 else if o%6=1 then 0x40000000 else if o%6=2 then 0x40400000 else 0x40800000
    else if r=("W":RegionName) then 0x40000000
    else if r=("Bias":RegionName) then if cancel then 0xc1a00000 else 0x3f800000
    else 0x42c60000)))
  regs := fun _ _ _ => none
  pids := fun _ => 3
  numPids := fun _ => 4
  undef := fun d _ _ => defaultValue d

def lane00 : TileIndex [2,2] := (⟨0,by decide⟩,⟨0,by decide⟩,())
def resultWord : Format → Word
  | .half => FP.Value.ofBits .fp32 0x4d40
  | .bfloat => FP.Value.ofBits .fp32 0x41a8
  | .single => FP.Value.ofBits .fp32 0x41a80000

theorem witness_outputOffset (fmt : Format) (cancel : Bool) (name : String) (i : TileIndex [2,2]) :
    outputOffset (witnessParameters name) (witnessInitial fmt cancel) i=14+i.1.val*6+i.2.1.val := by
  have hi := i.1.isLt
  have hj := i.2.1.isLt
  simp [outputOffset,row,col,blockRow,blockCol,groupIdx,groupSize,width,gridRows,gridCols,
    witnessParameters,witnessInitial]
  omega

theorem witness_savedOffset (fmt : Format) (cancel : Bool) (name : String) (i : TileIndex [2,2]) :
    savedOffset (witnessParameters name) (witnessInitial fmt cancel) i=16+i.1.val*7+i.2.1.val := by
  have hi := i.1.isLt
  have hj := i.2.1.isLt
  simp [savedOffset,row,col,blockRow,blockCol,groupIdx,groupSize,width,gridRows,gridCols,
    witnessParameters,witnessInitial]
  omega

theorem witness_active (fmt : Format) (cancel : Bool) (name : String) :
    active (witnessParameters name) (witnessInitial fmt cancel) lane00=Bool.true := by
  simp [active,row,col,blockRow,blockCol,groupIdx,groupSize,width,gridRows,gridCols,
    witnessParameters,witnessInitial,lane00]

private theorem witness_injective6 : Function.Injective (fun i : TileIndex [2,2] => 14+i.1.val*6+i.2.1.val) := by decide +kernel
private theorem witness_injective7 : Function.Injective (fun i : TileIndex [2,2] => 16+i.1.val*7+i.2.1.val) := by decide +kernel

theorem witness_layout (fmt : Format) (cancel : Bool) (name : String) :
    Function.Injective (outputOffset (witnessParameters name) (witnessInitial fmt cancel)) ∧
    Function.Injective (savedOffset (witnessParameters name) (witnessInitial fmt cancel)) := by
  constructor
  · have hh : outputOffset (witnessParameters name) (witnessInitial fmt cancel)=
        fun i : TileIndex [2,2] => 14+i.1.val*6+i.2.1.val := funext (witness_outputOffset fmt cancel name)
    rw [hh]
    exact witness_injective6
  · have hh : savedOffset (witnessParameters name) (witnessInitial fmt cancel)=
        fun i : TileIndex [2,2] => 16+i.1.val*7+i.2.1.val := funext (witness_savedOffset fmt cancel name)
    rw [hh]
    exact witness_injective7

theorem witness_accumulated (g : Graph) (fmt : Format) (name : String) :
    accumulated .ieeeValue (softwareDot .ieeeValue) (softwareDotAcc .ieeeValue) g fmt
      (witnessParameters name) (witnessInitial fmt) lane00=FP.Value.ofBits .fp32 0x41a80000 := by
  change accumulated .ieeeValue (softwareDot .ieeeValue) (softwareDotAcc .ieeeValue) g fmt
    (witnessParameters "") (witnessInitial fmt) lane00=_
  cases g <;> cases fmt <;> decide +kernel

theorem witness_output (g : Graph) (fmt : Format) (relu : Bool) :
    castWord .ieeeValue .single fmt (activate .ieeeValue g zeroUnary zeroUnary (FP.div .ieeeValue)
      (if relu then "relu" else "identity") (FP.Value.ofBits .fp32 0x41a80000))=resultWord fmt := by
  cases g <;> cases fmt <;> cases relu <;> decide +kernel

theorem witness_saved (fmt : Format) :
    castWord .ieeeValue .single fmt (FP.Value.ofBits .fp32 0x41a80000)=resultWord fmt := by
  cases fmt <;> decide +kernel

/-- Two full K blocks and four active lanes, a nonzero grouped program ID,
nonunit major strides, physical loads, both stores and two concrete graphs. -/
theorem two_block_bit_success (rest : Algebra Word) (g : Graph) (fmt saved : Format) (relu : Bool) :
    ∃ final, FP.Structural.exec
      (backend rest .ieeeValue (softwareDot .ieeeValue) (softwareDotAcc .ieeeValue)
        zeroUnary zeroUnary (FP.div .ieeeValue))
      (source g fmt saved (witnessParameters (if relu then "relu" else "identity")))
      (witnessInitial fmt)=some final ∧
      final.mem "C" 14=outputCell fmt (resultWord fmt) ∧
      final.mem "Saved" 16=outputCell saved (resultWord saved) ∧
      final.mem "C" 0=(witnessInitial fmt).mem "C" 0 ∧
      final.mem "Saved" 0=(witnessInitial fmt).mem "Saved" 0 ∧
      final.mem "X" 12=(witnessInitial fmt).mem "X" 12 := by
  let name := if relu then "relu" else "identity"
  obtain ⟨final,he,hp⟩ := correctness rest .ieeeValue (softwareDot .ieeeValue)
    (softwareDotAcc .ieeeValue) zeroUnary zeroUnary (FP.div .ieeeValue) g fmt saved
    (witnessParameters name) (witnessInitial fmt) (witness_layout fmt Bool.false name).1
  refine ⟨final,he,?_,?_,?_,?_,?_⟩
  · have hv := hp.1 lane00 (witness_active fmt Bool.false name)
    rw [witness_outputOffset] at hv
    change final.mem "C" 14=_ at hv
    rw [witness_accumulated] at hv
    exact hv.trans (congrArg (outputCell fmt) (witness_output g fmt relu))
  · have hv := hp.2.1 rfl (by simp [witnessParameters]) (witness_layout fmt Bool.false name).2 lane00
    rw [witness_savedOffset] at hv
    change final.mem "Saved" 16=_ at hv
    rw [witness_accumulated,witness_saved] at hv
    exact hv
  · apply hp.2.2 "C" 0 _ (Or.inr (Or.inl (by simp [witnessParameters])))
    right
    intro i _
    rw [witness_outputOffset]
    omega
  · apply hp.2.2 "Saved" 0 (Or.inl (by simp [witnessParameters]))
    right; right
    intro i
    rw [witness_savedOffset]
    omega
  · exact hp.2.2 "X" 12 (Or.inl (by simp [witnessParameters])) (Or.inr (Or.inl (by simp [witnessParameters])))

inductive Activation where | identity | relu | tanh | gelu | fastGelu deriving DecidableEq
def Activation.name : Activation → String
  | .identity => "identity" | .relu => "relu" | .tanh => "tanh"
  | .gelu => "gelu" | .fastGelu => "fast_gelu"

theorem cancellation_value (g : Graph) (a : Activation) :
    castWord .ieeeValue .single .half (activate .ieeeValue g zeroUnary zeroUnary (FP.div .ieeeValue)
      a.name (accumulated .ieeeValue (softwareDot .ieeeValue) (softwareDotAcc .ieeeValue) g .half
        (witnessParameters a.name) (witnessInitial .half Bool.true) lane00))=FP.Value.zero .fp32 := by
  cases g <;> cases a <;> decide +kernel

/-- Each activation is executed after a nonempty two-block cancellation.
The tanh/erf auxiliary callbacks are defined only as witness software choices;
this theorem is not a statement about native libdevice. -/
theorem all_activation_bit_success (rest : Algebra Word) (g : Graph) (a : Activation) :
    ∃ final, FP.Structural.exec
      (backend rest .ieeeValue (softwareDot .ieeeValue) (softwareDotAcc .ieeeValue)
        zeroUnary zeroUnary (FP.div .ieeeValue))
      (source g .half .half (witnessParameters a.name)) (witnessInitial .half Bool.true)=some final ∧
      final.mem "C" 14=outputCell .half (FP.Value.zero .fp32) ∧
      final.mem "X" 12=(witnessInitial .half Bool.true).mem "X" 12 := by
  obtain ⟨final,he,hp⟩ := correctness rest .ieeeValue (softwareDot .ieeeValue)
    (softwareDotAcc .ieeeValue) zeroUnary zeroUnary (FP.div .ieeeValue) g .half .half
    (witnessParameters a.name) (witnessInitial .half Bool.true) (witness_layout .half Bool.true a.name).1
  refine ⟨final,he,?_,hp.2.2 "X" 12 (Or.inl (by simp [witnessParameters])) (Or.inr (Or.inl (by simp [witnessParameters])))⟩
  have hv := hp.1 lane00 (witness_active .half Bool.true a.name)
  rw [witness_outputOffset] at hv
  change final.mem "C" 14=_ at hv
  exact hv.trans (congrArg (outputCell .half) (cancellation_value g a))

#axiomsClean witness_outputOffset
#axiomsClean witness_savedOffset
#axiomsClean witness_active
#axiomsClean witness_layout
#axiomsClean witness_accumulated
#axiomsClean witness_output
#axiomsClean witness_saved
#axiomsClean two_block_bit_success
#axiomsClean cancellation_value
#axiomsClean all_activation_bit_success
end
end VeriTile.Bench.TritonBenchG.TritonLinearActivation.FPValue
