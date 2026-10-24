import bench.tritonbench_g.sgmv_expand_slice.Proofs.FPContract
import bench.tritonbench_g.sgmv_expand_slice.Proofs.FPSoftware
namespace VeriTile.Bench.TritonBenchG.SgmvExpandSlice.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 12000000
set_option maxRecDepth 35000
set_option linter.unusedSimpArgs false

def witnessParameters : Parameters :=
  ⟨"X","W","Y",⟨"start"⟩,⟨"length"⟩,⟨"adapter"⟩,3,4,6,1,12,4,1,10,1,5,2,2,2⟩

def inputCell (fmt : Storage) (w : Word) : Cell Word := match fmt with
  | .half => .mk .fp16 w | .bfloat => .mk .bf16 w

def witnessInitial (fmt : Storage) : State Word where
  mem := fun r o =>
    if r=("start":RegionName) then .mk .nat 5
    else if r=("length":RegionName) then .mk .nat 3
    else if r=("adapter":RegionName) then .mk .nat 1
    else inputCell fmt (Convert.cast .ieeeValue .single (storageFormat fmt) (FP.Value.ofBits .fp32
      (if r=("X":RegionName) then
        if o=42 then 0x3f800000 else if o=43 then 0x40000000 else if o=44 then 0x40400000
        else if o=45 then 0x40800000 else 0x40000000
       else if r=("W":RegionName) then if 20≤o ∧ o<24 then 0x40000000 else 0x3f800000
       else 0x42c60000)))
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

def lane00 : TileIndex [2,2] := (⟨0,by decide⟩,⟨0,by decide⟩,())
def resultWord : Output → Word
  | .half => FP.Value.ofBits .fp32 0x4d00
  | .bfloat => FP.Value.ofBits .fp32 0x41a0
  | .single => FP.Value.ofBits .fp32 0x41a00000

theorem witness_launch :
    0<witnessParameters.blockInner ∧ witnessParameters.inner=witnessParameters.blockInner*2 := by decide +kernel

theorem witness_value (fmt : Storage) (out : Output) :
    outputValues .ieeeValue fmt out witnessParameters
      (accumulated .ieeeValue (softwareDot .ieeeValue) fmt witnessParameters (witnessInitial fmt)) lane00 =
      resultWord out := by
  cases fmt <;> cases out <;> decide +kernel

theorem witness_layout (fmt : Storage) : Function.Injective (outputOffset witnessParameters (witnessInitial fmt)) := by
  cases fmt <;> decide +kernel

/-- Two full K blocks, both low-precision input formats and all three output
formats, nonzero PIDs/start/adapter/slice, modulo gather, and both masked axes. -/
theorem two_block_bit_success (rest : Algebra Word) (fmt : Storage) (out : Output) :
    ∃ final, FP.Structural.exec (backend rest .ieeeValue (softwareDot .ieeeValue))
      (source fmt out witnessParameters) (witnessInitial fmt)=some final ∧
      final.mem "Y" 77=outputCell out (resultWord out) ∧
      final.mem "Y" 78=(witnessInitial fmt).mem "Y" 78 ∧
      final.mem "Y" 87=(witnessInitial fmt).mem "Y" 87 ∧
      final.mem "X" 42=(witnessInitial fmt).mem "X" 42 := by
  obtain ⟨final,he,hp⟩ := correctness rest .ieeeValue (softwareDot .ieeeValue) fmt out
    witnessParameters (witnessInitial fmt) (witness_layout fmt)
  refine ⟨final,he,?_,?_,?_,hp.2 "X" 42 (Or.inl (by decide))⟩
  · have ha : active witnessParameters (witnessInitial fmt) lane00=Bool.true := by
      simp [active,row,col,seqLen,witnessParameters,witnessInitial,lane00]
    have hv := hp.1 lane00 ha
    change final.mem "Y" 77=_ at hv
    exact hv.trans (congrArg (outputCell out) (witness_value fmt out))
  · apply hp.2 "Y" 78
    right
    intro i hi
    simp [active,row,col,seqLen,outputOffset,seqStart,witnessParameters,witnessInitial] at hi ⊢
    have hrow : i.1.val=0 := by omega
    have hcol : i.2.1.val=0 := by omega
    simp [hrow,hcol]
  · apply hp.2 "Y" 87
    right
    intro i hi
    simp [active,row,col,seqLen,outputOffset,seqStart,witnessParameters,witnessInitial] at hi ⊢
    have hrow : i.1.val=0 := by omega
    have hcol : i.2.1.val=0 := by omega
    simp [hrow,hcol]
#axiomsClean witness_launch
#axiomsClean witness_value
#axiomsClean witness_layout
#axiomsClean two_block_bit_success
end
end VeriTile.Bench.TritonBenchG.SgmvExpandSlice.FPValue
