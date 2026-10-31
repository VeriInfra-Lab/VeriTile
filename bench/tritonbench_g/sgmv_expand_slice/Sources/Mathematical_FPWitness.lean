import bench.tritonbench_g.sgmv_expand_slice.Sources.Mathematical_FPContract
import bench.tritonbench_g.sgmv_expand_slice.Sources.Mathematical_FPSoftware
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

#axiomsClean witness_launch
#axiomsClean witness_value
#axiomsClean witness_layout
end
end VeriTile.Bench.TritonBenchG.SgmvExpandSlice.FPValue
