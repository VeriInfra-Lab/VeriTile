import bench.tritonbench_g.sgmv_expand_slice.Sources.Mathematical_FPFusedSource
import bench.tritonbench_g.sgmv_expand_slice.Sources.Mathematical_FPWitness
namespace VeriTile.Bench.TritonBenchG.SgmvExpandSlice.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 12000000
set_option maxRecDepth 35000
set_option linter.unusedSimpArgs false

/-- A fully specified software dot-accumulator: widening the actual operand
format, increasing k, one fp32 FMA rounding per contribution, starting at the
provided accumulator. Native MMA order/precision is still a distinct binding. -/
def sequentialDotAcc (cfg : FP.Config) (left right : Convert.Format) :
    (batch : TileShape) → {m k n : Nat} →
    (TileIndex (batch++[m,k]) → Word) → (TileIndex (batch++[k,n]) → Word) →
    (TileIndex (batch++[m,n]) → Word) → (TileIndex (batch++[m,n]) → Word)
  | [], _, k, _, a, b, c => fun i => (List.finRange k).foldl
      (fun acc j => FP.fma cfg (Convert.cast cfg left .single (a (i.1,j,())))
        (Convert.cast cfg right .single (b (j,i.2.1,()))) acc) (c i)
  | _::tail, _, _, _, a, b, c => fun i => sequentialDotAcc cfg left right tail
      (fun j => a (i.1,j)) (fun j => b (i.1,j)) (fun j => c (i.1,j)) i.2

def softwareDotAcc (cfg : FP.Config) : DotAcc where
  apply := fun _ left right output _ {batch} {_} {_} {_} a b c =>
    match LoraExpandGemv.FPValue.formatOf left,LoraExpandGemv.FPValue.formatOf right,
        LoraExpandGemv.FPValue.formatOf output with
    | some l,some r,some o => fun i => Convert.cast cfg .single o
        (sequentialDotAcc cfg l r batch a b (fun j => Convert.cast cfg o .single (c j)) i)
    | _,_,_ => fun _ => FP.Value.qNaN .fp32

def counterLeft : TileIndex [1,2] → Word := fun i =>
  FP.Value.ofBits .fp32 (if i.2.1.val=0 then 0x3c00 else 0xbc00)
def counterRight : TileIndex [2,1] → Word := fun _ => FP.Value.ofBits .fp32 0x3c00
def counterCarry : TileIndex [1,1] → Word := fun _ => FP.Value.ofBits .fp32 0x4b800000
def counterLane : TileIndex [1,1] := (⟨0,by decide⟩,⟨0,by decide⟩,())

/-- A concrete refutation of an unconditional dot+add = dotAcc law for these
software profiles. This is not a claim that a particular GPU returns either
pattern: native MMA must be bound independently to its compiled accumulator. -/
theorem no_unconditional_dot_add_dotacc_law :
    FP.add .ieeeValue (counterCarry counterLane)
      ((softwareDot .ieeeValue).apply (some .fp32) .fp16 .fp16 .fp32 {} (batch := [])
        counterLeft counterRight counterLane)=FP.Value.ofBits .fp32 0x4b800000 ∧
    (softwareDotAcc .ieeeValue).apply (some .fp32) .fp16 .fp16 .real {} (batch := [])
      counterLeft counterRight counterCarry counterLane=FP.Value.ofBits .fp32 0x4b7fffff := by
  decide +kernel

theorem fused_witness_value (fmt : Storage) (out : Output) :
    outputValues .ieeeValue fmt out witnessParameters
      (fusedAccumulated (softwareDotAcc .ieeeValue) fmt witnessParameters (witnessInitial fmt)) lane00 =
      resultWord out := by
  cases fmt <;> cases out <;> decide +kernel

#axiomsClean no_unconditional_dot_add_dotacc_law
#axiomsClean fused_witness_value
end
end VeriTile.Bench.TritonBenchG.SgmvExpandSlice.FPValue
