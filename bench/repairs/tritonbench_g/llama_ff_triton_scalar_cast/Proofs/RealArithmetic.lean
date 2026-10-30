import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.Arithmetic
import bench.repairs.tritonbench_g.llama_ff_triton.Proofs.Real

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.Execution
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 6000000

/-- Explicit ideal interpretation of every byte pattern. Finite E5M2 values
use their exact rational decoding. Exceptional patterns map to zero only in
this mathematical interpretation; the native interpretation keeps all bits. -/
def decodeReal (b : Float8E5Bits) : ℝ := (b.decodeRat.getD 0 : ℚ)
def realArithmetic : Arithmetic ℝ where
  literal := fun _ _ r => r
  binary := fun _ _ op x y => match op with
    | .add => x+y | .sub => x-y | .mul => x*y | .div => x/y
    | .max => max x y | .pow => Real.rpow x y
  unary := fun _ op x => match op with
    | .rsqrt => 1/Real.sqrt x
    | .sigmoid => 1/(1+Real.exp (-x))
    | .neg _ => -x
    | _ => 0
  cast := fun _ _ _ x => x
  fromInt := fun _ n => n
  powi := fun _ x n => x^n.toNat
  decode := fun _ b => decodeReal b
  mma16 := LlamaFF.realOps.mma16

theorem decodeReal_zero : decodeReal (Float8E5Bits.ofInt 0)=0 := by
  norm_num [decodeReal,Float8E5Bits.ofInt,Float8E5Bits.decodeRat,
    Float8E5Bits.exponent,Float8E5Bits.fraction,Float8E5Bits.sign]

theorem chain64_ordered {α : Type} [Inhabited α] (a : Arithmetic α)
    (x y : Fin 64 → α) (initial : α) :
    chain a x y initial = LlamaFF.orderedK16.instructions.foldl
      (fun acc lanes => a.mma16 (fun l => x (lanes l)) (fun l => y (lanes l)) acc) initial := by
  have hf (v : Fin 64 → α) (j : Fin 4) :
      fragment v j.val = fun l : Fin 16 => v ⟨j.val*16+l.val,by omega⟩ := by
    funext l
    simp [fragment,show j.val*16+l.val<64 by omega]
  simp only [chain,show 64/16=4 by decide,LlamaFF.orderedK16,List.foldl_map]
  rw [← List.map_coe_finRange_eq_range]
  simp only [List.foldl_map,hf]

#axiomsClean decodeReal_zero
#axiomsClean chain64_ordered
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.Execution
