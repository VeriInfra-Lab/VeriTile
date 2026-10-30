import bench.tritonbench_g.attention_llama.Proofs.Values

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.Source
open VeriTile Triton FP.Structural
noncomputable section

/-- Row reductions expose their actual ordered input function. -/
theorem row_reduction (xs : Values (Term α) .real [64,64]) (i : TileIndex [64]) :
    reductionInputs (shape:=[64,64]) ⟨1,by decide⟩ Bool.false xs i=
      (fun j : Fin 64=>xs (i.1,j,())) := by
  funext j
  rfl

theorem score_function (c : Config) (s : State (Term α)) (scale : Term α) (n : Nat) (r : Fin 64) :
    scores c s scale n r=(fun k : Fin 64=>
      let dot := Term.dot c.D (fun d=>query c s (r,d,()))
        (fun d=>loaded c s c.K c.skz c.skh c.skn c.skk n (k,d,()))
      let score := Term.mul (.add (.literal 0) dot) scale
      let masked := if n*64+k.val<c.N then score else .negativeInfinity
      if c.causal then if n*64+k.val+c.SP≤rows s r then masked else .negativeInfinity else masked) := rfl

#axiomsClean row_reduction
#axiomsClean score_function
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.Source
