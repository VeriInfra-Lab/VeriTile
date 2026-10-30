import bench.repairs.tritonbench_g.masked_add_cuda.Proofs.Typed

namespace VeriTile.Bench.Repairs.MaskedAddCuda.RealProof
open VeriTile.Triton FP.Structural
noncomputable section

/-- Only operations present in this source are interpreted mathematically. -/
def algebra (rest : Algebra ℝ) : Algebra ℝ :=
  { rest with literal := fun _ _ x => x
              fp32Load := id
              cast := fun _ _ _ x => x
              binary := fun precision dtype op x y =>
                match op with
                | .add => x + y
                | .mul => x * y
                | _ => rest.binary precision dtype op x y }

theorem update_real (rest : Algebra ℝ) (a g p : ℝ) :
    updateValue (algebra rest) a g p = g + p*a := rfl

#axiomsClean update_real
end
end VeriTile.Bench.Repairs.MaskedAddCuda.RealProof
