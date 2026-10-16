import bench.tritonbench_g.isfinite_kernel.Proofs.BodyMemory

/-! Readback and full tagged frame for the already derived body-memory
formula. The scatter bridge only projects memory; it is not a kernel run or
an assumed prepared state. Positive output stride is required for readback,
while framing and preservation of the input region need no stride condition. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.Parametric
open VeriTile.Triton FP.Structural TypedBits
noncomputable section

theorem tileMemory_eq_scatter {α : Type} [Inhabited α]
    (classify : α → Bool) (p : Parameters) (base : Nat) (s : State α) :
    tileMemory classify p base s.mem =
      ((TileShape.allIndices [p.block]).foldl (fun t i =>
        if base + i.1.val < p.size then
          t.write p.output ((base + i.1.val) * p.outputStride)
            (.mk .bool (classify ((s.mem p.input
              ((base + i.1.val) * p.inputStride)).read .real)))
        else t) s).mem := by
  exact (fold_writes_mem (TileShape.allIndices [p.block])
    (fun i => base + i.1.val < p.size) (fun _ => p.output)
    (fun i => (base + i.1.val) * p.outputStride)
    (fun i => .mk .bool (classify ((s.mem p.input
      ((base + i.1.val) * p.inputStride)).read .real))) s).symm

theorem lane_address_injective (p : Parameters) (base : Nat)
    (hstride : 0 < p.outputStride) :
    Function.Injective (fun i : TileIndex [p.block] =>
      (base + i.1.val) * p.outputStride) := by
  intro i j heq
  have hval : i.1.val = j.1.val :=
    Nat.add_left_cancel (Nat.mul_right_cancel hstride heq)
  apply Prod.ext
  · exact Fin.ext hval
  · exact Subsingleton.elim _ _

theorem tileMemory_readback {α : Type} [Inhabited α]
    (classify : α → Bool) (p : Parameters) (base : Nat) (s : State α)
    (i : Fin p.block) (hi : base + i.val < p.size)
    (hstride : 0 < p.outputStride) :
    tileMemory classify p base s.mem p.output ((base + i.val) * p.outputStride) =
      .mk .bool (classify ((s.mem p.input
        ((base + i.val) * p.inputStride)).read .real)) := by
  rw [tileMemory_eq_scatter]
  exact State.masked_scatter_readback p.output
    (fun k : TileIndex [p.block] => (base + k.1.val) * p.outputStride)
    (fun k => .mk .bool (classify ((s.mem p.input
      ((base + k.1.val) * p.inputStride)).read .real)))
    (fun k => base + k.1.val < p.size) s (i, PUnit.unit) hi
    (lane_address_injective p base hstride)

theorem tileMemory_frame {α : Type} [Inhabited α]
    (classify : α → Bool) (p : Parameters) (base : Nat) (s : State α)
    (r : RegionName) (o : Nat)
    (hmiss : r ≠ p.output ∨ ∀ i : Fin p.block,
      base + i.val < p.size → o ≠ (base + i.val) * p.outputStride) :
    tileMemory classify p base s.mem r o = s.mem r o := by
  rw [tileMemory_eq_scatter]
  apply State.masked_scatter_frame
  rcases hmiss with hr | ho
  · exact Or.inl hr
  · exact Or.inr (fun i _ hi => ho i.1 hi)

theorem tileMemory_input {α : Type} [Inhabited α]
    (classify : α → Bool) (p : Parameters) (base : Nat) (s : State α)
    (hne : p.input ≠ p.output) :
    tileMemory classify p base s.mem p.input = s.mem p.input := by
  funext o
  exact tileMemory_frame classify p base s p.input o (Or.inl hne)

#axiomsClean tileMemory_eq_scatter
#axiomsClean lane_address_injective
#axiomsClean tileMemory_readback
#axiomsClean tileMemory_frame
#axiomsClean tileMemory_input
end
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.Parametric
