import bench.repairs.tritonbench_g.flash_decode2_llama.Proofs.RawKernel

namespace VeriTile.Bench.Repairs.FlashDecode2Llama.RealKernelResult
open VeriTile.Triton FP.Structural FlashDecode2 KernelInterface
noncomputable section

/-- Full unmasked source execution on raw state, independent weighted summary
at every output coordinate, and all exterior tagged memory cells. -/
theorem kernel_real (g : KernelInterface.Geometry)
    (data : RegionName → Nat → ℝ) (before : State (Option ℝ))
    (bounds : LengthBounds g before) (order : List Nat)
    (coverage : order.Perm (List.range (g.B*g.H))) :
    Spec.Real (∃ after,
      Grid.source Reference.ideal Bool.false g.B g.H g.S g.D g.Q g.D order
        (realEntry data before)=some after ∧
      (∀ row (k : TileIndex [g.D]),row<g.B*g.H →
        after.mem "O" (row*g.D+lane k)=
          .mk .real (some (realTarget g data before row (lane k)))) ∧
      (∀ r i,r≠("O" : RegionName) ∨ g.B*g.H*g.D ≤ i →
        after.mem r i=(realEntry data before).mem r i)) := by
  obtain ⟨after,run,values,frame⟩ :=
    KernelInterface.typed Reference.ideal g order coverage (realEntry data before)
  refine ⟨after,run,?_,frame⟩
  intro row k inside
  rw [values row k inside,real_value g data before bounds row inside k]

#axiomsClean kernel_real
#specNonCircular realTarget avoiding [Grid.source,FP.Structural.exec,values,Reference.recurrence]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FlashDecode2Llama.RealKernelResult
