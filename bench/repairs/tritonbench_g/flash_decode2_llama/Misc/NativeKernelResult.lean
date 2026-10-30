import bench.repairs.tritonbench_g.flash_decode2_llama.Proofs.RawNative

namespace VeriTile.Bench.Repairs.FlashDecode2Llama.NativeKernelResult
open VeriTile.Triton FP.Structural FlashDecode2 KernelInterface
noncomputable section

/-- Full source success, independent native-word summary for the original
scalar and compiler patterns, and every exterior typed memory cell. -/
theorem kernel_native (p : RawNative.Ops) (g : KernelInterface.Geometry)
    (words : RegionName → Nat → Native.Word) (before : State Native.Term)
    (bounds : LengthBounds g before) (order : List Nat)
    (coverage : order.Perm (List.range (g.B*g.H))) :
    ∃ after,
      Grid.source Native.algebra Bool.false g.B g.H g.S g.D g.Q g.D order
        (RawNative.entry words before)=some after ∧
      (∀ row (k : TileIndex [g.D]),row<g.B*g.H →
        Native.observe (RawNative.lower p) (RawNative.schedule g) after "O" (row*g.D+lane k)=
          RawNative.target p g (fun b => (before.mem "L" b).read .nat)
            (words "E") (words "M") row (lane k)) ∧
      (∀ r i,r≠("O" : RegionName) ∨ g.B*g.H*g.D ≤ i →
        after.mem r i=(RawNative.entry words before).mem r i) := by
  obtain ⟨after,run,values,frame⟩ :=
    KernelInterface.typed Native.algebra g order coverage (RawNative.entry words before)
  refine ⟨after,run,?_,frame⟩
  intro row k inside
  unfold Native.observe
  rw [values row k inside,Cell.read_same]
  exact RawNative.value p g words before bounds row inside k

#axiomsClean kernel_native
#specNonCircular RawNative.target avoiding [Grid.source,FP.Structural.exec,values,Native.eval]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FlashDecode2Llama.NativeKernelResult
