import bench.tritonbench_g.token_softmax_bloom.Proofs.MixedContract
namespace VeriTile.Bench.TritonBenchG.TokenSoftmaxBloom.MixedTyped
open VeriTile.Triton FP.Structural TokenSoftmaxBloom.Typed
noncomputable section

theorem reference_readback {α : Type} [Inhabited α] (M : Algebra α) (d out : Storage)
    (p : Parameters) (s : State α) (hstride : p.outputTokenStride ≠ 0)
    (i : TileIndex [p.width]) (hi : i.1.val < count p s) :
    (reference M d out p s).mem p.output (outputOffset p s i) =
      .mk (dtype out).toTileDType (ofFloat (dtype out) (outputValues M d out p s i)) := by
  exact State.masked_scatter_readback _ _ _ _ _ i hi (offset_injective p s hstride)

theorem reference_frame {α : Type} [Inhabited α] (M : Algebra α) (d out : Storage)
    (p : Parameters) (s : State α) (r : RegionName) (o : Nat) (h : Outside p s r o) :
    (reference M d out p s).mem r o = s.mem r o := by
  apply State.masked_scatter_frame
  rcases h with h | h
  · exact Or.inl h
  · exact Or.inr (fun i _ hi => h i hi)

#axiomsClean reference_readback
#axiomsClean reference_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.TokenSoftmaxBloom.MixedTyped
