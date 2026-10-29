import bench.tritonbench_g.triton_conv2d_fwd.Proofs.SignedLoopSimulation
import VeriTile.Triton.Memory.Flatten

/-! The common terminal write preserves the masked-pointer relation, including
the entire memory. Its address may include an arbitrary flat-allocation base. -/
namespace VeriTile.Bench.TritonBenchG.TritonConv2dFwd.SignedPostSimulation
open VeriTile.Triton MaskedPointerSimulation SignedBodySimulation SignedLoopSimulation
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

theorem write_real (R : RoundingModel) {s t : BlockState} (h : Related s t)
    (r : RegionName) (o : Nat) (v : TileCarrier .real) :
    Related (s.writeMemTypedR R .real r o v) (t.writeMemTypedR R .real r o v) := by
  refine ⟨⟨?_,h.1.pids,h.1.numPids,h.1.undef,h.1.registers⟩,h.2⟩
  funext region offset
  simp only [BlockState.writeMemTypedR,BlockState.writeMemTyped,BlockState.writeMemAs,h.1.memory]

theorem fold_related {ι : Type} (xs : List ι) (f : BlockState → ι → BlockState)
    (step : ∀ s t, Related s t → ∀ i, Related (f s i) (f t i))
    (s t : BlockState) (h : Related s t) :
    Related (xs.foldl f s) (xs.foldl f t) := by
  induction xs generalizing s t with
  | nil => exact h
  | cons i rest ih => exact ih (f s i) (f t i) (step s t h i)

theorem store (R : RoundingModel) {shape : TileShape}
    (ptr : Op .ptr shape) (value : Op .real shape) (mask : Op .bool shape)
    (ptr_same : ∀ s t, Related s t → evalOpR R ptr s = evalOpR R ptr t)
    (value_same : ∀ s t, Related s t → evalOpR R value s = evalOpR R value t)
    (mask_same : ∀ s t, Related s t → evalOpR R mask s = evalOpR R mask t) :
    Simulates R [.store .real shape (.ptr ptr) value (.mask mask)]
      [.store .real shape (.ptr ptr) value (.mask mask)] := by
  intro s t h t' run
  rw [singleton] at run ⊢
  simp only [stepStmtR] at run ⊢
  rw [ptr_same s t h,value_same s t h,mask_same s t h]
  cases hv : evalOpR R value t with
  | none => simp [hv,bind,Option.bind] at run
  | some vs =>
    cases hm : evalOpR R mask t with
    | none => simp [hv,hm,bind,Option.bind] at run
    | some ms =>
      cases hp : evalOpR R ptr t with
      | none => simp [hv,hm,hp,bind,Option.bind] at run
      | some ps =>
        simp only [hv,hm,hp,bind,Option.bind,Option.map_some] at run ⊢
        cases run
        refine ⟨_,rfl,fold_related _ _ ?_ s t h⟩
        intro u v huv i
        split
        · exact write_real R huv _ _ _
        · exact huv

theorem postlude (R : RoundingModel) (A : FlatAlloc) (Input Weight Output : RegionName)
    (batch_dim out_height out_width OBS OOFS OHS OWS OGD BHW OF : Nat) :
    let body := A.flattenStmts (PositivePointerModel.convPostBody Input Weight Output
      batch_dim out_height out_width OBS OOFS OHS OWS OGD BHW OF)
    Simulates R body body := by
  dsimp only
  simp only [PositivePointerModel.convPostBody,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,
    FlatAlloc.flattenAccess,FlatAlloc.flattenMask]
  change Simulates R ([_] ++ [_] ++ [_]) ([_] ++ [_] ++ [_])
  apply append R
  · apply append R
    · apply assign R
      intro s t h
      simp only [FlatAlloc.flattenOp.eq_def,evalOpR.eq_def,
        h.1.registers .nat [BHW] "batch_offset" (by decide),
        h.1.registers .nat [] "group_pid" (by decide),
        h.1.registers .nat [] "out_group_dim" (by decide),
        h.1.registers .nat [OF] "output_feat_offset" (by decide),
        h.1.registers .nat [BHW] "output_height_offset" (by decide),
        h.1.registers .nat [BHW] "output_width_offset" (by decide)]
    · apply assign R
      intro s t h
      simp only [FlatAlloc.flattenOp.eq_def,evalOpR.eq_def,
        h.1.registers .nat [BHW] "batch_offset" (by decide),
        h.1.registers .nat [] "out_group_dim" (by decide),
        h.1.registers .nat [OF] "output_feat_offset" (by decide),
        h.1.registers .nat [BHW] "output_height_offset" (by decide),
        h.1.registers .nat [BHW] "output_width_offset" (by decide)]
  · apply store R
    · intro s t h
      simpa only [FlatAlloc.flattenOp,evalOpR] using h.1.registers .ptr [BHW,OF] "Output" (by decide)
    · intro s t h
      simpa only [FlatAlloc.flattenOp,evalOpR] using h.1.registers .real [BHW,OF] "accum" (by decide)
    · intro s t h
      simpa only [FlatAlloc.flattenOp,evalOpR] using h.1.registers .bool [BHW,OF] "output_mask" (by decide)

#axiomsClean postlude
end
end VeriTile.Bench.TritonBenchG.TritonConv2dFwd.SignedPostSimulation
