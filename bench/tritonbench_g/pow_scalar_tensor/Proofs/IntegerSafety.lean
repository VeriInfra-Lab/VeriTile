import bench.tritonbench_g.pow_scalar_tensor.Proofs.IntegerSteps

namespace VeriTile.Bench.TritonBenchG.PowScalarTensor.IntegerSteps
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000

private theorem stepStmt_assign_inv' {dtype : TileDType} {shape : TileShape}
    {name : RegName} {e : Op dtype shape} {s s' : BlockState}
    (h : stepStmt (.assign dtype shape name e) s = some s') :
    ∃ v, evalOp e s = some v ∧ s' = s.setReg name dtype shape v := by
  cases hv : evalOp e s with
  | none => simp [stepStmt, hv] at h
  | some v =>
      refine ⟨v, rfl, ?_⟩
      rw [stepStmt_assign_eq_some hv] at h
      exact (Option.some.inj h).symm

theorem perTile_traceSafe (candidate : Bool) (val0 : ℝ)
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 tile_size0 : Nat)
    (bounds : RegionBounds) (T : Nat) (t : BlockState) (V : Tile .nat [])
    (hT : t.regs .nat [] "tile_id" = some V)
    (hV : V.data PUnit.unit = T)
    (hin : ∀ i : Fin tile_size0, T * tile_size0 + i.val < s0 →
      (T * tile_size0 + i.val) * in0_stride0 < bounds in0_ptr)
    (hout : ∀ i : Fin tile_size0, T * tile_size0 + i.val < s0 →
      (T * tile_size0 + i.val) * out0_stride0 < bounds out0_ptr) :
    Stmt.TraceSafeList bounds
      (IntegerSource.perTile candidate val0 in0_ptr out0_ptr in0_stride0 out0_stride0 s0
        tile_size0) t := by
  set offV : Tile .nat [] :=
    Tile.bop (NumericDType.mul .nat) Broadcast.nil V (Tile.scalar tile_size0)
    with hoffV
  have hoffData : offV.data PUnit.unit = T * tile_size0 := by
    rw [hoffV, Tile.bop_data]
    show NumericDType.mul .nat (V.data PUnit.unit) tile_size0 = T * tile_size0
    rw [hV]
    rfl
  simp only [IntegerSource.perTile]
  -- `tile_id0 = tile_id`
  refine Stmt.TraceSafeList.cons_intro
    (by simp only [Stmt.TraceSafe, Op.SafeAt]) (fun t1 ht1 => ?_)
  rw [stepStmt_assign_eq_some
    (show evalOp (Op.ref .nat [] "tile_id") t = some V by simp [hT])] at ht1
  obtain rfl := Option.some.inj ht1
  -- `offset0 = tile_id0 * tile_size0`
  refine Stmt.TraceSafeList.cons_intro
    (by cases candidate <;> simp [Stmt.TraceSafe, Op.SafeAt, IntegerSource.offset]) (fun t2 ht2 => ?_)
  rw [stepStmt_assign_eq_some
    (show evalOp (IntegerSource.offset candidate tile_size0)
        (t.setReg "tile_id0" .nat [] V) = some offV by
      change evalOp (RealSource.offset candidate tile_size0) _ = some offV
      rw [RealTransport.offset_eval]
      simp [hoffV])] at ht2
  obtain rfl := Option.some.inj ht2
  -- `in0_bptr = tl.make_block_ptr(...)`
  refine Stmt.TraceSafeList.cons_intro
    (by simp only [Stmt.TraceSafe, Op.SafeAt, List.mem_cons, List.not_mem_nil,
      or_false, forall_eq, and_self]) (fun t3 ht3 => ?_)
  rw [stepStmt_assign_eq_some
    (makeBlockPtr_1d_eval in0_ptr s0 in0_stride0 tile_size0 "offset0"
      ((t.setReg "tile_id0" .nat [] V).setReg "offset0" .nat [] offV) offV
      (T * tile_size0) (by simp) hoffData)] at ht3
  obtain rfl := Option.some.inj ht3
  -- the boundary-checked block-pointer load
  refine Stmt.TraceSafeList.cons_intro ?_ (fun t4 ht4 => ?_)
  · simp only [Stmt.TraceSafe, Op.SafeAt, MaskOpt.Active, Op.MemorySafe,
      MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe]
    refine ⟨trivial, trivial, trivial, ?_⟩
    intro ptrs hptrs idx _
    rw [evalOp_ref, BlockState.setReg_same] at hptrs
    obtain rfl := Option.some.inj hptrs
    intro hib
    simp only [TileShape.indexToList, BlockPtr.address_1d, Nat.zero_add]
    refine hin idx.1 ?_
    simpa [TileShape.indexToList] using hib
  · obtain ⟨v4, -, rfl⟩ := stepStmt_assign_inv' ht4
    -- `out0 = pow(val0, in0)`
    refine Stmt.TraceSafeList.cons_intro
      (by simp only [Stmt.TraceSafe, Op.SafeAt, and_self])
      (fun t5 ht5 => ?_)
    obtain ⟨v5, -, rfl⟩ := stepStmt_assign_inv' ht5
    -- `out0_bptr = tl.make_block_ptr(...)`
    refine Stmt.TraceSafeList.cons_intro
      (by simp only [Stmt.TraceSafe, Op.SafeAt, List.mem_cons, List.not_mem_nil,
        or_false, forall_eq, and_self]) (fun t6 ht6 => ?_)
    rw [stepStmt_assign_eq_some
      (makeBlockPtr_1d_eval out0_ptr s0 out0_stride0 tile_size0 "offset0" _ offV
        (T * tile_size0) (by simp) hoffData)] at ht6
    obtain rfl := Option.some.inj ht6
    -- the boundary-checked block-pointer store
    refine Stmt.TraceSafeList.cons_intro ?_
      (fun _ _ => Stmt.TraceSafeList.nil_intro)
    simp only [Stmt.TraceSafe, Op.SafeAt, MaskOpt.SafeAt, MaskOpt.Active,
      MaskOpt.MemorySafe, Op.MemorySafe, MemAccess.SafeAt, MemAccess.MemorySafe,
      memAccessMemorySafe, MemAccess.ActiveAddressSafe,
      memAccessActiveAddressSafe]
    refine ⟨trivial, trivial, trivial, trivial, ?_⟩
    intro ptrs hptrs idx _
    rw [evalOp_ref, BlockState.setReg_same] at hptrs
    obtain rfl := Option.some.inj hptrs
    intro hib
    simp only [TileShape.indexToList, BlockPtr.address_1d, Nat.zero_add]
    refine hout idx.1 ?_
    simpa [TileShape.indexToList] using hib


end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.PowScalarTensor.IntegerSteps
