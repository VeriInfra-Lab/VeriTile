import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.NativeAddresses

namespace VeriTile.Bench.Repairs.ContextAttnFwd.Native
open VeriTile.Triton VeriTile.Bench.TritonBenchG.ContextAttnFwd
noncomputable section

/-- The compiler's hoisted byte-pointer backedge is modular pointer arithmetic.
It remains valid after the final body, even when that unused pointer is outside
the allocation. Actual dereferences are separately bounded by cache_event. -/
theorem pointer_advance (base offset stride tile : Nat) :
    ptr base 4 (word offset)+word (4*(tile*stride))=
      ptr base 4 (word (offset+tile*stride)) := by
  simp only [ptr_modular,word,BitVec.ofNat_add_ofNat]
  congr 1
  ring

/-- Final signed index update/test is representable; there is no body after
this final counter, hence no memory event for its hoisted pointer values. -/
theorem final_counter (c : Host.Call) (abi : ABI c) (idx : GridIndex c.grid) :
    (word (countAt c idx*c.layout.BN)).toInt=((countAt c idx*c.layout.BN : Nat) : Int) ∧
    ¬(word (countAt c idx*c.layout.BN)).slt (word (stopAt c idx))=Bool.true := by
  have hs : stopAt c idx<2^31 :=
    lt_of_le_of_lt (stop_le _ _ _ _) (abi.sequence _ (Host.batch_bound c idx))
  have ht : countAt c idx=ceilTiles (stopAt c idx) c.layout.BM := rfl
  have hb := final_index_bound (stopAt c idx) c.layout.BM (countAt c idx) (tile c) hs (by omega)
  rw [c.blockN]
  refine ⟨signed _ hb,?_⟩
  rw [slt_exact _ _ hb (by omega)]
  have he := ceil_step (stopAt c idx) c.layout.BM (countAt c idx) (tile c)
  omega
end
end VeriTile.Bench.Repairs.ContextAttnFwd.Native
