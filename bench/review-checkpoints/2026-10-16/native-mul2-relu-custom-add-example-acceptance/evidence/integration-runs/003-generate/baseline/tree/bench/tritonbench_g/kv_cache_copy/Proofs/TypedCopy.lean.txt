import bench.tritonbench_g.kv_cache_copy.Proofs.Execution

namespace VeriTile.Bench.TritonBenchG.KvCacheCopy.TypedCopy
open VeriTile.Triton FP.Structural KcacheCopyTriton.TypedCopy
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

/-- Both payloads are read from the incoming memory snapshot, including when
K/V input regions alias either output. Actual store ordering is retained. -/
theorem execution_memory {α : Type} [Inhabited α] (rest : Algebra α) (kd vd : Storage)
    (p : Parameters) (s : State α) :
    (execution rest kd vd p s).val.mem = (copiedKV kd vd p s).mem := by
  cases kd <;> cases vd
  all_goals simp +contextual [execution, copiedKV, copiedK, masked_scatter_memory, loaded, vLoaded,
    Storage.tag, active, inputOffset, outputOffset, vInputOffset, vOutputOffset,
    blockId, past, FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    integralFloorDiv, integralMod, Region.cast]

/-- K readback needs only the existing separate-cache condition. Its lane
addresses are automatically injective, independently of all strides. -/
theorem k_readback {α : Type} [Inhabited α] (rest : Algebra α) (kd vd : Storage)
    (p : Parameters) (s : State α) (i : TileIndex [p.width]) (hi : active p i)
    (hcache : p.KCache ≠ p.VCache) :
    (execution rest kd vd p s).val.mem p.KCache (outputOffset p.toParameters s i) =
      .mk kd.tag ((s.mem p.K (inputOffset p.toParameters s i)).read kd.tag) := by
  rw [execution_memory]
  unfold copiedKV
  rw [State.masked_scatter_frame _ _ _ _ _ _ _ (Or.inl hcache)]
  exact State.masked_scatter_readback _ _ _ _ _ i hi (output_injective p.toParameters s)

/-- V readback retains the selected contract's no-aliasing/write-injectivity
condition. This is an address condition, not a numerical law. -/
theorem v_readback {α : Type} [Inhabited α] (rest : Algebra α) (kd vd : Storage)
    (p : Parameters) (s : State α) (i : TileIndex [p.width]) (hi : active p i)
    (hinj : ∀ a b, active p a → active p b → vOutputOffset p s a = vOutputOffset p s b → a = b) :
    (execution rest kd vd p s).val.mem p.VCache (vOutputOffset p s i) =
      .mk vd.tag ((s.mem p.V (vInputOffset p s i)).read vd.tag) := by
  rw [execution_memory]
  exact masked_readback_active _ _ _ _ _ i hi hinj


 theorem k_typed_readback {α : Type} [Inhabited α] (rest : Algebra α) (kd vd : Storage)
    (p : Parameters) (s : State α) (i : TileIndex [p.width]) (hi : active p i)
    (hcache : p.KCache ≠ p.VCache) (value : Value α kd.tag)
    (htyped : s.mem p.K (inputOffset p.toParameters s i) = .mk kd.tag value) :
    (execution rest kd vd p s).val.mem p.KCache (outputOffset p.toParameters s i) = .mk kd.tag value := by
  rw [k_readback rest kd vd p s i hi hcache, htyped, Cell.read_same]

 theorem v_typed_readback {α : Type} [Inhabited α] (rest : Algebra α) (kd vd : Storage)
    (p : Parameters) (s : State α) (i : TileIndex [p.width]) (hi : active p i)
    (hinj : ∀ a b, active p a → active p b → vOutputOffset p s a = vOutputOffset p s b → a = b)
    (value : Value α vd.tag)
    (htyped : s.mem p.V (vInputOffset p s i) = .mk vd.tag value) :
    (execution rest kd vd p s).val.mem p.VCache (vOutputOffset p s i) = .mk vd.tag value := by
  rw [v_readback rest kd vd p s i hi hinj, htyped, Cell.read_same]

/-- Positive V component stride is a sufficient concrete check for injectivity. -/
theorem v_output_injective {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (hstride : 0 < p.strideVcd) : Function.Injective (vOutputOffset p s) := by
  intro i j h
  have hm := Nat.add_left_cancel h
  have ha := (Nat.mul_left_inj (Nat.ne_of_gt hstride)).mp hm
  exact Prod.ext (Fin.ext (Nat.add_left_cancel ha)) (Subsingleton.elim _ _)

/-- Inactive lanes and every cell outside both actual write sets are framed. -/
theorem frame {α : Type} [Inhabited α] (rest : Algebra α) (kd vd : Storage)
    (p : Parameters) (s : State α) (r : RegionName) (o : Nat)
    (hk : r ≠ p.KCache ∨ ∀ i : TileIndex [p.width], active p i → o ≠ outputOffset p.toParameters s i)
    (hv : r ≠ p.VCache ∨ ∀ i : TileIndex [p.width], active p i → o ≠ vOutputOffset p s i) :
    (execution rest kd vd p s).val.mem r o = s.mem r o := by
  rw [execution_memory]
  unfold copiedKV copiedK
  rw [State.masked_scatter_frame _ _ _ _ _ _ _ (hv.imp id (fun h i _ => h i)),
      State.masked_scatter_frame _ _ _ _ _ _ _ (hk.imp id (fun h i _ => h i))]

/-- Success and both independently specified copies, with the original
address-side readback gate and complete off-write-set preservation. -/
theorem correctness {α : Type} [Inhabited α] (rest : Algebra α) (kd vd : Storage)
    (p : Parameters) (s : State α) (hcache : p.KCache ≠ p.VCache)
    (hinj : ∀ a b, active p a → active p b → vOutputOffset p s a = vOutputOffset p s b → a = b) :
    ∃ t, FP.Structural.exec (copyBackend rest) (source kd vd p) s = some t ∧
      (∀ i : TileIndex [p.width], active p i →
        t.mem p.KCache (outputOffset p.toParameters s i) =
          .mk kd.tag ((s.mem p.K (inputOffset p.toParameters s i)).read kd.tag)) ∧
      (∀ i : TileIndex [p.width], active p i →
        t.mem p.VCache (vOutputOffset p s i) =
          .mk vd.tag ((s.mem p.V (vInputOffset p s i)).read vd.tag)) ∧
      (∀ r o,
        (r ≠ p.KCache ∨ ∀ i : TileIndex [p.width], active p i → o ≠ outputOffset p.toParameters s i) →
        (r ≠ p.VCache ∨ ∀ i : TileIndex [p.width], active p i → o ≠ vOutputOffset p s i) →
        t.mem r o = s.mem r o) :=
  ⟨(execution rest kd vd p s).val, (execution rest kd vd p s).property,
    fun i hi => k_readback rest kd vd p s i hi hcache,
    fun i hi => v_readback rest kd vd p s i hi hinj,
    frame rest kd vd p s⟩

#axiomsClean source_projection
#axiomsClean execution
#axiomsClean execution_memory
#axiomsClean k_readback
#axiomsClean v_readback
#axiomsClean k_typed_readback
#axiomsClean v_typed_readback
#axiomsClean frame
#axiomsClean correctness

end
end VeriTile.Bench.TritonBenchG.KvCacheCopy.TypedCopy
