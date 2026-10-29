import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Memory
import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Native
namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked
open VeriTile Triton Direct
noncomputable section
set_option maxHeartbeats 4000000

def mathStores (r : Request) (heap : ByteMemory) (h : Admitted r heap)
    (final : FPValues.Data ℝ) (before : Nat → ℝ) : Nat → ℝ :=
  writeback r heap h (fun i => final.output (outputIndex r heap h i.program)) before

def nativeStores (r : Request) (heap : ByteMemory) (h : Admitted r heap)
    (ops : Native.Instructions) (final : FPValues.Data Native.Expr) : ByteMemory :=
  writeback r heap h (fun i => encode ((final.output (outputIndex r heap h i.program)).eval ops) i.byte) heap

theorem mathStores_logical (r : Request) (heap : ByteMemory) (h : Admitted r heap)
    (final : FPValues.Data ℝ) (before : Nat → ℝ) (p : Program r heap h)
    (active : indices r heap p.batch≠-1) :
    logical (mathStores r heap h final before) r.output.tensor (outputIndex r heap h p)=
      final.output (outputIndex r heap h p) := by
  rw [logical_direct r.output _ (checked r heap h).dense.2.2.1.1 _ (output_inside r heap h p)]
  simpa only [mathStores,address,Nat.add_zero] using
    writeback_readback r heap h (fun i => final.output (outputIndex r heap h i.program)) before ⟨p,active,0⟩

theorem nativeStores_word (r : Request) (heap : ByteMemory) (h : Admitted r heap)
    (ops : Native.Instructions) (final : FPValues.Data Native.Expr) (p : Program r heap h)
    (active : indices r heap p.batch≠-1) :
    load32 (nativeStores r heap h ops final) (r.y.dataPtr.toNat+4*outputIndex r heap h p)=
      (final.output (outputIndex r heap h p)).eval ops := by
  have bytes : (fun b : Fin 4 => nativeStores r heap h ops final
      (r.y.dataPtr.toNat+4*outputIndex r heap h p+b.val))=
      encode ((final.output (outputIndex r heap h p)).eval ops) := by
    funext b
    exact writeback_readback r heap h _ heap ⟨p,active,b⟩
  rw [load32,bytes,decode_encode]

theorem nativeStores_logical (r : Request) (heap : ByteMemory) (h : Admitted r heap)
    (ops : Native.Instructions) (final : FPValues.Data Native.Expr) (p : Program r heap h)
    (active : indices r heap p.batch≠-1) :
    Torch.FP32.logicalValue (fun a => ⟨load32 (nativeStores r heap h ops final) a⟩)
      r.output.tensor r.output.laws (outputIndex r heap h p)=
        ⟨(final.output (outputIndex r heap h p)).eval ops⟩ := by
  rw [fp32_logical r.output _ (checked r heap h).dense.2.2.1.2.1,
    logical_direct r.output _ (checked r heap h).dense.2.2.1.1 _ (output_inside r heap h p),
    nativeStores_word r heap h ops final p active]

theorem index_complement (r : Request) (heap : ByteMemory) (h : Admitted r heap)
    (j : Nat) (outside : ∀ p : Program r heap h,indices r heap p.batch≠-1 → outputIndex r heap h p≠j)
    (b : Fin 4) : ¬∃ i : Lane r heap h,address r heap h i=r.y.dataPtr.toNat+4*j+b.val := by
  rintro ⟨i,hit⟩
  have bi:=i.byte.isLt
  have bj:=b.isLt
  simp only [address] at hit
  exact outside i.program i.active (by omega)

theorem nativeStores_untouched (r : Request) (heap : ByteMemory) (h : Admitted r heap)
    (ops : Native.Instructions) (final : FPValues.Data Native.Expr) (j : Nat) (hj : j<r.y.numel)
    (outside : ∀ p : Program r heap h,indices r heap p.batch≠-1 → outputIndex r heap h p≠j) :
    Torch.FP32.logicalValue (fun a => ⟨load32 (nativeStores r heap h ops final) a⟩)
      r.output.tensor r.output.laws j=
    Torch.FP32.logicalValue (fun a => ⟨load32 heap a⟩) r.output.tensor r.output.laws j := by
  rw [fp32_logical r.output _ (checked r heap h).dense.2.2.1.2.1,
    fp32_logical r.output _ (checked r heap h).dense.2.2.1.2.1,
    logical_direct r.output _ (checked r heap h).dense.2.2.1.1 j hj,
    logical_direct r.output _ (checked r heap h).dense.2.2.1.1 j hj]
  congr 1
  unfold load32
  congr 1
  funext b
  exact writeback_frame r heap h _ heap _ (index_complement r heap h j outside b)

theorem mathStores_untouched (r : Request) (heap : ByteMemory) (h : Admitted r heap)
    (final : FPValues.Data ℝ) (read : Nat → ℝ) (j : Nat) (hj : j<r.y.numel)
    (outside : ∀ p : Program r heap h,indices r heap p.batch≠-1 → outputIndex r heap h p≠j) :
    logical (mathStores r heap h final read) r.output.tensor j=logical read r.output.tensor j := by
  rw [logical_direct r.output _ (checked r heap h).dense.2.2.1.1 j hj,
    logical_direct r.output _ (checked r heap h).dense.2.2.1.1 j hj]
  exact writeback_frame r heap h _ read _ (by simpa using index_complement r heap h j outside 0)

#axiomsClean mathStores_logical
#axiomsClean nativeStores_word
#axiomsClean nativeStores_logical
#axiomsClean nativeStores_untouched
#axiomsClean mathStores_untouched
#specNonCircular nativeStores avoiding [Native.reference,Native.execute,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked
