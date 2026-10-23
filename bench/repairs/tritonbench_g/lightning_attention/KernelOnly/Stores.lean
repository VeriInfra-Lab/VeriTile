import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.Memory
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000

def mathStores {phase} (r : Request phase) (final : Memory ℝ) (before : Nat → ℝ) : Nat → ℝ :=
  writeback r (fun i => (final (resultRegion i.output) i.element.val).read .real) before

def nativeStores {phase} (r : Request phase) (final : RegionName → Nat → Word) (before : ByteMemory) : ByteMemory :=
  writeback r (fun i => encode (final (resultRegion i.output) i.element.val) i.byte) before

theorem mathStores_logical {phase} (r : Request phase) (h : Contract r) (final : Memory ℝ)
    (before : Nat → ℝ) (j : Fin (outputs phase)) (i : Fin (count r j)) :
    logical (mathStores r final before) (r.result j) i.val=(final (resultRegion j) i.val).read .real := by
  have bound : i.val<(r.result j).tensor.metadata.numel := by
    rw [descriptor_numel _ _ _ _ _ (h.result j)]
    exact i.isLt
  rw [logical_direct _ _ (h.result j).1 ((h.result j).2.2.2.1 rfl) _ bound]
  simpa only [mathStores,address,Nat.add_zero] using
    writeback_readback r h (fun i => (final (resultRegion i.output) i.element.val).read .real) before ⟨j,i,0⟩

theorem nativeStores_word {phase} (r : Request phase) (h : Contract r)
    (final : RegionName → Nat → Word) (before : ByteMemory)
    (j : Fin (outputs phase)) (i : Fin (count r j)) :
    load32 (nativeStores r final before) ((r.result j).tensor.dataPtr.toNat+4*i.val)=final (resultRegion j) i.val := by
  have each : (fun b : Fin 4 => nativeStores r final before ((r.result j).tensor.dataPtr.toNat+4*i.val+b.val)) =
      encode (final (resultRegion j) i.val) := by
    funext b
    exact writeback_readback r h _ before ⟨j,i,b⟩
  rw [load32,each,decode_encode]

theorem nativeStores_logical {phase} (r : Request phase) (h : Contract r)
    (final : RegionName → Nat → Word) (before : ByteMemory)
    (j : Fin (outputs phase)) (i : Fin (count r j)) :
    Torch.FP32.logicalValue (fun a => ⟨load32 (nativeStores r final before) a⟩)
      (r.result j).tensor (h.result j).1.tensor i.val = ⟨final (resultRegion j) i.val⟩ := by
  have bound : i.val<(r.result j).tensor.metadata.numel := by
    rw [descriptor_numel _ _ _ _ _ (h.result j)]
    exact i.isLt
  rw [←logical_fp32 _ _ (h.result j).1 (h.result j).2.2.2.2.1,
    logical_direct _ _ (h.result j).1 ((h.result j).2.2.2.1 rfl) _ bound,
    nativeStores_word r h]

#axiomsClean mathStores_logical
#axiomsClean nativeStores_word
#axiomsClean nativeStores_logical
#specNonCircular nativeStores avoiding [FP.Structural.exec,FP.Structural.run]
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly
