import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Correct
import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Misc.NativeCorrect
import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.TensorTarget
import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Stores
import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Inputs
import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Safety

/-! Original complete Tensor applications and physical byte/read-backing frames.
The source execution is the same Signed.Grid.executes/FPGrid.values composition
exposed by KernelRaw; the existing scalar store/readback lemmas supply the API bridge. -/
namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.TensorBridge
open VeriTile Triton
noncomputable section

/-- Complete signed shared-source Real execution, independent GEMV at actual
logical Tensor addresses, every untouched logical output, exact exterior, and
all complete physical read backings. No finite-value/encoder premise. -/
theorem correct (r : Request) (heap : ByteMemory) (h : Admitted r heap) (read : Nat → ℝ) :
    ∃ final,Real.execute r heap h read=some final ∧
      (∀ p : Program r heap h,indices r heap p.batch≠-1 →
        logical (mathStores r heap h final read) r.output.tensor (outputIndex r heap h p)=
          tensorTarget r heap h read p) ∧
      (∀ j,j<r.y.numel →
        (∀ p : Program r heap h,indices r heap p.batch≠-1 → outputIndex r heap h p≠j) →
        logical (mathStores r heap h final read) r.output.tensor j=logical read r.output.tensor j) ∧
      (∀ a,(¬∃ i : Lane r heap h,address r heap h i=a) → mathStores r heap h final read a=read a) ∧
      ReadFrame r read (mathStores r heap h final read) ∧
      (∀ p : Program r heap h,256*(call r heap h).BK≤2^20) := by
  obtain ⟨final,run,values⟩:=Real.source_result r heap h read
  refine ⟨final,run,?_,mathStores_untouched r heap h final read,writeback_frame r heap h _ read,
    writeback_inputs r heap h _ read,fun p => (static_tile r heap h p).1⟩
  intro p active
  rw [mathStores_logical r heap h final read p active,values p active,tensor_formula r heap h read p active]

#axiomsClean correct
#specNonCircular tensorTarget avoiding [Real.execute,mathStores]
#guard_msgs (drop info) in
#auditModuleAxioms


/-- Complete signed source execution and all actual logical output words and
bytes under only five fixed-arity local instruction laws. The exact exterior
and full read-backing frame include skipped rows, output padding and offsets. -/
theorem fp_equiv (r : Request) (heap : ByteMemory) (h : Admitted r heap) (plans : Native.Plans)
    (actual selected : Native.Instructions) (numeric : Native.LocalConformance actual selected) :
    ∃ final,Native.execute plans r heap h=some final ∧ NativeReads r heap actual ∧
      (∀ p : Program r heap h,indices r heap p.batch≠-1 →
        Torch.FP32.logicalValue (fun a => ⟨load32 (nativeStores r heap h actual final) a⟩)
          r.output.tensor r.output.laws (outputIndex r heap h p)=
            ⟨Native.reference selected plans r heap h p⟩) ∧
      (∀ i : Lane r heap h,nativeStores r heap h actual final (address r heap h i)=
        encode (Native.reference selected plans r heap h i.program) i.byte) ∧
      (∀ j,j<r.y.numel →
        (∀ p : Program r heap h,indices r heap p.batch≠-1 → outputIndex r heap h p≠j) →
        Torch.FP32.logicalValue (fun a => ⟨load32 (nativeStores r heap h actual final) a⟩)
          r.output.tensor r.output.laws j=
        Torch.FP32.logicalValue (fun a => ⟨load32 heap a⟩) r.output.tensor r.output.laws j) ∧
      (∀ a,(¬∃ i : Lane r heap h,address r heap h i=a) → nativeStores r heap h actual final a=heap a) ∧
      ReadFrame r heap (nativeStores r heap h actual final) ∧
      (∀ p : Program r heap h,256*(call r heap h).BK≤2^20) := by
  obtain ⟨final,run,values⟩:=Native.source_native plans r heap h actual selected numeric
  refine ⟨final,run,native_reads r heap h actual,?_,?_,nativeStores_untouched r heap h actual final,
    writeback_frame r heap h _ heap,writeback_inputs r heap h _ heap,fun p => (static_tile r heap h p).1⟩
  · intro p active
    rw [nativeStores_logical r heap h actual final p active]
    exact congrArg (fun w => (⟨w⟩ : Torch.FP32.Word)) (values p active)
  · intro i
    rw [show nativeStores r heap h actual final (address r heap h i)=
      encode ((final.output (outputIndex r heap h i.program)).eval actual) i.byte from
        writeback_readback r heap h _ heap i]
    exact congrArg (fun w => encode w i.byte) (values i.program i.active)

#axiomsClean fp_equiv
#specNonCircular Native.reference avoiding [Native.execute,nativeStores,Signed.Grid.run,FP.Structural.exec]
#guard_msgs (drop info) in
#auditModuleAxioms

end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.TensorBridge
