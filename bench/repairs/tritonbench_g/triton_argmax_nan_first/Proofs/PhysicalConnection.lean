import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.NativePublicIO
import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.NativeDimensionIO
import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.NativeKernelResult

/-! Explicit connection between the new complete typed schedule and the old
physical byte execution. Both use the same first/second quotations and existing
scalar codecs; no additional input or ordering restriction is introduced. -/
namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.PhysicalConnection
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source Memory
noncomputable section

/-- Every original requested-format physical execution has the same INT64
result as the new full typed schedule initialized from those exact bytes. -/
theorem global (mem : Primitives M) (casts : Request.Casts) (kind : Request.Kind)
    (binding : BoundSource.Binding casts kind) (rest : Algebra Word) (template : State Word)
    (c : GlobalConfig) (g : Native.GlobalPrograms c) (p : GlobalMemory.Buffers)
    (layout : GlobalMemory.Layout kind.format c p) (positive : 0 < c.block)
    (midPositive : 0 < c.midBlock) (active : ∀ pid, pid < c.blocks → pid*c.block < c.size)
    (index64 : Bool) (before : M)
    (initialized : ∀ i, i < c.midBlock → mem.read 8 before (address p.midIndex 8 i) = 0) :
    ∃ final, runSchedule (IO.nativeSchedule casts kind binding rest c index64)
        (BoundSource.entry mem binding p before template 0) = some final ∧
      BoundSource.GlobalRun mem binding rest template c p index64 before
        (Native.global mem (Request.scratch casts kind) kind.isFloating g p before) ∧
      mem.read 8 (Native.global mem (Request.scratch casts kind) kind.isFloating g p before)
        (address p.output 8 0) =
          BitVec.ofInt 64 ((final.mem (Region.cast BoundSource.O) 0).read .int) := by
  have init : ∀ i, i < c.midBlock →
      ((BoundSource.entry mem binding p before template 0).mem (Region.cast BoundSource.I) i).read .int = 0 := by
    intro i hi
    rw [BoundSource.entry_index, initialized i hi]
    rfl
  obtain ⟨final, ran, output, _⟩ := NativeIO.global_execution casts kind binding rest c g
    (GlobalMemory.input mem p before) (BoundSource.entry mem binding p before template 0)
    index64 positive midPositive active (by intro i _; exact BoundSource.entry_input ..) init
  have physical := NativeKernelResult.global mem casts kind binding rest template c g p layout
    positive midPositive active index64 before initialized
  refine ⟨final, ran, physical.1, ?_⟩
  rw [physical.2.1, output]
  rfl

/-- Every original physical dimension grid has the same result at every
coordinate as the complete typed source schedule. Empty grids remain admitted. -/
theorem dimension (mem : Primitives M) (rest : Algebra Word) (template : State Word)
    (c : DimConfig) (g : Native.DimPrograms c) (p : DimMemory.Buffers)
    (layout : DimMemory.Layout c p) (nonempty : 0 < c.extent) (index64 : Bool) (before : M) :
    ∃ final, runSchedule (IO.nativeDimensionSchedule rest c index64)
        (BoundSource.dimEntry mem p before template 0) = some final ∧
      BoundSource.DimensionRun mem rest template c p index64 before (Native.dimension mem g p before) ∧
      (∀ row inner, row < c.outer → inner < c.inner →
        mem.read 8 (Native.dimension mem g p before) (address p.output 8 (dimOutputOffset c row inner)) =
          BitVec.ofInt 64 ((final.mem (Region.cast BoundSource.O) (row*c.inner+inner)).read .int)) := by
  let input := fun r i col => DimMemory.input mem p before (r*c.extent*c.inner+col*c.inner+i)
  obtain ⟨final, ran, output, _⟩ := NativeDimensionIO.grid_execution rest c index64 nonempty
    (BoundSource.groups c) (fun job hj => BoundSource.group_inner c job hj)
    (BoundSource.dimEntry mem p before template 0) input
    (by intro r col inner _ _ _; exact BoundSource.dim_entry_input ..)
  have physical := NativeKernelResult.dimension mem rest template c g p layout nonempty index64 before
  refine ⟨final, ran, physical.1, ?_⟩
  intro row inner hr hi
  have member : (row/128,inner) ∈ BoundSource.groups c := by
    simp only [BoundSource.groups, List.mem_flatMap, List.mem_map, List.mem_range]
    exact ⟨row/128, by omega, inner, hi, rfl⟩
  rw [physical.2.1 row inner hr hi, output row inner hr hi, if_pos member]
  rfl

#axiomsClean global
#axiomsClean dimension
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.PhysicalConnection
