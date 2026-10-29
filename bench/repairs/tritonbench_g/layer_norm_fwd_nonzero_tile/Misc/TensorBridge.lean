import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Misc.RawRealResults
import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Misc.RawNativeResults
import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.Bytes
import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.ActualPhysical

/-! Optional physical-memory and Tensor-byte applications. These retain their
historical algorithm and compiler-artifact statements with explicit source views.
The current mathematical physical result uses the actual pasted source. Supporting modules do not import the public readers. -/
namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TensorBridge
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Bench.Repairs.LayerNormFwd
noncomputable section

/-- Historical algorithm realization permits read/read aliases and preserves every cell
outside the actual source writes. Byte-level native framing is separate. -/
theorem legacy_correct_physical (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (a : AliasMemory.Layout) (initial : AliasMemory.Physical a) (template : BlockState)
    (width : widthCheck N=Bool.true) (sep : Domain.ReadWriteSeparated f r)
    (geometry : NativeGeometry.Geometry f r stride M N)
    (fresh : AliasMemory.FreshWrites a (AliasMemory.Writes f r stride M N)) :
    ∃ final physical,
      Nonempty (Kernel.GridLaunchedOrdinary ((repaired f r stride M N eps p (tile N) (tile_arange N)).eraseDType.toAlgKernel)
        (WholeGrid.grid M) (AliasMemory.pull a initial template) final) ∧
      AliasMemory.pull a physical final=final ∧
      WholeGrid.Post f r stride M N eps p (AliasMemory.pull a initial template) final ∧
      HostReturn.observe f r stride M N (AliasMemory.pull a physical final)=
        HostReturn.oracle f r stride M N eps p (AliasMemory.pull a initial template) ∧
      (∀ i : Fin a.capacity, (∀ k, AliasMemory.Writes f r stride M N k → a.address k≠i.val) →
        physical i=initial i) :=
  RealRevision.ActiveGrid.grid_return f r stride M N eps p (tile N) (tile_arange N)
    a initial template ((widthCheck_iff N).mp width) sep geometry fresh

/-- Actual copied-source Real rows and all eight return observations admit
physical realization with the same alias/freshness conditions. -/
theorem correct_physical (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (a : AliasMemory.Layout) (initial : TypedPhysical.Physical ℝ a) (template : State ℝ)
    (width : widthCheck N=Bool.true) (sep : Domain.ReadWriteSeparated f r)
    (geometry : NativeGeometry.Geometry f r stride M N)
    (fresh : AliasMemory.FreshWrites a (AliasMemory.Writes f r stride M N)) :
    ∃ final physical,
      Nonempty (TypedSource.GridRun ActualMathematics.algebra f r stride M N eps p
        (tile N) (tile_arange N) (TypedPhysical.pull a initial template) final) ∧
      TypedPhysical.pull a physical final = final ∧
      TypedSource.GridPost ActualMathematics.algebra ActualMathematics.comparisons
        f r stride M N eps p (tile N) (TypedPhysical.pull a initial template) final ∧
      TypedSource.observe f r stride M N (TypedPhysical.pull a physical final) =
        TypedSource.oracle ActualMathematics.algebra ActualMathematics.comparisons
          f r stride M N eps p (tile N) ((widthCheck_iff N).mp width)
          (TypedPhysical.pull a initial template) ∧
      (∀ i : Fin a.capacity, (∀ k, AliasMemory.Writes f r stride M N k → a.address k ≠ i.val) →
        physical i = initial i) :=
  ActualPhysical.source ActualMathematics.algebra ActualMathematics.comparisons
    f r stride M N eps p (tile N) (tile_arange N) ((widthCheck_iff N).mp width)
    a initial template sep geometry fresh

/-- Complete requested-width byte realization and necessary exterior frame.
The byte trace is derived from current-memory execution, not an assumed result. -/
theorem fp_bytes {α : Type} [Inhabited α] (c : StorageMachine.Config α block)
    (artifact : StorageCompiler.Artifact c.profile c.flags c.N block)
    (encode : ByteExecution.Encode α)
    (separated : ByteExecution.Separated (PhysicalDomain.layout c.allocations) (ByteExecution.width c)
      (AliasMemory.Writes c.flags KernelEntry.regions c.strides c.M c.N))
    (initial : StorageMachine.Memory c) (bytes : ByteExecution.Bytes)
    (image : ByteExecution.Image (PhysicalDomain.layout c.allocations) (ByteExecution.width c) encode initial bytes) :
    ByteExecution.Image (PhysicalDomain.layout c.allocations) (ByteExecution.width c) encode
      (StorageMachine.execute c artifact initial) (ByteExecution.execute c artifact encode initial bytes) ∧
    (∀ address, (∀ key, AliasMemory.Writes c.flags KernelEntry.regions c.strides c.M c.N key →
      ¬ByteExecution.Span (PhysicalDomain.layout c.allocations) (ByteExecution.width c) key address) →
      ByteExecution.execute c artifact encode initial bytes address=bytes address) :=
  ⟨ByteExecution.execute_image c artifact encode separated initial bytes image,
    ByteExecution.execute_frame c artifact encode initial bytes⟩

/-- The same native execution is observable through actual Tensor metadata,
logical strides, pointers and requested element widths. -/
theorem fp_tensor {α : Type} [Inhabited α] (c : StorageMachine.Config α block)
    (artifact : StorageCompiler.Artifact c.profile c.flags c.N block)
    (encode : ByteExecution.Encode α)
    (separated : ByteExecution.Separated (PhysicalDomain.layout c.allocations) (ByteExecution.width c)
      (AliasMemory.Writes c.flags KernelEntry.regions c.strides c.M c.N))
    (initial : StorageMachine.Memory c) (bytes : ByteExecution.Bytes)
    (image : ByteExecution.Image (PhysicalDomain.layout c.allocations) (ByteExecution.width c) encode initial bytes)
    (tensor : ByteExecution.TensorView) (index : List Nat) (key : AliasMemory.Key)
    (binding : ByteExecution.Binds (PhysicalDomain.layout c.allocations) (ByteExecution.width c) tensor index key)
    (byte : Nat) (bound : byte<tensor.elementBytes) :
    tensor.bytes (ByteExecution.execute c artifact encode initial bytes) index byte=
      encode key (TypedPhysical.read (PhysicalDomain.layout c.allocations)
        (StorageMachine.execute c artifact initial) key) byte :=
  ByteExecution.tensor_readback _ _ encode _ _
    (ByteExecution.execute_image c artifact encode separated initial bytes image)
    tensor index key binding byte bound

/-- Compose the independent input-only reference with the actual Tensor byte
read. Both optional branches and masked columns are represented by `none`;
every issued store, including scalar statistics at N=0, returns its reference
encoding. No final-value or whole-output equality is a hypothesis. -/
theorem fp_tensor_output {α : Type} [Inhabited α] (c : StorageMachine.Config α block)
    (selectedTile : block=tile c.N)
    (artifact : StorageCompiler.Artifact c.profile c.flags c.N block)
    (encode : ByteExecution.Encode α)
    (separated : ByteExecution.Separated (PhysicalDomain.layout c.allocations) (ByteExecution.width c)
      (AliasMemory.Writes c.flags KernelEntry.regions c.strides c.M c.N))
    (initial : StorageMachine.Memory c) (bytes : ByteExecution.Bytes) (template : State α)
    (image : ByteExecution.Image (PhysicalDomain.layout c.allocations) (ByteExecution.width c) encode initial bytes)
    (row : Fin c.M) (destination : Compiler.Destination)
    (tensor : ByteExecution.TensorView) (index : List Nat)
    (binding : ByteExecution.Binds (PhysicalDomain.layout c.allocations) (ByteExecution.width c)
      tensor index (StorageMachine.key c (row,destination)))
    (byte : Nat) (bound : byte<tensor.elementBytes) :
    (artifact.stores destination).map
      (fun _ => tensor.bytes (ByteExecution.execute c artifact encode initial bytes) index byte)=
      (Compiler.storeCoordinate c.flags
        (StorageCompiler.reference c.profile c.ops c.casts artifact.graph.schedule c.flags
          KernelEntry.regions c.strides c.M c.N c.eps c.p
          (TypedPhysical.pull (PhysicalDomain.layout c.allocations) initial template)) row destination).map
        (fun value => encode (StorageMachine.key c (row,destination)) value byte) := by
  rw [fp_tensor c artifact encode separated initial bytes image tensor index _ binding byte bound]
  have coordinate := (StorageMachine.physical_store_coordinate c artifact
    (StorageMachine.execute c artifact initial) template row destination).trans
    (congrArg (fun result => Compiler.storeCoordinate c.flags result row destination)
      (fp_equiv c selectedTile artifact initial template))
  simpa only [Option.map_map] using congrArg
    (Option.map (fun value => encode (StorageMachine.key c (row,destination)) value byte)) coordinate

/-- Returned seeds and the residual fallback are input views. Their actual
Tensor bytes are preserved, as are all other input views, including read/read
aliases. This complements the issued-store Tensor theorem. -/
theorem fp_tensor_input {α : Type} [Inhabited α] (c : StorageMachine.Config α block)
    (artifact : StorageCompiler.Artifact c.profile c.flags c.N block)
    (encode : ByteExecution.Encode α)
    (separated : ByteExecution.Separated (PhysicalDomain.layout c.allocations) (ByteExecution.width c)
      (AliasMemory.Writes c.flags KernelEntry.regions c.strides c.M c.N))
    (initial : StorageMachine.Memory c) (bytes : ByteExecution.Bytes)
    (image : ByteExecution.Image (PhysicalDomain.layout c.allocations) (ByteExecution.width c) encode initial bytes)
    (region : RegionName) (input : StorageMachine.inputRegion region) (offset : Nat)
    (tensor : ByteExecution.TensorView) (index : List Nat)
    (binding : ByteExecution.Binds (PhysicalDomain.layout c.allocations) (ByteExecution.width c)
      tensor index (region,offset)) (byte : Nat) (bound : byte<tensor.elementBytes) :
    tensor.bytes (ByteExecution.execute c artifact encode initial bytes) index byte=
      tensor.bytes bytes index byte := by
  rw [fp_tensor c artifact encode separated initial bytes image tensor index _ binding byte bound]
  rw [StorageMachine.execute,StorageMachine.run_input_read c artifact _ initial region input offset]
  exact (ByteExecution.tensor_readback _ _ encode initial bytes image tensor index _ binding byte bound).symm

/-- At zero columns only full statistics byte spans may change. In RMS mode
the mean span is also framed. All masked column destinations are preserved. -/
theorem fp_zero_columns_frame {α : Type} [Inhabited α] (c : StorageMachine.Config α block)
    (zero : c.N=0) (artifact : StorageCompiler.Artifact c.profile c.flags c.N block)
    (encode : ByteExecution.Encode α) (initial : StorageMachine.Memory c) (bytes : ByteExecution.Bytes)
    (address : Nat)
    (outside : ∀ row : Fin c.M,
      ¬ByteExecution.Span (PhysicalDomain.layout c.allocations) (ByteExecution.width c)
        (KernelEntry.regions.rstd,row.val) address ∧
      (c.flags.rms=Bool.false → ¬ByteExecution.Span (PhysicalDomain.layout c.allocations)
        (ByteExecution.width c) (KernelEntry.regions.mean,row.val) address)) :
    ByteExecution.execute c artifact encode initial bytes address=bytes address := by
  apply ByteExecution.execute_frame
  intro key written
  obtain ⟨row,bound,written⟩ := written
  rw [zero,zero_columns_writes] at written
  rcases written with ⟨region,offset⟩ | ⟨layerNorm,region,offset⟩
  · simpa [←region,←offset] using (outside ⟨row,bound⟩).1
  · simpa [←region,←offset] using (outside ⟨row,bound⟩).2 layerNorm

#axiomsClean legacy_correct_physical
#axiomsClean correct_physical
#axiomsClean fp_bytes
#axiomsClean fp_tensor
#axiomsClean fp_tensor_output
#axiomsClean fp_tensor_input
#axiomsClean fp_zero_columns_frame
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TensorBridge
