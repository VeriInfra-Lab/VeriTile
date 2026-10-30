import bench.repairs.tritonbench_g.triton_argmax.Proofs.DimMemory
import bench.repairs.tritonbench_g.triton_argmax.Proofs.RealDim

/-! Independent Real projection: logical input values are arbitrary Reals,
never encoded into FP32 bytes. Both source reduction stages/full dimension
loop are interpreted over Real-with-bottom; output indices alone occupy i64
memory. The raw FP physical execution is connected separately in FPEquiv. -/
namespace VeriTile.Bench.Repairs.TritonArgmax.RealMemory
open Source Memory Reduction
noncomputable section

def globalRun {M : Type} (mem : Primitives M) (c : GlobalConfig)
    (logical : Nat → ℝ) (output : Allocation) (s : M) : M :=
  mem.write 8 s (address output 8 0) (BitVec.ofNat 64 (Math.globalIndex c logical))

theorem global_returned {M : Type} (mem : Primitives M) (c : GlobalConfig)
    (coverage : Math.GlobalCoverage c) (logical : Nat → ℝ) (output : Allocation) (s : M) :
    mem.read 8 (globalRun mem c logical output s) (address output 8 0)=
      BitVec.ofNat 64 (Math.argmax c.size coverage.sizePositive (fun i : Fin c.size => logical i.val)).val := by
  rw [globalRun,mem.readSame]
  have unique := least_maximum_unique (fun i : Fin c.size => logical i.val)
    ⟨Math.globalIndex c logical,Math.global_valid c coverage logical⟩
    (Math.argmax c.size coverage.sizePositive (fun i : Fin c.size => logical i.val))
    (Math.global_mathematical c coverage logical) (Math.argmax_spec _ _ _)
  exact congrArg (fun i : Fin c.size => BitVec.ofNat 64 i.val) unique

theorem global_frame {M : Type} (mem : Primitives M) (c : GlobalConfig)
    (logical : Nat → ℝ) (output : Allocation) (s : M) (point : Address)
    (apart : Disjoint (address output 8 0) 8 point 1) :
    mem.byte (globalRun mem c logical output s) point=mem.byte s point :=
  mem.byteFrame _ _ _ _ _ apart

def rowLogical (c : DimConfig) (logical : Nat → ℝ) (j : DimMemory.Job) : Nat → ℝ :=
  fun col => logical (dimInputOffset c j.1 col j.2)

def rowIndex (c : DimConfig) (logical : Nat → ℝ) (j : DimMemory.Job) : Nat :=
  Math.dimensionIndex c.extent 128 (by omega) (rowLogical c logical j)
    (fun cur => c.tree j.1 j.2 (cur/128))

def dimensionRun {M : Type} (mem : Primitives M) (c : DimConfig) (logical : Nat → ℝ)
    (output : Allocation) (s : M) : M :=
  stores mem output 8 (DimMemory.key c) (fun j => BitVec.ofNat 64 (rowIndex c logical j))
    ((DimMemory.jobs c).filter fun j => j.1 < c.outer) s

theorem dimension_source_readback {M : Type} (mem : Primitives M) (c : DimConfig)
    (logical : Nat → ℝ) (output : Allocation) (s : M) (row inner : Nat)
    (hr : row < c.outer) (hi : inner < c.inner) :
    mem.read 8 (dimensionRun mem c logical output s) (address output 8 (dimOutputOffset c row inner))=
      BitVec.ofNat 64 (rowIndex c logical (row,inner)) := by
  apply readback mem output 8 (by omega) (DimMemory.key c)
    (fun j => BitVec.ofNat 64 (rowIndex c logical j)) _ s (row,inner)
  · simp only [List.mem_filter,decide_eq_true_eq]
    exact ⟨DimMemory.job_coverage c row inner hr hi,hr⟩
  · intro j member same
    have hi' := DimMemory.job_inner c j (List.mem_filter.mp member).1
    have pair := Native.output_unique c.inner j.1 j.2 row inner hi' hi same
    obtain ⟨r,k⟩ := j
    obtain ⟨rfl,rfl⟩ := pair
    rfl

theorem dimension_returned {M : Type} (mem : Primitives M) (c : DimConfig)
    (logical : Nat → ℝ) (output : Allocation) (s : M) (row inner : Nat)
    (hr : row < c.outer) (hi : inner < c.inner) (nonempty : 0 < c.extent)
    (complete : ∀ tile,(c.tree row inner tile).Complete) :
    mem.read 8 (dimensionRun mem c logical output s) (address output 8 (dimOutputOffset c row inner))=
      BitVec.ofNat 64 (Math.argmax c.extent nonempty
        (fun i : Fin c.extent => rowLogical c logical (row,inner) i.val)).val := by
  rw [dimension_source_readback mem c logical output s row inner hr hi]
  obtain ⟨bound,correct⟩ := Math.dimension_mathematical c.extent 128 nonempty (by omega)
    (rowLogical c logical (row,inner)) (fun cur => c.tree row inner (cur/128)) (fun cur _ => complete _)
  have unique := least_maximum_unique (fun i : Fin c.extent => rowLogical c logical (row,inner) i.val)
    ⟨_,bound⟩ (Math.argmax c.extent nonempty (fun i : Fin c.extent => rowLogical c logical (row,inner) i.val))
    correct (Math.argmax_spec _ _ _)
  exact congrArg (fun i : Fin c.extent => BitVec.ofNat 64 i.val) unique

theorem dimension_frame {M : Type} (mem : Primitives M) (c : DimConfig)
    (logical : Nat → ℝ) (output : Allocation) (s : M) (point : Address)
    (apart : ∀ row inner,row < c.outer → inner < c.inner →
      Disjoint (address output 8 (dimOutputOffset c row inner)) 8 point 1) :
    mem.byte (dimensionRun mem c logical output s) point=mem.byte s point := by
  apply bytes_frame
  intro j member
  have pair := List.mem_filter.mp member
  exact apart j.1 j.2 (by simpa using pair.2) (DimMemory.job_inner c j pair.1)

#axiomsClean global_returned
#axiomsClean global_frame
#axiomsClean dimension_returned
#axiomsClean dimension_frame
end
end VeriTile.Bench.Repairs.TritonArgmax.RealMemory
