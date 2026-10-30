import bench.tritonbench_g.quantize_kv_transform.Proofs.KernelGridResult

namespace VeriTile.Bench.Repairs.QuantizeKvTransform.KernelInterface
open VeriTile.Triton FP.Structural KernelSource
noncomputable section

/-- Every active code/scale cell and the full exterior frame, over raw kernel
geometry and destination indices. The targets contain no execution result. -/
def Result {α : Type} (p : Parameters) (count : Nat) (dest : Nat → Nat)
    (codes : Nat → TileIndex [p.height, p.width] → Int)
    (scales : Nat → Fin p.height → α) (initial final : State α) : Prop :=
  (∀ token (i : TileIndex [p.height, p.width]), token < count → KernelResult.active p i →
    final.mem "output" (KernelResult.codeOffset p (dest token) i.1.val i.2.1.val) =
      .mk .int (codes token i)) ∧
  (∀ token (head : Fin p.height), token < count → head.val < p.heads →
    final.mem "scale" (KernelResult.scaleOffset p (dest token) head.val) =
      .mk .real (scales token head)) ∧
  (∀ (r : RegionName) o,
    (r ≠ "output" ∨ ∀ token head column, token < count → head < p.heads → column < p.columns →
      o ≠ KernelResult.codeOffset p (dest token) head column) →
    (r ≠ "scale" ∨ ∀ token head, token < count → head < p.heads →
      o ≠ KernelResult.scaleOffset p (dest token) head) →
    final.mem r o = initial.mem r o)

/-- Assemble the existing complete per-program execution, initial-input
recurrence and grid frame. No execution or output premise is required. -/
theorem complete {α : Type} [Inhabited α] (M : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (p : Parameters) (hw : 0 < p.width)
    (count : Nat) (dest : Nat → Nat) (initial : State α)
    (metadata : ∀ token, token < count →
      (initial.mem "destination" token).read .int = (dest token : Int))
    (unique : ∀ a b, a < count → b < count → dest a = dest b → a = b)
    (order : List Nat) (schedule : order.Perm (List.range count)) :
    ∃ final, KernelGrid.run (KernelExecution.backend M convert equal) p order initial = some final ∧
      Result p count dest
        (fun token => KernelResult.code M convert equal p (KernelGrid.atProgram initial token))
        (fun token => KernelResult.scale M p (KernelGrid.atProgram initial token)) initial final := by
  have valid : ∀ token ∈ order, token < count :=
    fun _ h => List.mem_range.mp (schedule.mem_iff.mp h)
  have nodup := schedule.nodup_iff.mpr List.nodup_range
  refine ⟨KernelGridResult.finalState M convert equal p hw order initial,
    KernelGridResult.run_finalState M convert equal p hw order initial, ?_, ?_, ?_⟩
  · intro token i ht hi
    exact KernelGridResult.final_code M convert equal p hw initial count dest metadata unique
      order valid nodup token (schedule.mem_iff.mpr (List.mem_range.mpr ht)) i hi
  · intro token head ht hh
    exact KernelGridResult.final_scale M convert equal p hw initial count dest metadata unique
      order valid nodup token (schedule.mem_iff.mpr (List.mem_range.mpr ht))
      (head, ⟨0, by decide⟩, PUnit.unit) hh
  · intro r o codes scales
    exact KernelGridResult.final_frame M convert equal p hw initial count dest metadata order valid
      r o codes scales initial ⟨fun _ => rfl, fun _ => rfl⟩

#axiomsClean complete
#specNonCircular Result avoiding [KernelSource.kernel, FP.Structural.exec, KernelGrid.run]
end
end VeriTile.Bench.Repairs.QuantizeKvTransform.KernelInterface
