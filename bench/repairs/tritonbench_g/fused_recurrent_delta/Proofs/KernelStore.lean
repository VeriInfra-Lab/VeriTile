import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelMemory
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelStore
open VeriTile.Triton FP.Structural KernelProgram KernelRegisters KernelMemory
noncomputable section
variable {α : Type} [Inhabited α]

theorem masked_mem {ι : Type} (r : RegionName) (off : ι → Nat) (value : ι → Cell α)
    (active : ι → Bool) (xs : List ι) (u v : State α) (same : u.mem=v.mem) :
    (xs.foldl (fun w i => if active i then w.write r (off i) (value i) else w) u).mem=
      (xs.foldl (fun w i => if active i then w.write r (off i) (value i) else w) v).mem := by
  induction xs generalizing u v with
  | nil => exact same
  | cons i rest ih =>
    simp only [List.foldl_cons]
    apply ih
    split <;> simp_all [State.write]

@[simp] theorem masked_setReg {ι : Type} (r : RegionName) (off : ι → Nat) (value : ι → Cell α)
    (active : ι → Bool) (xs : List ι) (u : State α) (name : RegName) (d : TileDType)
    (shape : TileShape) (values : Values α d shape) :
    (xs.foldl (fun w i => if active i then w.write r (off i) (value i) else w)
      (u.setReg name d shape values)).mem=
      (xs.foldl (fun w i => if active i then w.write r (off i) (value i) else w) u).mem :=
  masked_mem r off value active xs _ _ rfl

#axiomsClean masked_mem
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelStore
