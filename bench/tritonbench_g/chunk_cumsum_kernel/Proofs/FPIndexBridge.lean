import bench.tritonbench_g.chunk_cumsum_kernel.Proofs.FPRow

/-! Compatibility names for proof fragments extracted from the repair itself.
There is no second whole program or execution bridge. -/
namespace VeriTile.Bench.Repairs.ChunkCumsumKernel.FPIndexBridge
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section

abbrev parts (input output : RegionName) (T BT : Nat) :=
  FPRow.parts .fp32 .fp32 input output T BT

theorem before (input output : RegionName) (T BT : Nat) :
    (parts input output T BT).before=(FPRow.parts .fp32 .fp32 input output T BT).before := rfl

theorem masked_mem {α ι : Type} (r : RegionName) (off : ι → Nat)
    (val : ι → Cell α) (active : ι → Prop) [DecidablePred active]
    (l : List ι) (s t : State α) (hm : s.mem=t.mem) :
    (l.foldl (fun t i => if active i then t.write r (off i) (val i) else t) s).mem=
    (l.foldl (fun t i => if active i then t.write r (off i) (val i) else t) t).mem := by
  induction l generalizing s t with
  | nil => exact hm
  | cons i rest ih =>
    simp only [List.foldl_cons]
    apply ih
    by_cases h : active i
    · simp only [if_pos h,State.write,hm]
    · simpa only [if_neg h] using hm

abbrev split_source := FPRow.split_source .fp32 .fp32
theorem complete {α : Type} [Inhabited α] (ops : FPRow.Ops α)
    (input output : RegionName) (T BT : Nat) (s : State α) :
    ∃ t, FP.Structural.exec (engine ops.algebra) (repairedKernel .fp32 .fp32 input output T BT) s=some t ∧
      FPRow.Frame output (s.pids 0) T s t :=
  FPRow.complete ops input output T BT s

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkCumsumKernel.FPIndexBridge
