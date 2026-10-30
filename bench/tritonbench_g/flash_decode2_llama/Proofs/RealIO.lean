import bench.tritonbench_g.flash_decode2_llama.Proofs.PublicIO

namespace VeriTile.Bench.Repairs.FlashDecode2Llama.RealIO
open VeriTile.Triton FP.Structural FlashDecode2 KernelInterface
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000

def observe (x : Option ℝ) : ℝ := x.getD 0

theorem value (g : KernelInterface.Geometry) (order : List Nat)
    (vals : IO.Inputs ℝ g) (domain : IO.Lengths g vals) (s : State (Option ℝ))
    (pins : (IO.decodeIO Reference.ideal g order some observe).InputBinding vals s)
    (row : Fin (g.B * g.H)) (dim : Fin g.D) :
    observe (Grid.value Reference.ideal s Bool.false g.B g.H g.S g.D g.Q g.D row.val
      (dim, PUnit.unit)) = Reference.target (IO.logValues vals row.val) (IO.midValues vals row.val)
        ((IO.lengths vals (row.val / g.H) + g.Q - 1) / g.Q) dim.val := by
  have bounds := PublicIO.lengths_bound Reference.ideal g order some observe vals domain s pins
  have counted := KernelInterface.count_bound g s bounds row.val row.isLt
  let entry := Grid.setup g.B g.H row.val s
  have hx : ∀ j, j < count entry g.Q →
      logvalue Reference.ideal entry g.H g.S j = some (IO.logValues vals row.val j) := by
    intro j hj
    have split : j < g.S := lt_of_lt_of_le (counted.1 ▸ hj) counted.2.2
    simp only [logvalue, entry, row_log_offset, Reference.ideal]
    exact PublicIO.log_binding Reference.ideal g order some observe vals s pins row.val j row.isLt split
  have hv : ∀ j, j < count entry g.Q → ∀ k : TileIndex [g.D],
      FlashDecode2.value Reference.ideal entry Bool.false g.H g.S g.D j k =
        some (IO.midValues vals row.val j (lane k)) := by
    intro j hj k
    have split : j < g.S := lt_of_lt_of_le (counted.1 ▸ hj) counted.2.2
    simp only [FlashDecode2.value, active, true_or, ↓reduceIte, entry, row_mid_offset, Reference.ideal]
    exact PublicIO.mid_binding Reference.ideal g order some observe vals s pins
      row.val j (lane k) row.isLt split k.1.isLt
  have result := Reference.source_correct entry Bool.false g.H g.S g.D g.Q g.D
    (IO.logValues vals row.val) (fun j k => IO.midValues vals row.val j (lane k))
    (counted.1.symm ▸ counted.2.1) hx hv (dim, PUnit.unit)
  change observe (output Reference.ideal entry Bool.false g.H g.S g.D g.Q g.D (dim, PUnit.unit)) = _
  rw [result]
  change Reference.target _ _ (count entry g.Q) _ = _
  rw [counted.1]
  have length := PublicIO.length_binding Reference.ideal g order some observe vals s pins
    (row.val / g.H) (PublicIO.batch_bound g row.val row.isLt)
  simp only [KernelInterface.steps, length]
  rfl

theorem implements (g : KernelInterface.Geometry) (order : List Nat)
    (coverage : order.Perm (List.range (g.B * g.H))) :
    (IO.decodeIO Reference.ideal g order some observe).ImplementsOn (IO.Lengths g)
      (IO.mathematical g) :=
  PublicIO.implements Reference.ideal g order coverage some observe _ (value g order)

#axiomsClean implements
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FlashDecode2Llama.RealIO
