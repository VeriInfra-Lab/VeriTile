import bench.tritonbench_g.layer_norm_ops.Proofs.ForwardModel

namespace VeriTile.Bench.Repairs.LayerNormOps.Forward
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits (fold_writes_mem)
noncomputable section
set_option pp.maxSteps 1200
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

@[simp] theorem fold_writes_undef {α ι : Type} (xs : List ι)
    (active : ι → Prop) [DecidablePred active]
    (region : ι → RegionName) (offset : ι → Nat) (value : ι → Cell α)
    (s : State α) :
    (xs.foldl (fun t i => if active i then t.write (region i) (offset i) (value i)
      else t) s).undef = s.undef := by
  induction xs generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons, ih]
    split <;> rfl

@[simp] theorem vector_bop {α β γ : Type} {n : Nat}
    (f : α → β → γ) (a : TileIndex [n] → α) (b : TileIndex [n] → β) :
    bop f Broadcast.nil.consSame a b = (fun i => f (a i) (b i)) := by
  funext i
  rcases i with ⟨i,⟨⟩⟩
  rfl

theorem region_name_eq (r q : RegionName) : (r.name = q.name) ↔ r = q := by
  cases r; cases q; simp


@[simp] theorem write_mem {α : Type} (s : State α) (r : RegionName) (o : Nat) (v : Cell α) :
    (s.write r o v).mem = fun r' o' => if r'=r ∧ o'=o then v else s.mem r' o' := rfl

end
end VeriTile.Bench.Repairs.LayerNormOps.Forward
