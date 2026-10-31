import bench.tritonbench_g.triton_argmax.Proofs.TypedLoop

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.Typed
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem fold_memory {α ι : Type} (l : List ι) (r : RegionName)
    (off : ι → Nat) (value : ι → Cell α) (active : ι → Prop) [DecidablePred active] (s : State α) :
    (l.foldl (fun t i => if active i then t.write r (off i) (value i) else t) s).mem =
      l.foldl (fun mem i => fun r' o =>
        if active i ∧ r'=r ∧ o=off i then value i else mem r' o) s.mem := by
  induction l generalizing s with
  | nil => rfl
  | cons i l ih =>
    by_cases hi : active i
    · simp only [List.foldl_cons,if_pos hi,ih]
      congr 1
      funext r' o
      simp [hi,State.write]
    · simp only [List.foldl_cons,if_neg hi,ih]
      congr 1
      funext r' o
      simp [hi]

def written (s : State Word) (O : Region .int) (M K BM row inner : Nat)
    (current : Fin BM → Word × Nat) : State Word :=
  (TileShape.allIndices [BM]).foldl (fun t i =>
    if row+i.1.val<M then t.write (Region.cast O) ((row+i.1.val)*K+inner)
      (.mk .int (Int.ofNat (current i.1).2)) else t) s

theorem written_memory (s initial : State Word) (same : s.mem=initial.mem) (O : Region .int)
    (M K BM row inner : Nat) (current : Fin BM → Word × Nat) :
    (written s O M K BM row inner current).mem=(written initial O M K BM row inner current).mem := by
  simp only [written,fold_memory,same]

end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.Typed
