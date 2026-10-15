import bench.tritonbench_g.flash_decode2_llama.Proofs.FPSource

/-! Independent launch-memory output and frame contracts. The only readback
condition is the original output-address injectivity. Memory equality itself
also covers collisions, in the modeled store order, without new assumptions.
-/
namespace VeriTile.Bench.TritonBenchG.FlashDecode2Llama.FPStages
open VeriTile.Triton FP.Structural
open FusedRecurrentHgrn.FPStages (scatter)
noncomputable section

def Post {α : Type} (s t : State α) (O : RegionName) (o : Row) (D : Nat)
    (values : TileIndex [D] → α) : Prop :=
  (∀ i, t.mem O (address s o i) = .mk .real (values i)) ∧
  (∀ R off, (R ≠ O ∨ ∀ i : TileIndex [D], off ≠ address s o i) →
    t.mem R off = s.mem R off)

theorem post_of_memory {α : Type} (s t : State α) (O : RegionName) (o : Row) (D : Nat)
    (v : TileIndex [D] → α) (hinj : Function.Injective (address s o (D := D)))
    (hm : t.mem = (scatter s O D (address s o) (fun _ => True) v).mem) :
    Post s t O o D v := by
  constructor
  · intro i
    rw [hm]
    exact State.masked_scatter_readback O (address s o) (fun j => .mk .real (v j))
      (fun _ => True) s i trivial hinj
  · intro R off h
    rw [hm]
    apply State.masked_scatter_frame
    rcases h with h | h
    · exact Or.inl h
    · exact Or.inr fun i _ _ => h i

def ScalarPost {α : Type} (s t : State α) (Next : RegionName) (m : Scalar) (v : α) : Prop :=
  t.mem Next (scalarAddress s m) = .mk .real v ∧
  (∀ R off, R ≠ Next ∨ off ≠ scalarAddress s m → t.mem R off = s.mem R off)

theorem scalar_post {α : Type} (s t : State α) (Next : RegionName) (m : Scalar) (v : α)
    (hm : t.mem = (s.write Next (scalarAddress s m) (.mk .real v)).mem) :
    ScalarPost s t Next m v := by
  constructor
  · simp [hm, State.write]
  · intro R off h
    rw [hm]
    simp only [State.write]
    have hn : ¬ (R = Next ∧ off = scalarAddress s m) := by
      rcases h with h | h
      · exact fun he => h he.1
      · exact fun he => h he.2
    simp [hn]

#axiomsClean post_of_memory
#axiomsClean scalar_post
end
end VeriTile.Bench.TritonBenchG.FlashDecode2Llama.FPStages
