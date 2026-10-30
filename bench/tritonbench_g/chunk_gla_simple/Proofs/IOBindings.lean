import bench.tritonbench_g.chunk_gla_simple.Targets

namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.IOBindings
open VeriTile Triton Host PublicIO
open VeriTile.Bench.TritonBenchG.ChunkGlaSimple Math Real
noncomputable section
set_option maxHeartbeats 1600000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem capacity_bounds (c : Shape) (q : GridIndex (grid c)) (bounds : RegionBounds)
    (h : ∀ r, capacity c r ≤ bounds r) :
    Safe.Bounds (layout c) (launchIndex c q) regions bounds :=
  Host.bounds c q ⟨"unused",[],fun _ => 0,bounds⟩ h

theorem extent_le {n b : Nat} (h : ∀ i : Fin n, i.val<b) : n≤b := by
  by_contra hn
  exact Nat.lt_irrefl b (h ⟨b,by omega⟩)

theorem real_capacities (c : Shape) (scale : ℝ) (xs : (realIO c scale).Ctx)
    (s : BlockState) (bounds : RegionBounds) (h : (realIO c scale).RegionalBounds bounds s xs) :
    ∀ r, capacity c r ≤ bounds r := by
  have port (i : Fin 5) : inputSize c i ≤ bounds (inputRegion i) :=
    extent_le (fun j => h.1 i j trivial)
  have output : c.heads*c.T*c.V ≤ bounds "o" :=
    extent_le (fun j => h.2.1 (0 : Fin 1) j trivial)
  intro r
  by_cases hq : r="q"
  · subst r; simpa [capacity,inputSize,inputRegion] using port 0
  by_cases hk : r="k"
  · subst r; simpa [capacity,inputSize,inputRegion] using port 1
  by_cases hv : r="v"
  · subst r; simpa [capacity,inputSize,inputRegion] using port 2
  by_cases hh : r="h"
  · subst r; simpa [capacity,inputSize,inputRegion] using port 3
  by_cases hg : r="g"
  · subst r; simpa [capacity,inputSize,inputRegion] using port 4
  by_cases ho : r="o"
  · subst r; simpa [capacity] using output
  simp [capacity,hq,hk,hv,hh,hg,ho]

theorem real_read (c : Shape) (scale : ℝ) (xs : (realIO c scale).Ctx) (s : BlockState)
    (h : (realIO c scale).RegionalInputs s xs) (i : Fin 5) (off : Nat)
    (hi : off < inputSize c i) :
    s.readMem (inputRegion i) off = PublicMath.extend (0 : ℝ) (xs i) off := by
  simpa only [realIO,ChanTy.read,PublicMath.extend,hi,↓reduceDIte]
    using h i ⟨off,hi⟩ trivial

/-- The independent formula depends only on its declared finite input
windows, including zero padding and every K/time/value boundary. -/
theorem expected_congr (c : Shape) (q : GridIndex (grid c)) (a b : Inputs)
    (query : ∀ off, off<c.heads*c.T*c.K → a.query off=b.query off)
    (key : ∀ off, off<c.heads*c.T*c.K → a.key off=b.key off)
    (value : ∀ off, off<c.heads*c.T*c.V → a.value off=b.value off)
    (history : ∀ off, off<c.heads*c.NT*c.K*c.V → a.history off=b.history off)
    (gate : ∀ off, off<c.heads*c.T → a.gate off=b.gate off)
    (scale : ℝ) (i j : Nat) :
    expected (layout c) (launchIndex c q) a scale i j =
      expected (layout c) (launchIndex c q) b scale i j := by
  obtain ⟨hv,ht,hh⟩ := launch_lt c q
  have qeq (i k : Nat) : Math.query (layout c) (launchIndex c q) a i k =
      Math.query (layout c) (launchIndex c q) b i k := by
    unfold Math.query
    split_ifs with h
    · exact query _ (by simpa [keyAddress,layout] using triple_lt hh h.1 h.2)
    · rfl
  have keq (k j : Nat) : Math.key (layout c) (launchIndex c q) a k j =
      Math.key (layout c) (launchIndex c q) b k j := by
    unfold Math.key
    split_ifs with h
    · apply key
      have bound := triple_lt hh h.2 h.1
      dsimp [layout] at bound ⊢
      omega
    · rfl
  have heq (k j : Nat) : Math.history (layout c) (launchIndex c q) a k j =
      Math.history (layout c) (launchIndex c q) b k j := by
    unfold Math.history
    split_ifs with h
    · apply history
      have bound := pair_lt (pair_lt (pair_lt hh ht) h.1) h.2
      dsimp [historyAddress,layout] at bound ⊢
      nlinarith
    · rfl
  have veq (i j : Nat) : Math.value (layout c) (launchIndex c q) a i j =
      Math.value (layout c) (launchIndex c q) b i j := by
    unfold Math.value
    split_ifs with h
    · exact value _ (by simpa [outputAddress,layout] using triple_lt hh h.1 h.2)
    · rfl
  have geq (i : Nat) : Math.gate (layout c) (launchIndex c q) a i =
      Math.gate (layout c) (launchIndex c q) b i := by
    unfold Math.gate
    split_ifs with h
    · exact gate _ (by simpa [gateAddress,layout] using pair_lt hh h)
    · rfl
  simp only [expected,qhSum,qkSum,score,qeq,keq,heq,veq,geq]

theorem real_formula (c : Shape) (bt : 0<c.BT) (scale : ℝ)
    (xs : (realIO c scale).Ctx) (s : BlockState) (h : (realIO c scale).RegionalInputs s xs)
    (head : Fin c.heads) (row : Fin c.T) (value : Fin c.V) :
    Host.formula c bt (inputs regions s) scale head row value =
      Host.formula c bt (PublicMath.arrays c scale xs) scale head row value := by
  apply expected_congr
  · exact fun off hi => real_read c scale xs s h (0 : Fin 5) off hi
  · exact fun off hi => real_read c scale xs s h (1 : Fin 5) off hi
  · exact fun off hi => real_read c scale xs s h (2 : Fin 5) off hi
  · exact fun off hi => real_read c scale xs s h (3 : Fin 5) off hi
  · exact fun off hi => real_read c scale xs s h (4 : Fin 5) off hi

theorem output_index (c : Shape) (i : Fin (c.heads*c.T*c.V)) :
    i.divNat.divNat.val*(c.T*c.V)+i.divNat.modNat.val*c.V+i.modNat.val = i.val := by
  have h1 := Nat.div_add_mod i.val c.V
  have h2 := Nat.div_add_mod (i.val/c.V) c.T
  dsimp [Fin.divNat,Fin.modNat]
  nlinarith

end
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.IOBindings
