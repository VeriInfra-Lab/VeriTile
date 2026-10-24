import bench.repairs.tritonbench_g.chunk_gla_simple_min_dot_tile.Proofs.IOBindings

namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.IONativeBindings
open VeriTile Triton FP.Structural Host PublicIO IOBindings
open NativeGraph (Expr)
noncomputable section
set_option maxHeartbeats 1600000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def memoryArrays (s : State Expr) : PublicNative.Arrays :=
  ⟨fun i => (s.mem "q" i).read .real,fun i => (s.mem "k" i).read .real,
   fun i => (s.mem "v" i).read .real,fun i => (s.mem "h" i).read .real,
   fun i => (s.mem "g" i).read .real⟩

theorem output_expression (c : Shape) (s : State Expr) (i : FPValue.OIndex c) :
    FPValue.outputValue ScalarGraph.algebra c (ScalarGraph.dots c) s ScalarGraph.scaleTag i =
      PublicNative.expression c (memoryArrays s) (s.pids 2) (s.pids 1) (s.pids 0) i := rfl

theorem expression_congr (c : Shape) (q : GridIndex (grid c)) (a b : PublicNative.Arrays)
    (query : ∀ off, off<c.heads*c.T*c.K → a.query off=b.query off)
    (key : ∀ off, off<c.heads*c.T*c.K → a.key off=b.key off)
    (value : ∀ off, off<c.heads*c.T*c.V → a.value off=b.value off)
    (history : ∀ off, off<c.heads*c.NT*c.K*c.V → a.history off=b.history off)
    (gate : ∀ off, off<c.heads*c.T → a.gate off=b.gate off)
    (i : FPValue.OIndex c) :
    PublicNative.expression c a (launchIndex c q).head (launchIndex c q).chunk (launchIndex c q).value i =
      PublicNative.expression c b (launchIndex c q).head (launchIndex c q).chunk (launchIndex c q).value i := by
  obtain ⟨hv,ht,hh⟩ := launch_lt c q
  have qeq (block : Nat) :
      PublicNative.query c a (launchIndex c q).head (launchIndex c q).chunk block =
      PublicNative.query c b (launchIndex c q).head (launchIndex c q).chunk block := by
    funext j
    unfold PublicNative.query
    congr 1
    split_ifs with h
    · exact query _ (triple_lt hh h.1 h.2)
    · rfl
  have keq (block : Nat) :
      PublicNative.keyValue c a (launchIndex c q).head (launchIndex c q).chunk block =
      PublicNative.keyValue c b (launchIndex c q).head (launchIndex c q).chunk block := by
    funext j
    unfold PublicNative.keyValue
    congr 1
    split_ifs with h
    · apply key
      have bound := triple_lt hh h.2 h.1
      omega
    · rfl
  have heq (block : Nat) :
      PublicNative.history c a (launchIndex c q).head (launchIndex c q).chunk (launchIndex c q).value block =
      PublicNative.history c b (launchIndex c q).head (launchIndex c q).chunk (launchIndex c q).value block := by
    funext j
    unfold PublicNative.history
    congr 1
    split_ifs with h
    · apply history
      have bound := pair_lt (pair_lt (pair_lt hh ht) h.1) h.2
      nlinarith
    · rfl
  have veq : PublicNative.value c a (launchIndex c q).head (launchIndex c q).chunk (launchIndex c q).value =
      PublicNative.value c b (launchIndex c q).head (launchIndex c q).chunk (launchIndex c q).value := by
    funext j
    unfold PublicNative.value
    congr 1
    split_ifs with h
    · exact value _ (triple_lt hh h.1 h.2)
    · rfl
  have geq : PublicNative.gate c a (launchIndex c q).head (launchIndex c q).chunk =
      PublicNative.gate c b (launchIndex c q).head (launchIndex c q).chunk := by
    funext j
    unfold PublicNative.gate
    congr 1
    split_ifs with h
    · exact gate _ (pair_lt hh h)
    · rfl
  have hu : PublicNative.update c a (launchIndex c q).head (launchIndex c q).chunk (launchIndex c q).value =
      PublicNative.update c b (launchIndex c q).head (launchIndex c q).chunk (launchIndex c q).value := by
    funext block acc
    simp only [PublicNative.update,qeq,keq,heq]
  simp only [PublicNative.expression,hu,veq,geq]

theorem formula_binding (c : Shape) (bt : 0<c.BT) (xs : (nativeIO c).Inputs)
    (s : State Expr) (hin : (nativeIO c).InputBinding xs s)
    (h : Fin c.heads) (t : Fin c.T) (v : Fin c.V) :
    FPValue.outputValue ScalarGraph.algebra c (ScalarGraph.dots c)
      (FPGrid.launch c s (owner c bt h t v)) ScalarGraph.scaleTag (lane c bt t v) =
      PublicNative.expression c (PublicNative.arrays c xs) h.val (t.val/c.BT) (v.val/c.BV)
        (lane c bt t v) := by
  rw [output_expression]
  have he := expression_congr c (owner c bt h t v) (memoryArrays s) (PublicNative.arrays c xs)
  apply he
  · intro off hi
    simpa only [nativeIO,nativeInputs,ProgramIO.InputBinding,mapValue,id_eq,
      memoryArrays,PublicNative.arrays,PublicMath.extend,hi,↓reduceDIte]
      using hin (0 : Fin 5) (⟨off,hi⟩,()) trivial
  · intro off hi
    simpa only [nativeIO,nativeInputs,ProgramIO.InputBinding,mapValue,id_eq,
      memoryArrays,PublicNative.arrays,PublicMath.extend,hi,↓reduceDIte]
      using hin (1 : Fin 5) (⟨off,hi⟩,()) trivial
  · intro off hi
    simpa only [nativeIO,nativeInputs,ProgramIO.InputBinding,mapValue,id_eq,
      memoryArrays,PublicNative.arrays,PublicMath.extend,hi,↓reduceDIte]
      using hin (2 : Fin 5) (⟨off,hi⟩,()) trivial
  · intro off hi
    simpa only [nativeIO,nativeInputs,ProgramIO.InputBinding,mapValue,id_eq,
      memoryArrays,PublicNative.arrays,PublicMath.extend,hi,↓reduceDIte]
      using hin (3 : Fin 5) (⟨off,hi⟩,()) trivial
  · intro off hi
    simpa only [nativeIO,nativeInputs,ProgramIO.InputBinding,mapValue,id_eq,
      memoryArrays,PublicNative.arrays,PublicMath.extend,hi,↓reduceDIte]
      using hin (4 : Fin 5) (⟨off,hi⟩,()) trivial

theorem capacities (c : Shape) (xs : (nativeIO c).Inputs) (bounds : RegionName → Nat)
    (h : (nativeIO c).InBounds bounds xs) : ∀ r, capacity c r ≤ bounds r := by
  have hq : c.heads*c.T*c.K ≤ bounds "q" := extent_le (fun j => h.2.1 (0 : Fin 5) (j,()) trivial)
  have hk : c.heads*c.T*c.K ≤ bounds "k" := extent_le (fun j => h.2.1 (1 : Fin 5) (j,()) trivial)
  have hv : c.heads*c.T*c.V ≤ bounds "v" := extent_le (fun j => h.2.1 (2 : Fin 5) (j,()) trivial)
  have hh : c.heads*c.NT*c.K*c.V ≤ bounds "h" := extent_le (fun j => h.2.1 (3 : Fin 5) (j,()) trivial)
  have hg : c.heads*c.T ≤ bounds "g" := extent_le (fun j => h.2.1 (4 : Fin 5) (j,()) trivial)
  have ho : c.heads*c.T*c.V ≤ bounds "o" := by
    apply extent_le
    intro j
    have hl := h.2.2.1 (0 : Fin 1) (j.divNat.divNat,j.divNat.modNat,j.modNat,()) trivial
    change j.divNat.divNat.val*(c.T*c.V)+j.divNat.modNat.val*c.V+j.modNat.val < bounds "o" at hl
    simpa only [output_index] using hl
  intro r
  by_cases hqr : r="q"
  · subst r; simpa [capacity] using hq
  by_cases hkr : r="k"
  · subst r; simpa [capacity] using hk
  by_cases hvr : r="v"
  · subst r; simpa [capacity] using hv
  by_cases hhr : r="h"
  · subst r; simpa [capacity] using hh
  by_cases hgr : r="g"
  · subst r; simpa [capacity] using hg
  by_cases hor : r="o"
  · subst r; simpa [capacity] using ho
  simp [capacity,hqr,hkr,hvr,hhr,hgr,hor]

end
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.IONativeBindings
