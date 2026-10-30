import bench.tritonbench_g.int8_matmul_quantization.Proofs.QuantizeSource
namespace VeriTile.Bench.TritonBenchG.Int8MatmulQuantization.FPQuantize
open VeriTile.Triton FP.Structural FP.IntegerConversion
open KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 7000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false
local instance writesWordDefault : Inhabited Word := ⟨FP.Value.zero .fp32⟩

abbrev WriteIndex (p : Parameters) := Nat × Index p

def writes (p : Parameters) (n : Nat) : List (WriteIndex p) :=
  (List.range n).product (TileShape.allIndices [p.blockRows,p.blockColumns])

def writeActive (p : Parameters) (i : WriteIndex p) := active p i.1 i.2
instance (p : Parameters) : DecidablePred (writeActive p) := fun _ => inferInstanceAs (Decidable (active _ _ _))

def written (cfg : FP.Config) (p : Parameters) (s : State Word) (n : Nat) : State Word :=
  (writes p n).foldl (fun t i => if writeActive p i then
    t.write p.output (outputOffset p s i.1 i.2) (.mk .int (expected cfg p s i.1 i.2)) else t) s

@[simp] theorem mem_writes (p : Parameters) (n : Nat) (i : WriteIndex p) :
    i ∈ writes p n ↔ i.1 < n := by
  rcases i with ⟨k, i⟩
  simp [writes, List.mem_product]
  exact fun _ => TileShape.mem_allIndices _ i

theorem writes_nodup (p : Parameters) (n : Nat) : (writes p n).Nodup :=
  List.Nodup.product (List.nodup_range) (TileShape.allIndices_nodup _)

@[simp] theorem written_zero (cfg : FP.Config) (p : Parameters) (s : State Word) : written cfg p s 0 = s := by
  simp [written, writes, List.product]

theorem written_succ (cfg : FP.Config) (p : Parameters) (s : State Word) (n : Nat) :
    written cfg p s (n+1) =
      (TileShape.allIndices [p.blockRows,p.blockColumns]).foldl
        (fun t i => if active p n i then t.write p.output (outputOffset p s n i)
          (.mk .int (expected cfg p s n i)) else t) (written cfg p s n) := by
  simp [written, writes, List.product, List.range_succ, List.flatMap_append,
    List.foldl_append, List.foldl_map, writeActive]

theorem written_frame (cfg : FP.Config) (p : Parameters) (s : State Word) (n : Nat)
    (r : RegionName) (o : Nat)
    (h : r ≠ p.output ∨ ∀ k, k<n → ∀ i, active p k i → o ≠ outputOffset p s k i) :
    (written cfg p s n).mem r o = s.mem r o := by
  apply State.masked_scatter_frame
  exact h.imp id (fun h i hi => h i.1 ((mem_writes p n i).mp hi) i.2)

theorem written_input (cfg : FP.Config) (p : Parameters) (s : State Word) (n : Nat)
    (h : p.input ≠ p.output) (o : Nat) :
    (written cfg p s n).mem p.input o = s.mem p.input o :=
  written_frame cfg p s n p.input o (Or.inl h)

theorem written_readback (cfg : FP.Config) (p : Parameters) (s : State Word)
    (i : WriteIndex p) (hi : i.1 < p.steps) (ha : active p i.1 i.2)
    (hinj : ∀ a b : WriteIndex p, a.1 < p.steps → b.1 < p.steps →
      active p a.1 a.2 → active p b.1 b.2 →
      outputOffset p s a.1 a.2 = outputOffset p s b.1 b.2 → a=b) :
    (written cfg p s p.steps).mem p.output (outputOffset p s i.1 i.2) =
      .mk .int (expected cfg p s i.1 i.2) := by
  unfold written
  rw [State.masked_scatter_eq_filter]
  have himem : i ∈ (writes p p.steps).filter (fun j => decide (writeActive p j)) := by
    apply List.mem_filter.mpr
    exact ⟨(mem_writes p p.steps i).mpr hi, by simpa [writeActive] using ha⟩
  obtain ⟨before, after, hl⟩ := List.append_of_mem himem
  have hnd := (writes_nodup p p.steps).filter (fun j => decide (writeActive p j))
  rw [hl, List.nodup_append, List.nodup_cons] at hnd
  obtain ⟨_, ⟨hnot, _⟩, _⟩ := hnd
  rw [hl, List.foldl_append, List.foldl_cons, State.scatter_frame]
  · exact State.write_same _ _ _ _
  · right
    intro j hj heq
    have hjmem : j ∈ (writes p p.steps).filter (fun j => decide (writeActive p j)) := by
      rw [hl]
      simp only [List.mem_append, List.mem_cons]
      exact Or.inr (Or.inr hj)
    have hjk : j.1 < p.steps := (mem_writes p p.steps j).mp (List.mem_filter.mp hjmem).1
    have hja : active p j.1 j.2 := by simpa [writeActive] using (List.mem_filter.mp hjmem).2
    exact hnot ((hinj i j hi hjk ha hja heq) ▸ hj)

#axiomsClean written_succ
#axiomsClean written_frame
#axiomsClean written_input
#axiomsClean written_readback
end
end VeriTile.Bench.TritonBenchG.Int8MatmulQuantization.FPQuantize
