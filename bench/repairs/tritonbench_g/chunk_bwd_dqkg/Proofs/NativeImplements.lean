import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.NativeIOResults

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.NativeImplements
open VeriTile Triton FP.Structural Whole KernelNativeCarrier AddressBounds
noncomputable section
set_option maxHeartbeats 1000000

theorem length_bound (n b : Nat) (h : ∀ j : Fin n,j.val<b) : n≤b := by
  by_contra hn
  exact (Nat.lt_irrefl b) (h ⟨b,Nat.lt_of_not_ge hn⟩)

theorem capacity (isa : Instructions) (route : Routing) (scale : Word) (c : Config)
    (order : List (CTA c)) (vals : ChannelValues Node (IO.nativeInputs c))
    (bounds : RegionName → Nat) (hb : (IO.nativeIO isa route scale c order).InBounds bounds vals) :
    Capacity c bounds := by
  apply RealIOResults.capacity_of_windows c bounds
  · intro i
    fin_cases i
    · apply length_bound
      intro j
      exact hb.2.1 (0 : Fin 7) (j,PUnit.unit) trivial
    · apply length_bound
      intro j
      exact hb.2.1 (1 : Fin 7) (j,PUnit.unit) trivial
    · apply length_bound
      intro j
      exact hb.2.1 (2 : Fin 7) (j,PUnit.unit) trivial
    · apply length_bound
      intro j
      exact hb.2.1 (3 : Fin 7) (j,PUnit.unit) trivial
    · apply length_bound
      intro j
      exact hb.2.1 (4 : Fin 7) (j,PUnit.unit) trivial
    · apply length_bound
      intro j
      exact hb.2.1 (5 : Fin 7) (j,PUnit.unit) trivial
    · apply length_bound
      intro j
      exact hb.2.1 (6 : Fin 7) (j,PUnit.unit) trivial
  · intro o
    fin_cases o
    · apply length_bound
      intro j
      exact hb.2.2.1 (0 : Fin 3) (j,PUnit.unit) trivial
    · apply length_bound
      intro j
      exact hb.2.2.1 (1 : Fin 3) (j,PUnit.unit) trivial
    · apply length_bound
      intro j
      exact hb.2.2.1 (2 : Fin 3) (j,PUnit.unit) trivial

/-- Composition helper only: actual source safety is proved separately, never
accepted as a public input-domain or numerical premise. -/
theorem implements (isa : Instructions) (route : Routing) (scale : Word) (c : Config)
    (order : List (CTA c)) (ho : order.Perm (allCTAs c))
    (safe : ∀ s a bounds,Capacity c bounds → (IO.nativeJob isa route scale c a).Safe bounds s) :
    (IO.nativeIO isa route scale c order).Implements (NativeIOResults.formula isa route scale c) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds bounded initial binding
    have cap:=capacity isa route scale c order vals bounds bounded
    change ScheduleSafe bounds (order.map (IO.nativeJob isa route scale c)) initial
    clear ho bounded binding vals
    induction order generalizing initial with
    | nil=>trivial
    | cons a rest ih=>
      refine ⟨safe initial a bounds cap,?_⟩
      intro next _
      exact ih next
  · exact NativeIOResults.correct isa route scale c order ho

end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.NativeImplements
