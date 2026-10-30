import bench.tritonbench_g.bgmv_shrink_kernel.Sources.Direct_Geometry

namespace VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset
open VeriTile.Triton Geometry
/-- Ordinary atomic linearization supplies an arbitrary actual order separately
for each output cell. There is no shared program order, reassociation, numerical
identity, scheduler implementation, or output equation in this data. -/
structure Orders (h : Call) where
  order : Fin h.batches → Fin h.N → List (Fin 64)
  complete : ∀ b n,(order b n).Perm (List.finRange 64)

/-- The entire launch's statically known potential atomic events. The sentinel
removes all events of its row; the n<N store mask removes padded columns. -/
abbrev Event (h : Call) := Fin h.batches × Fin 64 × Fin h.N

def active (indices : Nat → Int) {h : Call} (e : Event h) : Prop := indices e.1.val≠-1

def eventAddress (h : Call) (e : Event h) := outputOffset h e.1.val e.2.2.val

/-- Derive the exact event fiber of a valid cell from the grid and mask. -/
theorem fiber (h : Call) (b : Fin h.batches) (n : Fin h.N) (e : Event h) :
    eventAddress h e=outputOffset h b.val n.val ↔
      ∃! s : Fin 64,e=(b,s,n) := by
  constructor
  · intro same
    obtain ⟨hb,hn⟩ := output_injective h e.1.val b.val
      e.2.2 n same
    have first : e.1=b := Fin.ext hb
    have last : e.2.2=n := Fin.ext (congrArg (fun i : Fin h.N => i.val) hn)
    refine ⟨e.2.1,?_,?_⟩
    · exact Prod.ext first (Prod.ext rfl last)
    · intro s eq
      exact (congrArg (fun t : Event h => t.2.1) eq).symm
  · rintro ⟨s,rfl,_⟩; rfl


#axiomsClean fiber
end VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset
