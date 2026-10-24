import bench.tritonbench_g.matmul_tma.Proofs.Real
import bench.tritonbench_g.matmul_tma.Proofs.Allocation

/-! The actual host wrapper, interpreted mathematically. The generic structural
execution helper is reused; no FP numerical premise or reader entry is imported. -/
namespace VeriTile.Bench.TritonBenchG.MatmulTma.WholeReal
open VeriTile.Triton FP.Structural FPValue Host
noncomputable section

/-- Mathematical interpretation of storage conversions. Unused primitive
fields are arbitrary; the source invokes only this cast and the supplied dot. -/
def algebra (rest : Algebra ℝ) : Algebra ℝ :=
  {rest with cast := fun _ _ _ x => x}

/-- Independent batched finite-sum contraction, specialized below to batch=[]. -/
def contraction {batch : TileShape} {m k n : Nat}
    (a : TileIndex (batch ++ [m,k]) → ℝ)
    (b : TileIndex (batch ++ [k,n]) → ℝ) : TileIndex (batch ++ [m,n]) → ℝ :=
  match batch with
  | [] => fun i => ∑ t : Fin k, a (i.1,t,PUnit.unit) * b (t,i.2.1,PUnit.unit)
  | _ :: tail => fun i => contraction (batch := tail)
      (fun j => a (i.1,j)) (fun j => b (i.1,j)) i.2
termination_by batch.length

def dot : Dot ℝ :=
  ⟨fun _ _ {batch} {m} {k} {n} a b => contraction (batch := batch) (m := m) (k := k) (n := n) a b⟩

theorem output_product (rest : Algebra ℝ) (h : Shape) (f : Flags) (r : Regions)
    (s : State ℝ) :
    outputValues (algebra rest) dot (parameters h f r) f.half s =
      Real.product h (aValues (parameters h f r) s) (bValues (parameters h f r) s) := by
  funext i
  simp only [outputValues,algebra,dot,contraction,Real.product]
  apply Finset.sum_congr rfl
  intro k _
  rfl

/-- After the same ordinary allocations and exact setup request, the actual
source wrapper returns the selected tensor holding the independent real sum.
Frame is relative to memory before the first randn, outside newly allocated
buffers; the source also preserves every non-output cell of launch memory. -/
theorem wrapper_mathematical (rest : Algebra ℝ) (alloc : Allocator ℝ) (admit : Admission)
    (h : Shape) (f : Flags) (cfg : Config) (r : Regions) (storage : FlatAlloc)
    (initial s : State ℝ) (prepared : Prepared alloc h f r initial s)
    (binding : AllocationBinding h f r storage) (frame : AllocatorFrame alloc)
    (ready : admit (request h f r cfg) = .ok ()) :
    ∃ t, wrapper (algebra rest) dot alloc admit h f cfg initial = .ok (returnedTensor h f r,t) ∧
      Post (parameters h f r) f.half s t
        (Real.product h (aValues (parameters h f r) s) (bValues (parameters h f r) s)) ∧
      (f.half = Bool.true → ∀ off, t.mem r.first off = s.mem r.first off) ∧
      (∀ R off, OutsideAllocated f r R → t.mem R off = initial.mem R off) := by
  obtain ⟨t,he,hp⟩ := wrapper_correct (algebra rest) dot alloc admit h f cfg r initial s prepared ready
  refine ⟨t,he,?_,?_,?_⟩
  · simpa only [Returned,output_product] using hp
  · exact fun half => replacement_framed (algebra rest) dot h f r storage s t binding half hp
  · exact returned_frame_initial (algebra rest) dot alloc h f r initial s t prepared frame hp

#axiomsClean output_product
#axiomsClean wrapper_mathematical
#specNonCircular contraction avoiding [source,wrapper,FP.Structural.exec]
end
end VeriTile.Bench.TritonBenchG.MatmulTma.WholeReal
