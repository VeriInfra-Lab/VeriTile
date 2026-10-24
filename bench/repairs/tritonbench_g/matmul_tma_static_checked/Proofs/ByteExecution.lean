import bench.repairs.tritonbench_g.matmul_tma_static_checked.Proofs.Bytes
import bench.repairs.tritonbench_g.matmul_tma_static_checked.Proofs.Addresses

namespace VeriTile.Bench.TritonBenchG.MatmulTma.StaticChecked.ByteExecution
open VeriTile.Triton FP.Structural FPValue Host TensorABI Native Bytes
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 4000000

abbrev Machine := State Word × Read

/-- Each source tile-store cell updates its typed cell and the finite physical
pointer's bytes. Operands/output values were evaluated before the first store. -/
def step (h : Shape) (f : Flags) (c : TensorABI.Tensor) (values : CIndex (params h f) → Word)
    (t : Machine) (i : CIndex (params h f)) : Machine :=
  (t.1.write (params h f).c (cOffset (params h f) i) (stored (outputDType f.half) (values i)),
    Bytes.store t.2 f.half
      (Addresses.cPointer h f c i).toNat (values i))

def program (M : Algebra Word) (D : Dot Word) (h : Shape) (f : Flags) (c : TensorABI.Tensor)
    (t : Machine) : Machine :=
  (TileShape.allIndices [h.rows,h.columns]).foldl
    (step h f c (outputValues M D (params h f) f.half t.1)) t

theorem fold_render (h : Shape) (f : Flags) (dim : Addresses.Dimensions h) (c : TensorABI.Tensor)
    (window : c.metadata.dataPtr.toNat+width f.half*(h.rows*h.columns)≤2^64)
    (values : CIndex (params h f) → Word) (read : Read)
    (items : List (CIndex (params h f))) (s : State Word) :
    items.foldl (step h f c values) (s,render read f.half c (h.rows*h.columns) s) =
      let final := items.foldl (fun t i => t.write (params h f).c (cOffset (params h f) i)
        (stored (outputDType f.half) (values i))) s
      (final,render read f.half c (h.rows*h.columns) final) := by
  induction items generalizing s with
  | nil => rfl
  | cons i tail ih =>
    simp only [List.foldl_cons]
    have bound : cOffset (params h f) i<h.rows*h.columns := c_bound h f regions i
    have address := pointer_exact c.metadata.dataPtr.toNat (width f.half) (h.rows*h.columns)
      (cOffset (params h f) i) (by cases f.half <;> decide) window bound
    simp only [step,Addresses.c_pointer h f dim,address,regionC,←render_write read f.half c (h.rows*h.columns) s _ bound]
    simpa only [regionC] using ih (s.write "C" (cOffset (params h f) i) (stored (outputDType f.half) (values i)))

theorem render_memory (read : Read) (h : Shape) (f : Flags) (c : TensorABI.Tensor)
    (s t : State Word) (eq : s.mem=t.mem) :
    render read f.half c (h.rows*h.columns) s = render read f.half c (h.rows*h.columns) t := by
  unfold render
  rw [eq]

/-- Full source execution agrees with the physical byte-store interpreter.
Success and store/frame behavior are conclusions, never input premises. -/
theorem refines (M : Algebra Word) (D : Dot Word) (h : Shape) (f : Flags) (dim : Addresses.Dimensions h) (c : TensorABI.Tensor)
    (window : c.metadata.dataPtr.toNat+width f.half*(h.rows*h.columns)≤2^64)
    (read : Read) (s : State Word) (bound : Bound read f.half c (h.rows*h.columns) s) :
    ∃ typed actual bytes,
      FP.Structural.exec (backend M D) (kernel h f) s=some typed ∧
      program M D h f c (s,read)=(actual,bytes) ∧
      typed.mem=actual.mem ∧ bytes=render read f.half c (h.rows*h.columns) typed ∧
      Post (params h f) f.half s typed (outputValues M D (params h f) f.half s) := by
  obtain ⟨typed,he,hm,post⟩ := executes M D h f s
  let actual := scatter (params h f) f.half s (outputValues M D (params h f) f.half s)
  refine ⟨typed,actual,render read f.half c (h.rows*h.columns) actual,he,?_,hm,?_,post⟩
  · have fold := fold_render h f dim c window (outputValues M D (params h f) f.half s)
      read (TileShape.allIndices [h.rows,h.columns]) s
    rw [render_initial read f.half c (h.rows*h.columns) s bound] at fold
    exact fold
  · exact render_memory read h f c actual typed hm.symm

#axiomsClean fold_render
#axiomsClean refines
end
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticChecked.ByteExecution
