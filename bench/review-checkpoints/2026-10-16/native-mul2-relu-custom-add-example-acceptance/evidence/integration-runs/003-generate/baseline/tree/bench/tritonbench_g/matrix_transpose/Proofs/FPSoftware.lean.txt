import bench.tritonbench_g.matrix_transpose.Proofs.FPBits

namespace VeriTile.Bench.TritonBenchG.MatrixTranspose.FPBits
open VeriTile.Triton FP.Structural

/-- Exactly sixteen storage bits. No decoding, NaN canonicalization or cast
is appropriate for this same-format, unmasked load/transpose/store kernel. -/
abbrev Word := BitVec 16

/-- These deliberately inert primitives are unreachable in this source.
`execute` and `backend_independent` establish this for every initial state
and every parameter, not merely for a witness. No half arithmetic is claimed. -/
def unusedPrimitives : Algebra Word where
  literal := fun _ _ _ => 0
  negInf := 0
  binary := fun _ _ _ _ _ => 0
  unary := fun _ _ _ => 0
  cast := fun _ _ _ _ => 0
  fromNat := fun _ _ => 0
  fromInt := fun _ _ => 0
  fp32Bits := fun _ => 0
  fp32Load := fun _ => 0
  reduceMax := fun _ _ _ _ _ => 0
  reduceSum := fun _ _ _ _ _ => 0

/-- Complete memory equality, every output payload, and the complete cell
frame. No successful execution or final value is assumed. -/
theorem software_correctness (v : Bool) (p : Parameters) (s : State Word)
    (hOutInj : Function.Injective (outputOffset p)) :
    ∃ t, exec unusedPrimitives (source v p) s = some t ∧
      t.mem = referenceMemory p s ∧
      (∀ i : TileIndex [p.d,p.m], t.mem p.output (outputOffset p i) =
        .mk .fp16 (outputValue p s i)) ∧
      (∀ r o, (r ≠ p.output ∨ ∀ i : TileIndex [p.d,p.m], o ≠ outputOffset p i) →
        t.mem r o = s.mem r o) := by
  exact ⟨execution p s, execute unusedPrimitives v p s, execution_memory p s,
    readback p s hOutInj, frame p s⟩

/-- Any arbitrary implementation of the unused floating operations produces
the same entire state, including registers, not only the same output. -/
theorem software_has_no_numerical_dependency (M : Algebra Word)
    (v : Bool) (p : Parameters) (s : State Word) :
    exec M (source v p) s = exec unusedPrimitives (source v p) s :=
  backend_independent M unusedPrimitives v p s

#axiomsClean software_correctness
#axiomsClean software_has_no_numerical_dependency
end VeriTile.Bench.TritonBenchG.MatrixTranspose.FPBits
