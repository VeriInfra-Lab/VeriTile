import bench.tritonbench_g.uniform_sampling.Proofs.ScalarSpec
import bench.tritonbench_g.uniform_sampling.Proofs.SourceDomain

/-! Input-only output formula. Output storage is four contiguous B-lane blocks
per program, while the input Philox counter index advances only B per program.
The formula uses runtime seed/offset and bounds, never an execution oracle. -/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.Formula
open VeriTile.Triton

def field (n : Nat) : Random.Direct.Field :=
  match n with
  | 0 => .c0
  | 1 => .c1
  | 2 => .c2
  | _ => .c3

noncomputable def expected (B : Nat) (seed offset : Int) (from_ upper : ℝ)
    (index : Nat) : ℝ :=
  ScalarSpec.value seed offset ((index/B/4)*B + index%B)
    (field ((index/B)%4)) from_ upper

theorem expected_at {B : Nat} (hB : 0 < B) (pid : Nat) (f : Fin 4) (lane : Fin B)
    (seed offset : Int) (from_ upper : ℝ) :
    expected B seed offset from_ upper (SourceDomain.outputIndex pid B f lane) =
      ScalarSpec.value seed offset (SourceDomain.counterIndex pid B lane.val)
        (field f.val) from_ upper := by
  have hd : (pid*4 + f.val)/4 = pid := by
    simpa [Nat.mul_comm, Nat.div_eq_of_lt f.isLt] using
      Nat.mul_add_div (by decide : 0 < 4) pid f.val
  have hm : (pid*4 + f.val)%4 = f.val := by
    simp [Nat.add_mod, Nat.mod_eq_of_lt f.isLt]
  simp only [expected, SourceDomain.output_div hB, SourceDomain.output_mod, hd, hm,
    SourceDomain.counterIndex]

/-- The fourth stream remains present and has its own Philox word. -/
theorem four_fields :
    [field 0, field 1, field 2, field 3] =
      [Random.Direct.Field.c0, .c1, .c2, .c3] := rfl

#axiomsClean expected_at
#axiomsClean four_fields

end VeriTile.Bench.TritonBenchG.UniformSampling.Formula
