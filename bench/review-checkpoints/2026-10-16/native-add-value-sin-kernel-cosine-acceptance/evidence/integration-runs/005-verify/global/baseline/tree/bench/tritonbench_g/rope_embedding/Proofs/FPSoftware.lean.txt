import bench.tritonbench_g.rope_embedding.Proofs.FPContract
namespace VeriTile.Bench.TritonBenchG.RopeEmbedding.FPHalves
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def roundedValue [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (backward second : Bool)
    (p : Parameters) (s : State (FP.Value .fp32)) (i : TileIndex [p.width]) : FP.Value .fp32 :=
  let x0 := (s.mem p.input (offset p s Bool.false i)).read .fp32
  let x1 := (s.mem p.input (offset p s Bool.true i)).read .fp32
  let c := (s.mem p.cosine (coefficientOffset p s p.cosineStride i)).read .fp32
  let rawSine := (s.mem p.sine (coefficientOffset p s p.sineStride i)).read .fp32
  let sn := if backward then rawSine.neg else rawSine
  if second then FP.add cfg (FP.mul cfg x1 c) (FP.mul cfg x0 sn)
  else FP.sub cfg (FP.mul cfg x0 c) (FP.mul cfg x1 sn)

theorem software_correctness [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (backward second : Bool)
    (p : Parameters) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest cfg) (source backward second p) s = some final ∧
      Post second p s final (roundedValue cfg backward second p s) := by
  obtain ⟨final, he, hp⟩ := correctness (FP.Software.scalarFP32 rest cfg) backward second p s
  refine ⟨final, he, ?_, hp.2⟩
  intro i hi
  cases backward <;> cases second <;>
    simpa [value, roundedValue, FP.Software.scalarFP32, FP.Software.fp32Context] using hp.1 i hi
#axiomsClean software_correctness
end
end VeriTile.Bench.TritonBenchG.RopeEmbedding.FPHalves
