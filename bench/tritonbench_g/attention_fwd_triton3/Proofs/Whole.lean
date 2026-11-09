import bench.tritonbench_g.attention_fwd_triton3.RealCorrect

/-! The complete repaired-source Real proof, including ordinary-grid execution,
finite six-input bindings, exact output cells, trace safety and frame. -/
namespace VeriTile.Bench.Repairs.AttentionFwdTriton3.Whole
abbrev complete := VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.correct
end VeriTile.Bench.Repairs.AttentionFwdTriton3.Whole
