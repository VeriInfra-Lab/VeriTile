import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.TypedGrid
import bench.tritonbench_g.chunk_bwd_dqkg.Domain

/-! Complete readable quote of chunk_simple_gla_bwd_kernel_dqkg from the
canonical repaired Python file, SHA256
7c2c6aef92283c9c3f9f4980291ab2367eaf6096436b5847c683a560335a1ac6.
All statements, masks, casts, reductions, pointer offsets and stores remain.
Real correctness uses its explicit erasure; native correctness uses its typed
interpretation. No wrapper/runtime operation occurs in these endpoints. -/
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg
abbrev kernel_source := Whole.kernel
abbrev kernel_source_exec := @Typed.source_runs
abbrev kernel_source_safety := @Whole.source_safe_dense
abbrev kernel_signed32 := @Whole.formed_signed32
abbrev kernel_pointer64 := @Whole.formed_pointer64
abbrev kernel_real_projection := @RealSource.real_body_matches
#axiomsClean kernel_source_exec
#axiomsClean kernel_source_safety
#axiomsClean kernel_signed32
#axiomsClean kernel_pointer64
#axiomsClean kernel_real_projection
end VeriTile.Bench.Repairs.ChunkBwdDqkg
