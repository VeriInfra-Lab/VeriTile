import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.RealArrayReference
import bench.repairs.tritonbench_g.mixed_sparse_attention.NativeIO

namespace VeriTile.Bench.Repairs.MixedSparseAttention.IOScope
open VeriTile Triton FP.Structural
open Whole002 Source Whole
noncomputable section

/-- Every raw Real input previously covered by Real.Reads supplies these arrays.
No new shape, count/index, tag, alias or allocation premise is imposed. -/
def realInputs (c : Config) (s : State Real.R) (values : RegionName → Nat → ℝ) : KernelIO.Values ℝ c
  | ⟨0,_⟩,off=>values c.q off
  | ⟨1,_⟩,off=>values c.k off
  | ⟨2,_⟩,off=>values c.v off
  | ⟨3,_⟩,off=>(s.mem (Region.cast c.seq) off).read .nat
  | ⟨4,_⟩,off=>(s.mem (Region.cast c.blocks) off).read .nat
  | ⟨5,_⟩,off=>(s.mem (Region.cast c.offsets) off).read .nat
  | ⟨6,_⟩,off=>(s.mem (Region.cast c.counts) off).read .nat
  | ⟨7,_⟩,off=>(s.mem (Region.cast c.columns) off).read .nat

theorem real_binding (c : Config) (work : List Program) (s : State Real.R)
    (values : RegionName → Nat → ℝ) (reads : Real.Reads c s values) :
    (RealIO.io c work).InputBinding (realInputs c s values) s := by
  intro i off
  change Fin 8 at i
  fin_cases i
  · exact Real.read_word c s values reads c.q (by simp) off
  · exact Real.read_word c s values reads c.k (by simp) off
  · exact Real.read_word c s values reads c.v (by simp) off
  all_goals rfl

theorem real_domain (c : Config) (s : State Real.R) (values : RegionName → Nat → ℝ) :
    RealIO.domain c (realInputs c s values) ↔ LengthsBound c s := Iff.rfl

theorem native_binding (rt : Native.Runtime) (I : Native.CompareISA (BitVec 32))
    (c : Config) (work : List Program) (s : State Native.Expr) :
    (NativeIO.io rt I c work).InputBinding ((NativeIO.io rt I c work).snapshot s) s :=
  ArrayProgramIO.snapshot_binding _ rfl s

theorem native_domain (rt : Native.Runtime) (I : Native.CompareISA (BitVec 32))
    (c : Config) (work : List Program) (s : State Native.Expr) :
    KernelIO.domain c ((NativeIO.io rt I c work).snapshot s) ↔ LengthsBound c s := Iff.rfl

#axiomsClean real_binding
#axiomsClean real_domain
#axiomsClean native_binding
#axiomsClean native_domain
end
end VeriTile.Bench.Repairs.MixedSparseAttention.IOScope
