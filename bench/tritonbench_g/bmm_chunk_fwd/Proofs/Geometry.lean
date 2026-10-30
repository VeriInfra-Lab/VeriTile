import bench.tritonbench_g.bmm_chunk_fwd.Kernels

/-! Source-only grid coordinates used by both independent semantic layers. -/
namespace VeriTile.Bench.Repairs.BmmChunkFwd.Geometry
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
abbrev PID (h : Host.Call) (c : Host.Config) := GridIndex (h.grid c)
def p0 {h : Host.Call} {c : Host.Config} (p : PID h c) : Nat := (p ⟨0,by change 0<3; decide⟩).val
def p1 {h : Host.Call} {c : Host.Config} (p : PID h c) : Nat := (p ⟨1,by change 1<3; decide⟩).val
def p2 {h : Host.Call} {c : Host.Config} (p : PID h c) : Nat := (p ⟨2,by change 2<3; decide⟩).val
theorem p2_lt {h : Host.Call} {c : Host.Config} (p : PID h c) :
    p2 p<h.shape.chunks*h.shape.groups := (p ⟨2,by change 2<3; decide⟩).isLt
theorem cdiv_covers {n b : Nat} (hb : 0<b) : n≤Host.cdiv n b*b := by
  have hmod := Nat.mod_lt (n+b-1) hb
  have he := Nat.mod_add_div (n+b-1) b
  rw [Nat.mul_comm b] at he
  unfold Host.cdiv
  omega
#axiomsClean cdiv_covers
end VeriTile.Bench.Repairs.BmmChunkFwd.Geometry

#guard_msgs (drop info) in
#auditModuleAxioms
