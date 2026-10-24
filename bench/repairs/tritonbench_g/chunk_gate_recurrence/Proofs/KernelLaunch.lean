import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelTensorInputs
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.SourceBounds

/-! The existing nonempty launch checks, restricted to the actual argument
roles of each phase. Signed negative dead reverse offsets are retained in the
machine expression ledger. No allocator, host return object or numerical
predicate appears in admission. Empty calls do not launch either kernel. -/
namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelLaunch
open VeriTile.Triton KernelProgram KernelTensor
noncomputable section
open Classical
def strides (s : KernelProgram.SeedStrides) : Machine.SeedStrides := ⟨s.batch,s.head,s.key,s.value⟩
def role (kind : Kind) (r : RegionName) : Region :=
  if r="S" then if kind=.forward then .x else .output
  else if r="D" then .gate
  else if r="O" then .output
  else if r="Seed" then .seed
  else if r="DS" then .incoming
  else if r="DI" then .dx
  else if r="DG" then .tileGate
  else if r="DL" then .dseed
  else .caller
def roles (kind : Kind) (seed : Bool) := KernelExecution.inputRoles kind seed++KernelExecution.outputRoles kind
def pointerCheck (b : Buffers c s kind seed) : Bool :=
  (roles kind seed).all fun r => (Checked.pointerIndices seed (role kind r)).all fun i =>
    let bound := ((Checked.sourceRanges c (strides s)).getD i (.fixed 0)).bound
    decide (0≤((b.tensor r).dataPtr.toNat:Int)+4*bound.lo ∧
      ((b.tensor r).dataPtr.toNat:Int)+4*bound.hi<2^64)
def check (b : Buffers c s kind seed) (currentDevice : Nat) : Bool :=
  decide (currentDevice=(b.tensor "S").metadata.device.index ∧ c.heads<2^31 ∧ c.KT≤65535 ∧ c.VT≤65535) &&
    Checked.rangeCheck c (strides s) && pointerCheck b
structure Admitted (b : Buffers c s kind seed) (currentDevice : Nat) : Prop where
  device : currentDevice=(b.tensor "S").metadata.device.index
  heads : c.heads<2^31
  keys : c.KT≤65535
  values : c.VT≤65535
  ranges : Checked.rangeCheck c (strides s)=Bool.true
  pointers : pointerCheck b=Bool.true
theorem check_iff (b : Buffers c s kind seed) (currentDevice : Nat) :
    check b currentDevice=Bool.true ↔ Admitted b currentDevice := by
  simp only [check,Bool.and_eq_true,decide_eq_true_eq]
  constructor
  · rintro ⟨⟨⟨hd,hh,hk,hv⟩,hr⟩,hp⟩
    exact ⟨hd,hh,hk,hv,hr,hp⟩
  · intro h
    exact ⟨⟨⟨h.device,h.heads,h.keys,h.values⟩,h.ranges⟩,h.pointers⟩

theorem expressions (b : Buffers c s kind seed) (current : Nat) (h : Admitted b current)
    (p : Program c) (e : Fin 64) (j : Fin 16) (count : Fin c.N)
    (x : Machine.Expr) (member : x∈Machine.sourceTrees c (strides s) p.head p.key p.value e.val j.val count.val) :
    x.Valid ∧ x.word.toInt=x.eval := by
  have valid := Checked.source_valid c (strides s) h.ranges p.head p.key p.value e.val j.val count.val
    p.headInside p.keyInside p.valueInside e.isLt j.isLt count.isLt x member
  exact ⟨valid,Machine.Expr.no_wrap x valid⟩

theorem formed (b : Buffers c s kind seed) (current : Nat) (h : Admitted b current)
    (p : Program c) (e : Fin 64) (j : Fin 16) (count : Fin c.N) (r : RegionName) (hr : r∈roles kind seed)
    (x : Machine.Expr) (member : x∈Machine.pointerExprs c (strides s) seed
      p.head p.key p.value e.val j.val count.val (role kind r)) :
    x.Valid ∧ x.word.toInt=x.eval ∧
      0≤((b.tensor r).dataPtr.toNat:Int)+4*x.eval ∧ ((b.tensor r).dataPtr.toNat:Int)+4*x.eval<2^64 := by
  have limits := List.all_eq_true.mp h.pointers r hr
  change x∈(Checked.pointerIndices seed (role kind r)).map
    (fun i => (Machine.sourceTrees c (strides s) p.head p.key p.value e.val j.val count.val).getD i (.lit 0)) at member
  obtain ⟨i,hi,rfl⟩ := List.mem_map.mp member
  have binds := Checked.binds_getD (Checked.source_ranges_bind c (strides s)
    p.head p.key p.value e.val j.val count.val p.headInside p.keyInside p.valueInside e.isLt j.isLt count.isLt) i
  have certificate := Checked.BExpr.certified _ _ binds (Checked.getD_valid _ h.ranges i)
  have limit := List.all_eq_true.mp limits i hi
  simp only [decide_eq_true_eq] at limit
  have inside := certificate.2
  simp only [Checked.Interval.Contains] at inside
  exact ⟨certificate.1,Machine.Expr.no_wrap _ certificate.1,by omega,by omega⟩

#axiomsClean check_iff
#axiomsClean expressions
#axiomsClean formed
#specNonCircular Admitted avoiding [FP.Structural.exec,FP.Structural.run,KernelExecution.expected]
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelLaunch
