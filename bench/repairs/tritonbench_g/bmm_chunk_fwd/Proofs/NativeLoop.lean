import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.SourcePointers

namespace VeriTile.Bench.Repairs.BmmChunkFwd.NativeLoop
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open SourcePointers
noncomputable section

/-- Signed-i64 loop control around the actual source body. Exhausted fuel still
checks the terminal guard; it cannot silently truncate an active iteration. -/
def run {α : Type} [Inhabited α] (M : Algebra α) (body : List ComputeStmt) :
    Nat → BitVec 64 → BitVec 64 → State α → Option (State α)
  | 0,cur,stop,s => if cur.slt stop then none else some s
  | fuel+1,cur,stop,s =>
    if cur.slt stop then
      (FP.Structural.run M body (s.setReg "k" .nat [] (fun _ => cur.toNat))).bind
        (run M body fuel (cur+1) stop)
    else some s

theorem exact_range {α : Type} [Inhabited α] (M : Algebra α) (body : List ComputeStmt)
    (fuel cur stop : Nat) (ht : cur+fuel=stop) (hs : stop<9223372036854775808)
    (s : State α) :
    run M body fuel (BitVec.ofNat 64 cur) (BitVec.ofNat 64 stop) s=
      range M "k" cur stop 1 body s := by
  induction fuel generalizing cur s with
  | zero =>
    have eq : cur=stop := by omega
    subst cur
    rw [run,Finite.compare64 _ _ hs hs,range]
    simp
  | succ fuel ih =>
    have hc : cur<stop := by omega
    have hcur : cur<9223372036854775808 := by omega
    rw [run,Finite.compare64 _ _ hcur hs,if_pos (by simp [hc]),range,
      if_neg (by decide),if_pos hc,NativeControl.nat64 cur hcur]
    congr 1
    funext t
    have inc : BitVec.ofNat 64 cur+1=BitVec.ofNat 64 (cur+1) := by
      change BitVec.ofNat 64 cur+BitVec.ofNat 64 1=BitVec.ofNat 64 (cur+1)
      exact (BitVec.ofNat_add _ _).symm
    rw [inc]
    exact ih (cur+1) (by omega) t

theorem stop_bound (h : Host.Call) (c : Host.Config) (hc : c∈Host.configurations)
    (sizes : Finite.AllocatedSizes h.shape) : Host.cdiv h.shape.K c.K<9223372036854775808 := by
  have hk := (Finite.dimensions_bounded h.shape sizes).2.2
  have hb := (Finite.configuration_small c hc).2.2
  have hd := Nat.div_le_self (h.shape.K+c.K-1) c.K
  unfold Host.cdiv
  omega

/-- Every entered body sees the exact signed K value; its backedge is exact
through the terminal value. This also proves the zero-iteration K=0 branch. -/
theorem control (h : Host.Call) (c : Host.Config) (hc : c∈Host.configurations)
    (sizes : Finite.AllocatedSizes h.shape) (q : Nat) (hq : q≤Host.cdiv h.shape.K c.K) :
    (BitVec.ofNat 64 q).slt (NativeControl.ceil h.shape.K c.K)=
      decide (q<Host.cdiv h.shape.K c.K) ∧
    (BitVec.ofNat 64 q).toInt=q ∧
    (q<Host.cdiv h.shape.K c.K → (BitVec.ofNat 64 q+1).toInt=q+1) := by
  have hs := stop_bound h c hc sizes
  have hk := (Finite.dimensions_bounded h.shape sizes).2.2
  have hb := (Finite.configuration_small c hc).2.2
  have hp := (Host.configuration_positive c hc).2.2
  rw [NativeControl.ceil_exact _ _ hp (by omega)]
  refine ⟨Finite.compare64 _ _ (by omega) hs,Finite.signed64 _ (by omega),?_⟩
  intro less
  change (BitVec.ofNat 64 q+BitVec.ofNat 64 1).toInt= _
  rw [←BitVec.ofNat_add,Finite.signed64 _ (by omega)]
  rfl

theorem source_iterations {α : Type} [Inhabited α] (o : FPSuccess.Ops α)
    (h : Host.Call) (c : Host.Config) (hc : c∈Host.configurations)
    (sizes : Finite.AllocatedSizes h.shape) (d : Coordinates) (s : State α)
    (hs : StateAt h c d 0 s) (ha : h.aStorage=.fp32) (hb : h.bStorage=.fp32) :
    ∃ t, range (engine o.algebra) "k" 0 (Host.cdiv h.shape.K c.K) 1 (FPSuccess.parts h c).body s=some t ∧
      run (engine o.algebra) (FPSuccess.parts h c).body (Host.cdiv h.shape.K c.K)
        0 (NativeControl.ceil h.shape.K c.K) s=some t ∧
      StateAt h c d (Host.cdiv h.shape.K c.K) t ∧ t.mem=s.mem := by
  obtain ⟨t,hr,hstate,hmem⟩ := SourcePointers.iterations o h c d 0 _ (Nat.zero_le _) s hs ha hb
  refine ⟨t,hr,?_,hstate,hmem⟩
  have hk := (Finite.dimensions_bounded h.shape sizes).2.2
  have hbk := (Finite.configuration_small c hc).2.2
  rw [NativeControl.ceil_exact _ _ (Host.configuration_positive c hc).2.2 (by omega)]
  exact (exact_range (engine o.algebra) (FPSuccess.parts h c).body
    (Host.cdiv h.shape.K c.K) 0 (Host.cdiv h.shape.K c.K) (by omega)
    (stop_bound h c hc sizes) s).trans hr

#axiomsClean exact_range
#axiomsClean control
#axiomsClean source_iterations
end
end VeriTile.Bench.Repairs.BmmChunkFwd.NativeLoop

#guard_msgs (drop info) in
#auditModuleAxioms
