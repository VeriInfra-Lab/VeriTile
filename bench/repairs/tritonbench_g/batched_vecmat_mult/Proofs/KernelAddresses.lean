import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.KernelModel

namespace VeriTile.Bench.Repairs.BatchedVecmatMult.KernelAddresses
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BatchedVecmatMult
open KernelDomain KernelBinding
noncomputable section
local instance : Inhabited SourceGraph.Expr := ⟨SourceGraph.Expr.zero⟩

/-- All finite source accesses use the same actual data pointers as the
Tensor binding. Offsets, signed interpretation and all four accessed bytes
are proved by the reused int64 interpreter, with no count cutoff guard. -/
structure AddressBinding (a : Arguments) (x : Allocated [a.M,a.K])
    (y : Allocated [a.M,a.N,a.K]) (z : Allocated [a.M,a.N]) (heap : Heap) : Prop where
  pointerA : 0<x.count → x.tensor.dataPtr=BitVec.ofNat 64 (4*x.base)
  pointerB : 0<y.count → y.tensor.dataPtr=BitVec.ofNat 64 (4*y.base)
  pointerOut : 0<z.count → z.tensor.dataPtr=BitVec.ofNat 64 (4*z.base)
  reads : ∀ (s : State SourceGraph.Expr), FiniteExecution.ProgramIds (parameters a) s →
    ∀ (c : Fin (a.K/a.BK)) (i : Fin a.BM) (j : Fin a.BN) (e : Fin a.BK),
    heap ((Index64.bytePointer (4*x.base)
      (FiniteExecution.aOffset (parameters a) s (Index64.word c.val) i e)).toNat/4)=
      x.value heap (SourceGraph.aOffset (parameters a) s i.val c.val e.val) ∧
    heap ((Index64.bytePointer (4*y.base)
      (FiniteExecution.bOffset (parameters a) s (Index64.word c.val) i j e)).toNat/4)=
      y.value heap (SourceGraph.bOffset (parameters a) s i.val j.val c.val e.val)
  writes : ∀ (s : State SourceGraph.Expr), FiniteExecution.ProgramIds (parameters a) s →
    ∀ i : FiniteExecution.Lane (parameters a),
    (Index64.bytePointer (4*z.base) (FiniteExecution.outOffset (parameters a) s i)).toNat=
      4*(z.base+SourceGraph.outputOffset (parameters a) s i) ∧
    (Index64.bytePointer (4*z.base) (FiniteExecution.outOffset (parameters a) s i)).toNat+3<
      4*(z.base+z.count)

theorem addresses (a : Arguments) (native : LaunchABI a) (x : Allocated [a.M,a.K])
    (y : Allocated [a.M,a.N,a.K]) (z : Allocated [a.M,a.N]) (heap : Heap) :
    AddressBinding a x y z heap where
  pointerA := x.actual_pointer
  pointerB := y.actual_pointer
  pointerOut := z.actual_pointer
  reads s hp c i j e := by
    have ia := integerAgreement a native x y z
    have wa : 4*x.base+(parameters a).M*(parameters a).K*4≤2^64 := by
      simpa [parameters,Nat.mul_assoc] using x.span
    have wb : 4*y.base+(parameters a).M*(parameters a).N*(parameters a).K*4≤2^64 := by
      simpa [parameters,Nat.mul_assoc] using y.span
    obtain ⟨ha,_,ba,hb,_,bb⟩ := FiniteExecution.input_byte_addresses (parameters a) s ia hp c i j e
      (4*x.base) (4*y.base) wa wb
    have ac : SourceGraph.aOffset (parameters a) s i.val c.val e.val<x.count := by
      rw [ha] at ba
      change 4*x.base+SourceGraph.aOffset (parameters a) s i.val c.val e.val*4+3<
        4*x.base+a.M*a.K*4 at ba
      simpa [parameters,Nat.mul_assoc] using (by omega : SourceGraph.aOffset (parameters a) s i.val c.val e.val<a.M*a.K)
    have bc : SourceGraph.bOffset (parameters a) s i.val j.val c.val e.val<y.count := by
      rw [hb] at bb
      change 4*y.base+SourceGraph.bOffset (parameters a) s i.val j.val c.val e.val*4+3<
        4*y.base+a.M*a.N*a.K*4 at bb
      simpa [parameters,Nat.mul_assoc] using (by omega : SourceGraph.bOffset (parameters a) s i.val j.val c.val e.val<a.M*a.N*a.K)
    rw [ha,hb,x.direct heap _ ac,y.direct heap _ bc]
    constructor <;> congr 1 <;> omega
  writes s hp i := by
    have window : 4*z.base+(parameters a).M*(parameters a).N*4≤2^64 := by
      simpa [parameters,Nat.mul_assoc] using z.span
    obtain ⟨ha,_,hb⟩ := FiniteExecution.output_byte_address (parameters a) s
      (integerAgreement a native x y z) hp i (4*z.base) window
    constructor
    · omega
    · simpa [parameters,Nat.mul_add,Nat.mul_assoc,Nat.mul_comm,Nat.mul_left_comm] using hb

#axiomsClean addresses
end
end VeriTile.Bench.Repairs.BatchedVecmatMult.KernelAddresses
