import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.Bounds
import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.KernelSource

namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.KernelIndex
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.FPTyped
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section

theorem ceil_le_self (x b : Nat) (hb : 0<b) : (x+b-1)/b≤x := by
  apply (Nat.div_le_iff_le_mul hb).mpr
  have hx : x≤x*b := by nlinarith
  omega

theorem area_bound (c : Call) (valid : c.Valid) (bounds : Bounds.Signed32 c) :
    c.M*c.N<2^31 := by
  have separated := valid.cValid.separated
  rw [valid.cShape.1,valid.cShape.2] at separated
  have row := valid.cValid.rowPositive
  have col := valid.cValid.colPositive
  have formed := bounds.cformed
  rcases separated with separated|separated
  · have width : c.N≤c.C.sr := by nlinarith
    nlinarith
  · have height : c.M≤c.C.sc := by nlinarith
    nlinarith

/-- Existing checked view/index bounds already imply the concrete one-axis
launch bound for every tuning. No additional launch premise is introduced. -/
theorem grid_bound (c : Call) (valid : c.Valid) (bounds : Bounds.Signed32 c) (i : Fin 17) :
    KernelSource.count c i<2^31 := by
  have hm := ceil_le_self c.M (tunings i).BM (tuning_facts i).1
  have hn := ceil_le_self c.N (tunings i).BN (tuning_facts i).2.2.1
  exact lt_of_le_of_lt (Nat.mul_le_mul hm hn) (area_bound c valid bounds)

theorem signed32 (x : Nat) (bound : x<2^31) :
    (BitVec.ofNat 32 x).toInt=(x : Int) := by
  simp only [BitVec.toInt_eq_toNat_cond,BitVec.toNat_ofNat,
    Nat.mod_eq_of_lt (by omega : x<2^32),if_pos (by omega : 2*x<2^32)]

theorem program_id (c : Call) (valid : c.Valid) (bounds : Bounds.Signed32 c)
    (i : Fin 17) (pid : Nat) (scheduled : pid<KernelSource.count c i) :
    (BitVec.ofNat 32 pid).toInt=(pid : Int) :=
  signed32 pid (lt_trans scheduled (grid_bound c valid bounds i))

theorem pointer64 (base width offset : Nat) (positive : 0<width)
    (offset_bound : offset<2^31) (span : base+width*2^31<2^64) :
    (BitVec.ofNat 64 base+BitVec.ofNat 64 width*BitVec.ofNat 64 offset).toNat=
      base+width*offset ∧ base+width*offset+width≤2^64 := by
  have bound : base+width*offset<2^64 := by nlinarith
  constructor
  · rw [←BitVec.ofNat_mul,←BitVec.ofNat_add,BitVec.toNat_ofNat,Nat.mod_eq_of_lt bound]
  · nlinarith

/-- All formed source element offsets are exact signed32 values, including
padded rows. Inactive lanes still have no memory event. -/
theorem formed_addresses (c : Call) (bounds : Bounds.Signed32 c) (r k n g : Nat)
    (hr : r<c.M+128) (hk : k<c.K+128) (hn : n<c.N)
    (hg : g<c.K/c.group) (hk8 : k/8<c.K/8+16) :
    (BitVec.ofNat 32 (c.A.addr r k)).toInt=(c.A.addr r k : Int) ∧
    (BitVec.ofNat 32 (c.Q.addr (k/8) n)).toInt=(c.Q.addr (k/8) n : Int) ∧
    (BitVec.ofNat 32 (c.S.addr g n)).toInt=(c.S.addr g n : Int) ∧
    (BitVec.ofNat 32 (c.Z.addr g (n/8))).toInt=(c.Z.addr g (n/8) : Int) ∧
    (BitVec.ofNat 32 (c.C.addr r n)).toInt=(c.C.addr r n : Int) := by
  have h := Bounds.formed_offsets c bounds r k n g hr hk hn hg hk8
  exact ⟨signed32 _ h.1,signed32 _ h.2.1,signed32 _ h.2.2.1,
    signed32 _ h.2.2.2.1,signed32 _ h.2.2.2.2⟩

#axiomsClean grid_bound
#axiomsClean program_id
#axiomsClean formed_addresses
#axiomsClean pointer64
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.KernelIndex
