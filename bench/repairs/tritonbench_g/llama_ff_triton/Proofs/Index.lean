import bench.repairs.tritonbench_g.llama_ff_triton.Proofs.Geometry
import bench.repairs.tritonbench_g.int8_dequant_matmul.Proofs.Machine

namespace VeriTile.Bench.Repairs.LlamaFF
open VeriTile.Bench.Repairs.LlamaFF
noncomputable section
set_option maxHeartbeats 5000000

def address64 (v : View3) (b m n : Nat) : BitVec 64 :=
  BitVec.ofNat 64 v.offset + BitVec.ofNat 64 b*BitVec.ofNat 64 v.sb +
    BitVec.ofNat 64 m*BitVec.ofNat 64 v.sm + BitVec.ofNat 64 n*BitVec.ofNat 64 v.sn
def View3.FormedSafe (v : View3) (B M N bytes : Nat) : Prop :=
  0<bytes ∧ bytes*(v.offset+B*v.sb+M*v.sm+N*v.sn)<2^63
theorem formed_exact (v : View3) (B M N bytes : Nat) (safe : v.FormedSafe B M N bytes)
    (b m n : Nat) (hb : b≤B) (hm : m≤M) (hn : n≤N) :
    (address64 v b m n).toInt=(v.offset+b*v.sb+m*v.sm+n*v.sn : Nat) ∧
      bytes*(v.offset+b*v.sb+m*v.sm+n*v.sn)<2^63 := by
  have h1 := Nat.mul_le_mul_right v.sb hb
  have h2 := Nat.mul_le_mul_right v.sm hm
  have h3 := Nat.mul_le_mul_right v.sn hn
  have bound : v.offset+b*v.sb+m*v.sm+n*v.sn≤v.offset+B*v.sb+M*v.sm+N*v.sn := by omega
  have byteBound := (Nat.mul_le_mul_left bytes bound).trans_lt safe.2
  constructor
  · simpa only [address64,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat] using
      DequantizeMatmul.signed64_exact (v.offset+b*v.sb+m*v.sm+n*v.sn) (by nlinarith [safe.1])
  · exact byteBound
def formedK (s : Shape) := steps s.K*64+63
def weightView (v : View2) : View3 := ⟨v.storage,v.offset,0,v.sn,v.sk⟩
def rmsView (v : View1) : View3 := ⟨v.storage,v.offset,0,0,v.stride⟩
structure IndexDomain (s : Shape) (b : Buffers) : Prop where
  dimensions : s.B<2^31 ∧ s.M<2^31 ∧ s.N<2^31 ∧ s.K<2^31
  xStrides : b.x.sb<2^31 ∧ b.x.sm<2^31 ∧ b.x.sn<2^31
  rmsStride : b.rms.stride<2^31
  wStrides : b.weight.sn<2^31 ∧ b.weight.sk<2^31
  w3Strides : b.weight3.sn<2^31 ∧ b.weight3.sk<2^31
  oStrides : b.output.sb<2^31 ∧ b.output.sm<2^31 ∧ b.output.sn<2^31
  gridX : tilesM s*tilesN s≤2^31-1
  padded : tilesM s*16+63<2^63 ∧ tilesN s*16+63<2^63 ∧ formedK s<2^63
  xFormed : b.x.FormedSafe s.B s.M (formedK s) 2
  rmsFormed : (rmsView b.rms).FormedSafe 0 0 (formedK s) (if b.useByte then 1 else 2)
  weightFormed : (weightView b.weight).FormedSafe 0 s.N (formedK s) (if b.useByte then 1 else 2)
  weight3Formed : (weightView b.weight3).FormedSafe 0 s.N (formedK s) (if b.useByte then 1 else 2)
  outputFormed : b.output.FormedSafe s.B (tilesM s*16) (tilesN s*16)
    2

/-- Includes all masked lanes and the final unused pointer increment. -/
theorem loop_formed (s : Shape) (step lane : Nat) (ht : step≤steps s.K) (hl : lane<64) :
    step*64+lane≤formedK s := by unfold formedK; omega
theorem IndexDomain.x_exact (domain : IndexDomain s b) (batch row step lane : Nat)
    (hb : batch<s.B) (hm : row<s.M) (ht : step≤steps s.K) (hl : lane<64) :
    (address64 b.x batch row (step*64+lane)).toInt=
      (b.x.offset+batch*b.x.sb+row*b.x.sm+(step*64+lane)*b.x.sn : Nat) :=
  (formed_exact b.x s.B s.M (formedK s) 2 domain.xFormed batch row (step*64+lane)
    hb.le hm.le (loop_formed s step lane ht hl)).1
theorem IndexDomain.output_exact (domain : IndexDomain s b)
    (batch tm tn rm cn : Nat) (hb : batch<s.B) (htm : tm<tilesM s) (htn : tn<tilesN s)
    (hr : rm<16) (hc : cn<16) :
    (address64 b.output batch (tm*16+rm) (tn*16+cn)).toInt=
      (b.output.offset+batch*b.output.sb+(tm*16+rm)*b.output.sm+
        (tn*16+cn)*b.output.sn : Nat) :=
  (formed_exact _ _ _ _ _ domain.outputFormed batch (tm*16+rm) (tn*16+cn)
    hb.le (by omega) (by omega)).1

theorem original_i32_overflow :
    ((BitVec.ofNat 32 65536)*(BitVec.ofNat 32 32768)).toInt= -2147483648 := by decide
theorem repaired_i64_exact :
    ((BitVec.ofNat 64 65536)*(BitVec.ofNat 64 32768)).toInt=2147483648 := by decide

#axiomsClean formed_exact
#axiomsClean loop_formed
#axiomsClean IndexDomain.x_exact
#axiomsClean IndexDomain.output_exact
#axiomsClean original_i32_overflow
#axiomsClean repaired_i64_exact
end
end VeriTile.Bench.Repairs.LlamaFF
