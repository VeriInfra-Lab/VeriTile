import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.Direct
namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton.Native
open VeriTile.Triton
noncomputable section

def nativeOffset (r columns lane : Nat) : BitVec 64 :=
  (BitVec.ofNat 32 r).signExtend 64 * BitVec.ofNat 64 columns +
    (BitVec.ofNat 32 lane).signExtend 64

theorem widen32 (n : Nat) (h : n<2^31) :
    (BitVec.ofNat 32 n).signExtend 64=BitVec.ofNat 64 n := by
  apply BitVec.eq_of_toInt_eq
  rw [BitVec.toInt_signExtend_of_le (by decide)]
  have h32 : (BitVec.ofNat 32 n).toInt=(n : Int) :=
    BitVec.toInt_ofInt_eq_self (w:=32) (n:=(n : Int)) (by decide) (by omega) (by norm_num;omega)
  have h64 : (BitVec.ofNat 64 n).toInt=(n : Int) :=
    BitVec.toInt_ofInt_eq_self (w:=64) (n:=(n : Int)) (by decide) (by omega) (by norm_num;omega)
  rw [h32,h64]

theorem offset_exact (x : TensorMetadata) (h : InputContract x) (r lane : Nat)
    (hr : r<rows x) (hl : lane<paddedWidth x) :
    (nativeOffset r (columns x) lane).toInt=(r*columns x+lane : Nat) := by
  obtain ⟨rb,lb,ob⟩ := native_indices x h r lane hr hl
  rw [nativeOffset,widen32 r rb,widen32 lane lb,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]
  exact BitVec.toInt_ofInt_eq_self (w:=64) (n:=((r*columns x+lane : Nat) : Int))
    (by decide) (by omega) (by norm_num;omega)

def pointer (base bytes : Nat) (offset : BitVec 64) : BitVec 64 :=
  BitVec.ofNat 64 base + offset * BitVec.ofNat 64 bytes

theorem pointer_exact (base bytes r cols lane : Nat)
    (rb : r<2^31) (lb : lane<2^31)
    (bound : base+(r*cols+lane)*bytes<2^64) :
    (pointer base bytes (nativeOffset r cols lane)).toNat=base+(r*cols+lane)*bytes := by
  rw [pointer,nativeOffset,widen32 r rb,widen32 lane lb,
    BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,BitVec.ofNat_mul_ofNat,
    BitVec.ofNat_add_ofNat,BitVec.toNat_ofNat,Nat.mod_eq_of_lt bound]

theorem input_bounds (x : Direct.Tensor) (laws : Direct.TensorLaws x)
    (h : InputContract x.metadata) (r c byte : Nat)
    (hr : r<rows x.metadata) (hc : c<columns x.metadata) (hb : byte<4) :
    (Direct.view x laws h).base%4=0 ∧
    (Direct.view x laws h).base+4*(r*columns x.metadata+c)+byte<2^64 := by
  have span := (Direct.view x laws h).spanFits
  change (Direct.view x laws h).base+4*x.metadata.numel≤2^64 at span
  have linear := linear_bound (rows x.metadata) (columns x.metadata) r c hr hc
  rw [numel_eq x.metadata h] at span
  exact ⟨VeriTile.Torch.FP32.dataView_aligned x laws h.contiguous h.alignment,by omega⟩

/-- Output allocators provide ordinary live byte windows and natural alignment;
all derived output source addresses lie inside these exact windows. -/
theorem output_bounds (p : Launch) (inp : Input p.rows p.columns) (before : State)
    (prepared : Prepared p inp before) (r c byte : Nat) (hr : r<p.rows) (hc : c<p.columns)
    (hb : byte<2) :
    prepared.quantized.tensor.base+r*p.columns+c<2^64 ∧
    prepared.maxima.tensor.base+2*r+byte<2^64 ∧
    prepared.maxima.tensor.base%2=0 := by
  have linear := linear_bound p.rows p.columns r c hr hc
  have qb := prepared.quantized.spanFits
  have mb := prepared.maxima.spanFits
  have ma := prepared.maxima.aligned
  simp only [DType.bytes,List.prod_cons,List.prod_nil,Nat.mul_one,Nat.one_mul] at qb mb ma
  omega

#axiomsClean offset_exact
#axiomsClean pointer_exact
#axiomsClean input_bounds
#axiomsClean output_bounds
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RowwiseQuantizationTriton.Native
