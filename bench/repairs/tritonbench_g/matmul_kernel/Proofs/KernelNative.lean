import bench.repairs.tritonbench_g.matmul_kernel.Proofs.KernelScalar
import bench.tritonbench_g.matmul_dequant_int4.Proofs.Binary16
import bench.tritonbench_g.matrix_reduction.Proofs.BinaryConversion
import VeriTile.Triton.Float.ScalarBackend
namespace VeriTile.Bench.Repairs.MatmulKernel.KernelNative
open VeriTile.Triton FP.Structural KernelSource KernelScalar
open VeriTile.Bench.TritonBenchG
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
abbrev Word := FP.Value .fp32
abbrev Half := MatmulDequantInt4.Binary16.Value
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Documented, fixed-shape TF32 instruction-lane semantics is the native
model boundary. The operands are the original FP32 register words, exactly as
in emitted PTX: the source contains no explicit cvt.tf32. The model accounts
for the instruction's TF32 operand interpretation, internal product/reduction
rounding, subnormals, signed zeros and NaNs. This is a fixed K=8 primitive,
not a variable-size dot, a whole-kernel oracle or an output-equality premise.
Compiler selection of opcode, lane mapping and K8 grouping remains an explicit
lowering boundary; the source carry/operand graph is proved below. -/
structure Instructions where
  mma8 : (op : Opcode) → Fin op.rows → Fin op.cols →
    (Fin 8 → Word) → (Fin 8 → Word) → Word → Word

def half (x : Word) : Half := ⟨x.bits.truncate 16⟩
def embed (x : Half) : Word := ⟨x.bits.zeroExtend 32⟩
def widen (x : Word) : Word := ⟨MatrixReduction.BinaryConversion.widen (half x).bits⟩
def narrow (x : Word) : Word := embed <|
  match x.decode with
  | .nan .. => MatmulDequantInt4.Binary16.Value.qNaN
  | .infinity negative => MatmulDequantInt4.Binary16.Value.infinity negative
  | .finite negative magnitude =>
      MatmulDequantInt4.Binary16.round .ieeeValue (if negative then -magnitude else magnitude) negative

/-- FP16 conversion is RNE with gradual underflow, canonical NaN and exact
bit-format widening. Native compiler/GPU conformance to this named conversion
profile is a model boundary; no conversion callback is assumed by the theorem. -/
def scalar : Algebra Word :=
  {FP.Software.scalarFP32 (fallback (FP.Value.zero .fp32)) .ieeeValue with
    cast:=fun _ src dst x=>match src,dst with
      | .real,.fp16=>narrow x
      | .fp16,.real=>widen x
      | _,_=>x}

def dot (p : Instructions) (op : Opcode) : LaneDot Word := fun r n k a b acc=>
  (List.range (k/8)).foldl (fun z j=>
    p.mma8 op ⟨r%op.rows,Nat.mod_lt _ op.rows_pos⟩ ⟨n%op.cols,Nat.mod_lt _ op.cols_pos⟩
      (fun e=>if h:j*8+e.val<k then a ⟨j*8+e.val,h⟩ else FP.Value.zero .fp32)
      (fun e=>if h:j*8+e.val<k then b ⟨j*8+e.val,h⟩ else FP.Value.zero .fp32) z) acc

/-- Every K8 operand index exists; admitted block sizes cover every source
lane exactly once. This bound justifies the finite instruction graph. -/
theorem instruction_lane_bound (p : Config) (hc : p.Valid) (j : Nat)
    (hj : j<p.BK/8) (e : Fin 8) : j*8+e.val<p.BK := by
  have h:=tiled_bound j e.val (p.BK/8) 8 hj e.isLt
  simpa only [hc.atoms] using h

/-- Independent coordinate target: source-ordered block/K8 instruction
recurrence, original register words, every padded lane, live FP32 carry and
both final conversions. No source state, AST, execution or output appears. -/
def target (instructions : Instructions) (p : Config) (a b : Nat → Nat → Word) (r n : Nat) : Word :=
  widen (narrow (foldRange (fun q acc=>
    (List.range (p.BK/8)).foldl (fun z j=>
      instructions.mma8 p.opcode
        ⟨(r%p.BM)%p.opcode.rows,Nat.mod_lt _ p.opcode.rows_pos⟩
        ⟨(n%p.BN)%p.opcode.cols,Nat.mod_lt _ p.opcode.cols_pos⟩
        (fun e=>if j*8+e.val<p.BK then
          if q*p.BK+(j*8+e.val)<p.K then a r (q*p.BK+(j*8+e.val)) else FP.Value.zero .fp32
          else FP.Value.zero .fp32)
        (fun e=>if j*8+e.val<p.BK then
          if q*p.BK+(j*8+e.val)<p.K then b (q*p.BK+(j*8+e.val)) n else FP.Value.zero .fp32
          else FP.Value.zero .fp32) z) acc)
    0 ((p.K+p.BK-1)/p.BK) 1 (FP.Value.zero .fp32)))

theorem reference_target (instructions : Instructions) (p : Config)
    (a b : Nat → Nat → Word) (r n : Nat) :
    reference scalar (dot instructions p.opcode) p a b r n=target instructions p a b r n := by
  simp only [reference,scalar,FP.Software.scalarFP32,FP.Software.fp32Context,if_true,
    dot,target,ceil,id_eq]
  rfl

theorem launch_correct (instructions : Instructions) (p : Config) (hc : p.Valid) (s : State Word) :
    ∃ t,KernelSource.launch scalar (dotAcc (dot instructions p.opcode)) p s=some t ∧
      (∀ w,p.Active w → t.mem "C" (p.C.addr w.1 w.2)=stored
        (target instructions p (fun r k=>read (s.mem "A" (p.A.addr r k)))
          (fun k n=>read (s.mem "B" (p.B.addr k n))) w.1 w.2)) ∧
      (∀ (region : RegionName) offset,(region≠"C" ∨ ∀ w,p.Active w → offset≠p.C.addr w.1 w.2) →
        t.mem region offset=s.mem region offset) := by
  obtain ⟨t,he,hr,hf⟩:=KernelSource.launch_correct scalar (dotAcc (dot instructions p.opcode)) p hc s
  refine ⟨t,he,?_,hf⟩
  intro w hw
  rw [hr w hw,point_reference _ _ _ hc s w hw,reference_target]

theorem conversion_vectors :
    (narrow (FP.Value.ofBits .fp32 0x3f800000)).bits=BitVec.ofNat 32 0x3c00 ∧
    (narrow (FP.Value.ofBits .fp32 0x33800000)).bits=BitVec.ofNat 32 1 ∧
    (narrow (FP.Value.ofBits .fp32 0x33000000)).bits=BitVec.ofNat 32 0 ∧
    (narrow (FP.Value.ofBits .fp32 0x477ff000)).bits=BitVec.ofNat 32 0x7c00 ∧
    (narrow (FP.Value.ofBits .fp32 0x80000000)).bits=BitVec.ofNat 32 0x8000 ∧
    (widen (embed (MatmulDequantInt4.Binary16.Value.ofBits 1))).bits=BitVec.ofNat 32 0x33800000 ∧
    (widen (embed (MatmulDequantInt4.Binary16.Value.ofBits 0x8000))).bits=BitVec.ofNat 32 0x80000000 := by
  decide +kernel

#specNonCircular target avoiding [KernelSource.source,KernelSource.outputValue,KernelSource.point,KernelSource.launch,reference,dot]
#axiomsClean reference_target
#axiomsClean launch_correct
#axiomsClean conversion_vectors
end
end VeriTile.Bench.Repairs.MatmulKernel.KernelNative
