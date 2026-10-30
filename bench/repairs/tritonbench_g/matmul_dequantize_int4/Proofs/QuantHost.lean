import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.QuantFP
import bench.repairs.tritonbench_g.matmul_kernel.Proofs.Bounds
namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.QuantHost
open VeriTile.Triton
open QuantFP (Word)
noncomputable section
set_option maxHeartbeats 4000000
structure Call where
  shape : Quantizer.Shape
  W : View
  tpRank : Nat
  tree : Quantizer.Tree
structure Call.Valid (c : Call) : Prop where
  shape : c.shape.Valid
  wShape : c.W.rows=c.shape.K ∧ c.W.cols=c.shape.N
  view : c.W.Valid
  tree : c.tree.Valid c.shape.group

def input (c : Call) (m : Physical.Bytes) (k n : Nat) : Word :=
  ⟨Physical.decode32 (fun i : Fin 4 => m "W" (4*c.W.addr k n+i.val))⟩

def scalarResult (c : Call) (p : QuantFP.Instructions) (m : Physical.Bytes) : Quantizer.Result Word :=
  Quantizer.source (QuantFP.ops p c.tree) c.shape (input c m)

def encode32 := MatmulKernel.Physical.encode
/-- Public fresh-allocation/contiguous-copy semantics. Values are supplied by
the proved source loops. Each returned tensor has row-major signed32 or FP32
storage and all its bytes are written; caller/input bytes are preserved. -/
def publish (c : Call) (v : Quantizer.Result Word) (m : Physical.Bytes) : Physical.Bytes := fun r b =>
  let a := b/4
  let byte : Fin 4 := ⟨b%4,Nat.mod_lt _ (by decide)⟩
  if r="Qout" ∧ a<(c.shape.K/8)*c.shape.N then encode32 (v.Q (a/c.shape.N) (a%c.shape.N)) byte
  else if r="Sout" ∧ a<c.shape.groups*c.shape.N then encode32 (v.S (a/c.shape.N) (a%c.shape.N)).bits byte
  else if r="Zout" ∧ a<c.shape.groups*(c.shape.N/8) then encode32 (v.Z (a/(c.shape.N/8)) (a%(c.shape.N/8))) byte
  else m r b

inductive DType where | int32 | fp32 deriving DecidableEq, Repr
structure Tensor where
  identity : RegionName
  rows : Nat
  columns : Nat
  dtype : DType
  device : Nat
  deriving DecidableEq, Repr
structure Result where
  memory : Physical.Bytes
  returned : Tensor × Tensor × Tensor × Nat

def call (c : Call) (p : QuantFP.Instructions) (m : Physical.Bytes) : Result where
  memory := publish c (scalarResult c p m) m
  returned := (⟨"Qout",c.shape.K/8,c.shape.N,.int32,c.tpRank⟩,
    ⟨"Sout",c.shape.groups,c.shape.N,.fp32,c.tpRank⟩,
    ⟨"Zout",c.shape.groups,c.shape.N/8,.int32,c.tpRank⟩,c.shape.group)

def Writes (c : Call) (r : RegionName) (b : Nat) : Prop :=
  (r="Qout" ∧ b/4<(c.shape.K/8)*c.shape.N) ∨
  (r="Sout" ∧ b/4<c.shape.groups*c.shape.N) ∨
  (r="Zout" ∧ b/4<c.shape.groups*(c.shape.N/8))

theorem frame (c : Call) (p : QuantFP.Instructions) (m : Physical.Bytes) (r : RegionName) (b : Nat)
    (h : ¬Writes c r b) : (call c p m).memory r b=m r b := by
  have hq : ¬(r="Qout" ∧ b/4<(c.shape.K/8)*c.shape.N) := fun hh => h (Or.inl hh)
  have hs : ¬(r="Sout" ∧ b/4<c.shape.groups*c.shape.N) := fun hh => h (Or.inr (Or.inl hh))
  have hz : ¬(r="Zout" ∧ b/4<c.shape.groups*(c.shape.N/8)) := fun hh => h (Or.inr (Or.inr hh))
  simp only [call,publish,hq,hs,hz,ite_false]

theorem input_frame (c : Call) (p : QuantFP.Instructions) (m : Physical.Bytes) (b : Nat) :
    (call c p m).memory "W" b=m "W" b := by
  apply frame
  simp [Writes]

theorem flat_coordinates (rows cols row col byte : Nat) (hr : row<rows) (hc : col<cols) (hb : byte<4) :
    (4*(row*cols+col)+byte)/4<rows*cols ∧
    ((4*(row*cols+col)+byte)/4)/cols=row ∧
    ((4*(row*cols+col)+byte)/4)%cols=col ∧
    (4*(row*cols+col)+byte)%4=byte := by
  have hd : (4*(row*cols+col)+byte)/4=row*cols+col := by omega
  have hm : (4*(row*cols+col)+byte)%4=byte := by omega
  rw [hd,hm]
  refine ⟨MatmulKernel.tiled_bound row col rows cols hr hc,?_,?_,rfl⟩
  · rw [Nat.add_comm,Nat.add_mul_div_right _ _ (by omega),Nat.div_eq_of_lt hc,Nat.zero_add]
  · rw [Nat.add_comm,Nat.add_mul_mod_self_right,Nat.mod_eq_of_lt hc]

def Post (c : Call) (initial : Physical.Bytes) (result : Result) : Prop :=
  (∀ k n,k<c.shape.K/8 → n<c.shape.N → ∀ byte : Fin 4,
    result.memory "Qout" (4*(k*c.shape.N+n)+byte.val)=
      encode32 (Quantizer.targetPackedQ (QuantFP.ops QuantFP.software c.tree) c.shape (input c initial) k n) byte) ∧
  (∀ g n,g<c.shape.groups → n<c.shape.N → ∀ byte : Fin 4,
    result.memory "Sout" (4*(g*c.shape.N+n)+byte.val)=
      encode32 (Quantizer.targetScale (QuantFP.ops QuantFP.software c.tree) c.shape (input c initial) g n).bits byte) ∧
  (∀ g n,g<c.shape.groups → n<c.shape.N/8 → ∀ byte : Fin 4,
    result.memory "Zout" (4*(g*(c.shape.N/8)+n)+byte.val)=
      encode32 (Quantizer.targetPackedZ (QuantFP.ops QuantFP.software c.tree) c.shape (input c initial) g n) byte) ∧
  (∀ r b,¬Writes c r b → result.memory r b=initial r b) ∧
  result.returned=(⟨"Qout",c.shape.K/8,c.shape.N,.int32,c.tpRank⟩,
    ⟨"Sout",c.shape.groups,c.shape.N,.fp32,c.tpRank⟩,
    ⟨"Zout",c.shape.groups,c.shape.N/8,.int32,c.tpRank⟩,c.shape.group)

theorem whole_call (c : Call) (h : c.Valid) (p : QuantFP.Instructions) (ports : QuantFP.Obligations p)
    (m : Physical.Bytes) : Post c m (call c p m) := by
  have q := Quantizer.whole_quantizer (QuantFP.ops QuantFP.software c.tree) c.shape h.shape (input c m) h.tree
  have hp : scalarResult c p m=Quantizer.source (QuantFP.ops QuantFP.software c.tree) c.shape (input c m) :=
    QuantFP.local_transport p ports _ _ c.tree
  refine ⟨?_,?_,?_,frame c p m,rfl⟩
  · intro k n hk hn byte
    have hf := flat_coordinates (c.shape.K/8) c.shape.N k n byte.val hk hn byte.isLt
    simp only [call,publish,hf.1,ite_true,true_and,hf.2.1,hf.2.2.1,hp,q.1 k n hk hn]
    congr 1
    exact Fin.ext hf.2.2.2
  · intro g n hg hn byte
    have hf := flat_coordinates c.shape.groups c.shape.N g n byte.val hg hn byte.isLt
    simp only [call,publish,hf.1,true_and,ite_true,hf.2.1,hf.2.2.1,hp,q.2.1 g n hg hn]
    simp only [show ("Sout" : RegionName)≠"Qout" by decide,false_and,ite_false]
    congr 1
    exact Fin.ext hf.2.2.2
  · intro g n hg hn byte
    have hf := flat_coordinates c.shape.groups (c.shape.N/8) g n byte.val hg hn byte.isLt
    simp only [call,publish,hf.1,true_and,ite_true,hf.2.1,hf.2.2.1,hp,q.2.2.1 g n hg hn]
    simp only [show ("Zout" : RegionName)≠"Qout" by decide,show ("Zout" : RegionName)≠"Sout" by decide,false_and,ite_false]
    congr 1
    exact Fin.ext hf.2.2.2

#specNonCircular Post avoiding [call,publish,scalarResult,Quantizer.source]
#axiomsClean input_frame
#axiomsClean flat_coordinates
#axiomsClean whole_call
end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.QuantHost
