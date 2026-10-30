import bench.tritonbench_g.bmm_chunk_fwd.Proofs.SourceAccess
import bench.tritonbench_g.bmm_chunk_fwd.Proofs.NativePointers

namespace VeriTile.Bench.Repairs.BmmChunkFwd.NativeAccess
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open SourcePointers NativePointers
noncomputable section

def aMask (h : Host.Call) (c : Host.Config) (p : Geometry.PID h c) (q i e : Nat) : Bool :=
  Finite.mask (coordinates h c p).m c.M i (limit h (coordinates h c p)) &&
    (BitVec.ofNat 64 e).slt (BitVec.ofNat 64 h.shape.K-BitVec.ofNat 64 q*BitVec.ofNat 64 c.K)
def bMask (h : Host.Call) (c : Host.Config) (p : Geometry.PID h c) (q e j : Nat) : Bool :=
  (BitVec.ofNat 64 e).slt (BitVec.ofNat 64 h.shape.K-BitVec.ofNat 64 q*BitVec.ofNat 64 c.K) &&
    Finite.mask (coordinates h c p).n c.N j (limit h (coordinates h c p))
def outMask (h : Host.Call) (c : Host.Config) (p : Geometry.PID h c) (i j : Nat) : Bool :=
  Finite.mask (coordinates h c p).m c.M i h.shape.chunk &&
    Finite.mask (coordinates h c p).n c.N j h.shape.chunk

theorem masks (h : Host.Call) (c : Host.Config) (hc : c∈Host.configurations)
    (sizes : Finite.AllocatedSizes h.shape) (p : Geometry.PID h c)
    (q : Nat) (hq : q<Host.cdiv h.shape.K c.K) (i : Fin c.M) (j : Fin c.N) (e : Fin c.K) :
    aMask h c p q i e=SourceAccess.aMask h c (coordinates h c p) q i e ∧
    bMask h c p q e j=SourceAccess.bMask h c (coordinates h c p) q e j ∧
    outMask h c p i j=SourceAccess.outMask h c (coordinates h c p) i j := by
  have indices := Finite.source_indices h c hc sizes
  have limit_le : limit h (coordinates h c p)≤h.shape.chunk := Nat.min_le_left _ _
  have hm := indices.rowMasks p i _ limit_le
  have hn := indices.columnMasks p j _ limit_le
  have hmo := indices.rowMasks p i _ (Nat.le_refl _)
  have hno := indices.columnMasks p j _ (Nat.le_refl _)
  have hk := indices.kMasks q hq e
  have base := Finite.block_base h.shape.K c.K q (Host.configuration_positive c hc).2.2 hq
  have keq : (q*c.K+e<h.shape.K)=(e.val<h.shape.K-q*c.K) := by apply propext; omega
  dsimp only [coordinates] at hm hn
  simp only [aMask,bMask,outMask,SourceAccess.aMask,SourceAccess.bMask,SourceAccess.outMask,
    coordinates,row,col,hm,hn,hmo,hno,hk,keq,and_self]
  exact ⟨rfl,rfl,rfl⟩

theorem a_active (h : Host.Call) (c : Host.Config) (hc : c∈Host.configurations)
    (sizes : Finite.AllocatedSizes h.shape) (layout : Layout h) (p : Geometry.PID h c)
    (q i e : Nat) (hm : SourceAccess.aMask h c (coordinates h c p) q i e=Bool.true) :
    Address layout.a (aWord h c (coordinates h c p) q i e) (aOffset h c (coordinates h c p) q i e) := by
  simp only [SourceAccess.aMask,Bool.and_eq_true,decide_eq_true_eq] at hm
  exact NativePointers.a_active h c hc sizes layout p q i e hm.1 (by omega)

theorem output_mask (h : Host.Call) (c : Host.Config) (hc : c∈Host.configurations)
    (sizes : Finite.AllocatedSizes h.shape) (p : Geometry.PID h c) (i : Fin c.M) (j : Fin c.N) :
    outMask h c p i j=SourceAccess.outMask h c (coordinates h c p) i j := by
  have indices := Finite.source_indices h c hc sizes
  have hm := indices.rowMasks p i h.shape.chunk (Nat.le_refl _)
  have hn := indices.columnMasks p j h.shape.chunk (Nat.le_refl _)
  simp only [outMask,SourceAccess.outMask,coordinates,row,col,hm,hn]
  rfl

theorem seq_masks (h : Host.Call) (c : Host.Config) (hc : c∈Host.configurations)
    (sizes : Finite.AllocatedSizes h.shape) (p : Geometry.PID h c) (i : Fin c.M) (j : Fin c.N) :
    Finite.mask (coordinates h c p).m c.M i (limit h (coordinates h c p))=
      decide (row c (coordinates h c p) i<limit h (coordinates h c p)) ∧
    Finite.mask (coordinates h c p).n c.N j (limit h (coordinates h c p))=
      decide (col c (coordinates h c p) j<limit h (coordinates h c p)) :=
  ⟨(Finite.source_indices h c hc sizes).rowMasks p i _ (Nat.min_le_left _ _),
    (Finite.source_indices h c hc sizes).columnMasks p j _ (Nat.min_le_left _ _)⟩

theorem b_active (h : Host.Call) (c : Host.Config) (hc : c∈Host.configurations)
    (sizes : Finite.AllocatedSizes h.shape) (layout : Layout h) (p : Geometry.PID h c)
    (q e j : Nat) (hm : SourceAccess.bMask h c (coordinates h c p) q e j=Bool.true) :
    Address layout.b (bWord h c (coordinates h c p) q e j) (bOffset h c (coordinates h c p) q e j) := by
  simp only [SourceAccess.bMask,Bool.and_eq_true,decide_eq_true_eq] at hm
  exact NativePointers.b_active h c hc sizes layout p q e j hm.2 (by omega)

theorem out_active (h : Host.Call) (c : Host.Config) (layout : Layout h) (p : Geometry.PID h c)
    (i j : Nat) (hm : SourceAccess.outMask h c (coordinates h c p) i j=Bool.true) :
    Address layout.out (outWord h c (coordinates h c p) i j) (outOffset h c (coordinates h c p) i j) := by
  simp only [SourceAccess.outMask,Bool.and_eq_true,decide_eq_true_eq] at hm
  exact NativePointers.out_active h c layout p i j hm.1 hm.2

theorem a_inactive {α : Type} [Inhabited α] (o : FPSuccess.Ops α) (h : Host.Call)
    (c : Host.Config) (d : Coordinates) (q i e : Nat) (s : State α)
    (hm : SourceAccess.aMask h c d q i e=Bool.false) :
    SourceAccess.aValue o h c d q s i e=
      o.scalar.cast (some .fp32) .real .real (o.scalar.literal (some .fp32) .real 0) := by
  simp only [SourceAccess.aValue,hm,Bool.false_eq_true,↓reduceIte]

theorem b_inactive {α : Type} [Inhabited α] (o : FPSuccess.Ops α) (h : Host.Call)
    (c : Host.Config) (d : Coordinates) (q e j : Nat) (s : State α)
    (hm : SourceAccess.bMask h c d q e j=Bool.false) :
    SourceAccess.bValue o h c d q s e j=
      o.scalar.cast (some .fp32) .real .real (o.scalar.literal (some .fp32) .real 0) := by
  simp only [SourceAccess.bValue,hm,Bool.false_eq_true,↓reduceIte]

#axiomsClean masks
#axiomsClean a_active
#axiomsClean b_active
#axiomsClean out_active
#axiomsClean a_inactive
#axiomsClean b_inactive
end
end VeriTile.Bench.Repairs.BmmChunkFwd.NativeAccess

#guard_msgs (drop info) in
#auditModuleAxioms
