import bench.tritonbench_g.bgmv_expand_slice.Contract
import bench.tritonbench_g.bgmv_expand_slice.Proofs.Direct.Geometry
import bench.tritonbench_g.bgmv_expand_slice.Proofs.NativeIndex
import Mathlib.Tactic.Linarith

namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Direct.Bounds
open Contract
noncomputable section
set_option maxHeartbeats 4000000

theorem matrix_numel (t : TensorInfo) (two : t.shape.length=2) :
    t.numel=rows t*columns t := by
  obtain ⟨a,b,shape⟩:=List.length_eq_two.mp two
  simp [VeriTile.Torch.TensorMetadata.numel,rows,columns,shape]

theorem weights_numel (ok : InputContract x w y i s indices) :
    w.numel=rows w*(width w*rank w) := by
  rcases ok.ranks.2.2.2 with three|four
  · obtain ⟨a,b,c,shape⟩:=List.length_eq_three.mp three
    simp [VeriTile.Torch.TensorMetadata.numel,rows,width,rank,shape]
  · obtain ⟨a,b,c,d,shape⟩:=List.length_eq_four.mp four
    have singleton : b=1 := by simpa [shape] using ok.singleton four
    simp [VeriTile.Torch.TensorMetadata.numel,rows,width,rank,shape,singleton]

def inputOffset (h : Call) (p : Geometry.Program h) (k : Nat) := p.batch*h.K+k
def weightOffset (h : Call) (p : Geometry.Program h) (slot n k : Nat) :=
  slot*(h.N*h.K)+p.row*h.K+n*h.K+k
def outputOffset (h : Call) (p : Geometry.Program h) (n : Nat) := p.batch*h.columns+p.row+h.slice+n

theorem active (x w y i : TensorInfo) (s : Scalars) (indices : Nat → Int)
    (ok : Admitted x w y i s indices) (p : Geometry.Program (call x w y i s indices ok))
    (slot : Nat) (selected : slot<rows w) (k : Nat) (hk : k<rank w) :
    inputOffset (call x w y i s indices ok) p k<x.numel ∧
    weightOffset (call x w y i s indices ok) p slot 0 k<w.numel ∧
    outputOffset (call x w y i s indices ok) p 0<y.numel := by
  have c:=(python_checks_iff x w y i s indices).mp ok
  have hb : p.batch<i.numel := p.batchInside
  have hr : p.row<width w := p.rowInside
  have bx : p.batch<rows x := hb.trans_le c.shapes.2.2.1
  have by' : p.batch<rows y := hb.trans_le c.shapes.2.2.2.1
  have slice : p.row+s.offset.toNat<columns y := by have hs:=c.shapes.2.2.2.2.2;omega
  have wx:=matrix_numel x c.ranks.1
  have wy:=matrix_numel y c.ranks.2.1
  have ww:=weights_numel c
  change p.batch*rank w+k<x.numel ∧
    slot*(width w*rank w)+p.row*rank w+0*rank w+k<w.numel ∧
    p.batch*columns y+p.row+s.offset.toNat+0<y.numel
  rw [wx,wy,ww,c.shapes.2.1]
  have inner : p.row*rank w+k<width w*rank w := by nlinarith
  constructor
  · nlinarith
  constructor
  · nlinarith
  · nlinarith

/-- These bounds cover every formed lane, including masked-off lanes.
Only active lanes need an allocated byte span. -/
theorem formed (x w y i : TensorInfo) (s : Scalars) (indices : Nat → Int)
    (ok : Admitted x w y i s indices) (p : Geometry.Program (call x w y i s indices ok))
    (slot : Nat) (selected : slot<rows w) (n : Fin 256) (k : Fin (block w)) :
    4*inputOffset (call x w y i s indices ok) p k.val<2^63 ∧
    2*weightOffset (call x w y i s indices ok) p slot n.val k.val<2^63 ∧
    4*outputOffset (call x w y i s indices ok) p n.val<2^63 := by
  have c:=(python_checks_iff x w y i s indices).mp ok
  have a:=active x w y i s indices ok p slot selected 0 c.shapes.1
  have nk : k.val<block w := k.isLt
  have nn : n.val≤255 := by have hn:=n.isLt;omega
  have mult:=Nat.mul_le_mul_right (rank w) nn
  rcases c.formed with ⟨fx,fw,fy⟩
  dsimp [inputOffset,weightOffset,outputOffset,call] at a ⊢
  dsimp only [block] at nk fx fw
  constructor
  · omega
  constructor
  · omega
  · omega

def inputWord (h : Call) (p : Geometry.Program h) (k : Nat) : BitVec 64 :=
  BitVec.ofInt 64 (BitVec.ofNat 32 p.batch).toInt*BitVec.ofNat 64 h.K+BitVec.ofNat 64 k
def weightWord (h : Call) (p : Geometry.Program h) (slot n k : Nat) : BitVec 64 :=
  BitVec.ofNat 64 (h.N*h.K)*BitVec.ofNat 64 slot+
    BitVec.ofInt 64 (BitVec.ofNat 32 p.row).toInt*BitVec.ofNat 64 1*BitVec.ofNat 64 h.K+
    BitVec.ofNat 64 n*BitVec.ofNat 64 h.K+BitVec.ofNat 64 k
def outputWord (h : Call) (p : Geometry.Program h) (n : Nat) : BitVec 64 :=
  BitVec.ofInt 64 (BitVec.ofNat 32 p.batch).toInt*BitVec.ofNat 64 h.columns+
    BitVec.ofInt 64 (BitVec.ofNat 32 p.row).toInt*BitVec.ofNat 64 1+
    BitVec.ofNat 64 h.slice*BitVec.ofNat 64 1+BitVec.ofNat 64 n

theorem word_offsets (x w y i : TensorInfo) (s : Scalars) (indices : Nat → Int)
    (ok : Admitted x w y i s indices) (p : Geometry.Program (call x w y i s indices ok))
    (slot n k : Nat) :
    inputWord (call x w y i s indices ok) p k=BitVec.ofNat 64 (inputOffset (call x w y i s indices ok) p k) ∧
    weightWord (call x w y i s indices ok) p slot n k=BitVec.ofNat 64 (weightOffset (call x w y i s indices ok) p slot n k) ∧
    outputWord (call x w y i s indices ok) p n=BitVec.ofNat 64 (outputOffset (call x w y i s indices ok) p n) := by
  have c:=(python_checks_iff x w y i s indices).mp ok
  have hb : p.batch<i.numel := p.batchInside
  have hr : p.row<width w := p.rowInside
  have b32 : p.batch<2147483648 := by have h:=c.dimensions.1;omega
  have r32 : p.row<2147483648 := by have h:=c.dimensions.2.2.1;omega
  simp only [inputWord,weightWord,outputWord,VeriTile.Bench.Repairs.BgmvExpandSlice.NativeIndex.widen32 _ b32,VeriTile.Bench.Repairs.BgmvExpandSlice.NativeIndex.widen32 _ r32,
    inputOffset,weightOffset,outputOffset,BitVec.ofNat_add,BitVec.ofNat_mul,
    BitVec.mul_one,BitVec.one_mul,BitVec.mul_comm,true_and,and_true]

theorem signed_offsets (x w y i : TensorInfo) (s : Scalars) (indices : Nat → Int)
    (ok : Admitted x w y i s indices) (p : Geometry.Program (call x w y i s indices ok))
    (slot : Nat) (selected : slot<rows w) (n : Fin 256) (k : Fin (block w)) :
    (inputWord (call x w y i s indices ok) p k.val).toInt=(inputOffset (call x w y i s indices ok) p k.val:Int) ∧
    (weightWord (call x w y i s indices ok) p slot n.val k.val).toInt=(weightOffset (call x w y i s indices ok) p slot n.val k.val:Int) ∧
    (outputWord (call x w y i s indices ok) p n.val).toInt=(outputOffset (call x w y i s indices ok) p n.val:Int) := by
  have f:=formed x w y i s indices ok p slot selected n k
  obtain ⟨wx,ww,wy⟩:=word_offsets x w y i s indices ok p slot n.val k.val
  rw [wx,ww,wy]
  exact ⟨VeriTile.Bench.Repairs.BgmvExpandSlice.NativeIndex.signed64 _ (by dsimp [VeriTile.Bench.Repairs.BgmvExpandSlice.Storage.signedLimit];omega),
    VeriTile.Bench.Repairs.BgmvExpandSlice.NativeIndex.signed64 _ (by dsimp [VeriTile.Bench.Repairs.BgmvExpandSlice.Storage.signedLimit];omega),
    VeriTile.Bench.Repairs.BgmvExpandSlice.NativeIndex.signed64 _ (by dsimp [VeriTile.Bench.Repairs.BgmvExpandSlice.Storage.signedLimit];omega)⟩

#axiomsClean matrix_numel
#axiomsClean weights_numel
#axiomsClean active
#axiomsClean formed
#axiomsClean word_offsets
#axiomsClean signed_offsets
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Direct.Bounds
