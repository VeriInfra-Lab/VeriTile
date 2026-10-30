import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Contract
import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Proofs.Direct.Geometry
import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.Direct.Bounds
namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.Bounds
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct
open Contract
noncomputable section
theorem matrix_numel (t : TensorInfo) (two : t.shape.length=2) : t.numel=rows t*columns t :=
  VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.Bounds.matrix_numel t.toTensorInfo two
namespace Word
abbrev signed64 := VeriTile.Bench.Repairs.BgmvExpandSlice.NativeIndex.signed64
abbrev widen32 := VeriTile.Bench.Repairs.BgmvExpandSlice.NativeIndex.widen32
end Word

theorem weights_numel (ok : InputContract x w y i s indices) : w.numel=rows w*(width w*rank w) := by
  rcases ok.ranks.2.2.2 with three|four
  · obtain ⟨a,b,c,shape⟩:=List.length_eq_three.mp three
    simp [VeriTile.Torch.TensorMetadata.numel,rows,width,rank,
      VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.rows,
      VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.width,
      VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.rank,shape]
  · obtain ⟨a,b,c,d,shape⟩:=List.length_eq_four.mp four
    have singleton : b=1 := by simpa [shape] using ok.singleton four
    simp [VeriTile.Torch.TensorMetadata.numel,rows,width,rank,
      VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.rows,
      VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.width,
      VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.rank,shape,singleton]

theorem output_inside (ok : Admitted x w y i s indices)
    (b : Fin (call x w y i s indices ok).batches) (n : Fin (call x w y i s indices ok).N) :
    Geometry.outputOffset (call x w y i s indices ok) b.val n.val < y.numel := by
  have c:=(python_checks_iff x w y i s indices).mp ok
  have hb : b.val < i.numel := b.isLt
  have hn : n.val < width w := n.isLt
  have br:=hb.trans_le c.shapes.2.2.1
  have nr:=hn.trans_le c.shapes.2.2.2
  rw [matrix_numel y c.ranks.2.1]
  change b.val*columns y+n.val < rows y*columns y
  nlinarith

theorem input_inside (ok : Admitted x w y i s indices)
    (b : Fin (call x w y i s indices ok).batches) (k : Nat) (hk : k < rank w) :
    Geometry.inputOffset (call x w y i s indices ok) b.val k < x.numel := by
  have c:=(python_checks_iff x w y i s indices).mp ok
  have hb : b.val < i.numel := b.isLt
  have br:=hb.trans_le c.shapes.2.1
  rw [matrix_numel x c.ranks.1]
  change b.val*rank w+k < rows x*columns x
  rw [c.shapes.1]
  nlinarith

theorem weight_inside (ok : Admitted x w y i s indices)
    (slot : Nat) (selected : slot < rows w) (n : Fin (call x w y i s indices ok).N)
    (k : Nat) (hk : k < rank w) :
    Geometry.weightOffset (call x w y i s indices ok) slot n.val k < w.numel := by
  have c:=(python_checks_iff x w y i s indices).mp ok
  have hn : n.val < width w := n.isLt
  have inner : n.val*rank w+k < width w*rank w := by nlinarith
  rw [weights_numel c]
  change slot*(width w*rank w)+n.val*rank w+k < rows w*(width w*rank w)
  have outer:=Nat.mul_le_mul_right (width w*rank w) (Nat.succ_le_of_lt selected)
  rw [Nat.succ_mul] at outer
  omega

theorem formed (ok : Admitted x w y i s indices)
    (p : Geometry.Program (call x w y i s indices ok)) (slot : Nat) (selected : slot < rows w)
    (n : Fin (block w)) (cycle : Fin (call x w y i s indices ok).cycles) (e : Fin 256) :
    let h:=call x w y i s indices ok
    let k:=Geometry.rank p.split.val cycle.val e.val
    2*Geometry.inputOffset h p.batch.val k < 2^63 ∧
    2*Geometry.weightOffset h slot n.val k < 2^63 ∧
    4*Geometry.outputOffset h p.batch.val n.val < 2^63 := by
  have c:=(python_checks_iff x w y i s indices).mp ok
  have nk : Geometry.rank p.split.val cycle.val e.val < formedK w := Geometry.rank_inside _ p.split cycle e
  have hb : p.batch.val < i.numel := p.batch.isLt
  have bx:=hb.trans_le c.shapes.2.1
  have by':=hb.trans_le c.shapes.2.2.1
  have xn:=matrix_numel x c.ranks.1
  have yn:=matrix_numel y c.ranks.2.1
  have wn:=weights_numel c
  have nn:=n.isLt
  have xb : p.batch.val*rank w ≤ x.numel := by rw [xn,c.shapes.1];nlinarith
  have wb : slot*(width w*rank w) ≤ w.numel := by
    rw [wn];exact Nat.mul_le_mul_right _ (Nat.le_of_lt selected)
  have yb : p.batch.val*columns y ≤ y.numel := by rw [yn];nlinarith
  have nb : n.val*rank w ≤ block w*rank w := Nat.mul_le_mul_right _ (Nat.le_of_lt nn)
  rcases c.formed with ⟨fx,fw,fy⟩
  change 2*(p.batch.val*rank w+Geometry.rank p.split.val cycle.val e.val) < 2^63 ∧
    2*(slot*(width w*rank w)+n.val*rank w+Geometry.rank p.split.val cycle.val e.val) < 2^63 ∧
    4*(p.batch.val*columns y+n.val) < 2^63
  omega

def rankWord (p : Geometry.Program h) (c e : Nat) : BitVec 64 :=
  BitVec.ofNat 64 c*BitVec.ofNat 64 16384+
    (BitVec.ofNat 64 e+BitVec.ofInt 64 (BitVec.ofNat 32 p.split.val).toInt*BitVec.ofNat 64 256)
def inputWord (h : Call) (p : Geometry.Program h) (c e : Nat) : BitVec 64 :=
  BitVec.ofInt 64 (BitVec.ofNat 32 p.batch.val).toInt*BitVec.ofNat 64 h.K+rankWord p c e
def weightWord (h : Call) (p : Geometry.Program h) (slot n c e : Nat) : BitVec 64 :=
  BitVec.ofNat 64 (h.N*h.K)*BitVec.ofNat 64 slot+BitVec.ofNat 64 n*BitVec.ofNat 64 h.K+rankWord p c e
def outputWord (h : Call) (p : Geometry.Program h) (n : Nat) : BitVec 64 :=
  BitVec.ofInt 64 (BitVec.ofNat 32 p.batch.val).toInt*BitVec.ofNat 64 h.columns+BitVec.ofNat 64 n

theorem word_offsets (ok : Admitted x w y i s indices)
    (p : Geometry.Program (call x w y i s indices ok)) (slot n c e : Nat) :
    let h:=call x w y i s indices ok
    let k:=Geometry.rank p.split.val c e
    inputWord h p c e=BitVec.ofNat 64 (Geometry.inputOffset h p.batch.val k) ∧
    weightWord h p slot n c e=BitVec.ofNat 64 (Geometry.weightOffset h slot n k) ∧
    outputWord h p n=BitVec.ofNat 64 (Geometry.outputOffset h p.batch.val n) := by
  have checks:=(python_checks_iff x w y i s indices).mp ok
  have hb : p.batch.val < i.numel := p.batch.isLt
  have b32 : p.batch.val < 2147483648 := by have bound:=checks.dimensions.1;omega
  have s32 : p.split.val < 2147483648 := by have bound:=p.split.isLt;omega
  dsimp only
  simp only [inputWord,weightWord,outputWord,rankWord,Word.widen32 _ b32,Word.widen32 _ s32,
    Geometry.inputOffset,Geometry.weightOffset,Geometry.outputOffset,Geometry.rank,
    BitVec.ofNat_add,BitVec.ofNat_mul,BitVec.mul_comm,true_and]

theorem signed_offsets (ok : Admitted x w y i s indices)
    (p : Geometry.Program (call x w y i s indices ok)) (slot : Nat) (selected : slot < rows w)
    (n : Fin (block w)) (cycle : Fin (call x w y i s indices ok).cycles) (e : Fin 256) :
    let h:=call x w y i s indices ok
    let k:=Geometry.rank p.split.val cycle.val e.val
    (inputWord h p cycle.val e.val).toInt=(Geometry.inputOffset h p.batch.val k:Int) ∧
    (weightWord h p slot n.val cycle.val e.val).toInt=(Geometry.weightOffset h slot n.val k:Int) ∧
    (outputWord h p n.val).toInt=(Geometry.outputOffset h p.batch.val n.val:Int) := by
  have bounds:=formed ok p slot selected n cycle e
  obtain ⟨wx,ww,wy⟩:=word_offsets ok p slot n.val cycle.val e.val
  dsimp only at bounds wx ww wy ⊢
  rw [wx,ww,wy]
  exact ⟨Word.signed64 _ (by dsimp [VeriTile.Bench.Repairs.BgmvExpandSlice.Storage.signedLimit];omega),
    Word.signed64 _ (by dsimp [VeriTile.Bench.Repairs.BgmvExpandSlice.Storage.signedLimit];omega),
    Word.signed64 _ (by dsimp [VeriTile.Bench.Repairs.BgmvExpandSlice.Storage.signedLimit];omega)⟩

#axiomsClean input_inside
#axiomsClean weight_inside
#axiomsClean output_inside
#axiomsClean formed
#axiomsClean word_offsets
#axiomsClean signed_offsets
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.Bounds
