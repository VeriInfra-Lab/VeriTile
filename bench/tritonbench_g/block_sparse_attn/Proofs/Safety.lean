import bench.tritonbench_g.block_sparse_attn.Proofs.Source
import bench.tritonbench_g.block_sparse_attn.Proofs.Native
namespace VeriTile.Bench.Repairs.BlockSparseAttn.Safety
set_option maxHeartbeats 8000000

def I32 (z : Int) : Prop := -2^31≤z ∧ z<2^31
theorem signed32_exact (z : Int) (h : I32 z) : (BitVec.ofInt 32 z).toInt=z :=
  BitVec.toInt_ofInt_eq_self (by decide) h.1 h.2

structure Storage where
  offset : Nat
  bytes : Nat
def Allocated (s : Storage) (z : Int) : Prop :=
  0≤z ∧ 4*(s.offset+z.toNat+1)≤s.bytes
theorem byte_allocated (s : Storage) (z : Int) (h : Allocated s z) (i : Fin 4) :
    4*(s.offset+z.toNat)+i.val<s.bytes := by
  have hi:=i.isLt
  have hb:=h.2
  omega
theorem byte_signed64 (z : Int) (h : I32 z) (i : Fin 4) :
    -(2^63:Int)≤4*z+(i.val:Int) ∧ 4*z+(i.val:Int)<2^63 := by
  have hi:=i.isLt
  rcases h with ⟨lo,hi32⟩
  omega

/-- All integer subexpressions which depend on CSR data in the source's
column load, key formation, K/V offset updates and loop counter. This is a
finite arithmetic input condition, not an execution/result premise. -/
def dataNodes (x : Layout) (l : Lane) (visit dblock dlane nlane : Nat) : List Int :=
  let w:=laneWork x.shape l
  let base:=((w.head%x.shape.layouts)*x.csr.colStride : Nat)
  let first:=x.csr.first x.shape w.head w.row
  let last:=x.csr.last x.shape w.head w.row
  let iter:=first+(visit:Int)
  let slot:=(base:Int)+iter
  let col:=x.csr.cols slot
  let start:=col*(x.shape.blockN:Int)
  let key:=start+(nlane:Int)
  let kc:=dblock*x.shape.blockD+dlane
  let vc:=l.channelProgram*x.shape.blockD+dlane
  let kb:=w.batch*x.k.batch+kvHead x.shape w.head*x.k.head
  let vb:=w.batch*x.v.batch+kvHead x.shape w.head*x.v.head
  [base,first,last,iter,iter+1,slot,col,start,key,
   (nlane*x.k.row : Nat),(dlane*x.k.dim : Nat),
   (nlane*x.k.row+dlane*x.k.dim : Nat),start*(x.k.row:Int),
   (dblock*x.shape.blockD*x.k.dim : Nat),(kc*x.k.dim : Nat),kb,
   (kb:Int)+(nlane*x.k.row+dlane*x.k.dim : Nat),
   (kb:Int)+(nlane*x.k.row+dlane*x.k.dim : Nat)+start*(x.k.row:Int),
   signedAddress x.k w.batch (kvHead x.shape w.head) key kc,
   (nlane*x.v.row : Nat),(vc*x.v.dim : Nat),
   (nlane*x.v.row+vc*x.v.dim : Nat),start*(x.v.row:Int),vb,
   (vb:Int)+(nlane*x.v.row+vc*x.v.dim : Nat),
   signedAddress x.v w.batch (kvHead x.shape w.head) key vc]

structure DataContract (x : Layout) (rows cols k v : Storage) : Prop where
  rowEndpoints : ∀ l,l.Valid x.shape →
    let w:=laneWork x.shape l
    let p:=((w.head%x.shape.layouts)*x.csr.rowStride+l.rowProgram : Nat)
    I32 p ∧ I32 (p+1) ∧ Allocated rows p ∧ Allocated rows (p+1)
  formed : ∀ l,l.Valid x.shape → ∀ visit,visit<x.csr.count x.shape (laneWork x.shape l).head (laneWork x.shape l).row →
    ∀ db,db<x.shape.blocks → ∀ d,d<x.shape.blockD → ∀ n,n<x.shape.blockN →
      ∀ z∈dataNodes x l visit db d n,I32 z
  columns : ∀ l,l.Valid x.shape → ∀ visit,visit<x.csr.count x.shape (laneWork x.shape l).head (laneWork x.shape l).row →
    Allocated cols (x.csr.slot x.shape (laneWork x.shape l).head (laneWork x.shape l).row visit)
  keys : ∀ l,l.Valid x.shape → ∀ visit,visit<x.csr.count x.shape (laneWork x.shape l).head (laneWork x.shape l).row →
    ∀ db,db<x.shape.blocks → ∀ d,d<x.shape.blockD → ∀ n,n<x.shape.blockN →
    let w:=laneWork x.shape l
    let key:=x.csr.key x.shape w.head w.row visit n
    (x.evenN=Bool.true ∨ key<(x.shape.length:Int)) →
      Allocated k (signedAddress x.k w.batch (kvHead x.shape w.head) key (db*x.shape.blockD+d)) ∧
      Allocated v (signedAddress x.v w.batch (kvHead x.shape w.head) key (l.channelProgram*x.shape.blockD+d))

theorem all_data_nodes_exact (x : Layout) (r c k v : Storage) (h : DataContract x r c k v)
    (l : Lane) (hl : l.Valid x.shape) (visit : Nat)
    (hv : visit<x.csr.count x.shape (laneWork x.shape l).head (laneWork x.shape l).row)
    (db d n : Nat) (hdb : db<x.shape.blocks) (hd : d<x.shape.blockD) (hn : n<x.shape.blockN)
    (z : Int) (hz : z∈dataNodes x l visit db d n) :
    (BitVec.ofInt 32 z).toInt=z :=
  signed32_exact z (h.formed l hl visit hv db hdb d hd n hn z hz)

def decodeCSR (rowStride colStride : Nat) (m : Native.Bytes) : CSR where
  rowStride:=rowStride
  colStride:=colStride
  rows:=fun i=>(Native.load m .l i.toNat).toInt
  cols:=fun i=>(Native.load m (.other 0) i.toNat).toInt

theorem csr_readonly (x : Layout) (p : Ops Word) (scale : Word) (ws : List Work) (m : Native.Bytes)
    (rs cs : Nat) : decodeCSR rs cs (Native.byteExecute x p scale ws m)=decodeCSR rs cs m := by
  have hr:=execute_readonly x p scale ws (Native.load m) (Native.load m) (fun _ _ _=>rfl)
  rw [←Native.execute_load] at hr
  have hl : Native.load (Native.byteExecute x p scale ws m) .l=Native.load m .l :=
    funext (fun a=>hr .l a (by decide))
  have hm : Native.load (Native.byteExecute x p scale ws m) (.other 0)=Native.load m (.other 0) :=
    funext (fun a=>hr (.other 0) a (by decide))
  simp only [decodeCSR,hl,hm]

theorem grouped_head_bound (s : Shape) (positive : 0<s.kvHeads) (divides : s.kvHeads∣s.heads)
    (h : Nat) (hh : h<s.heads) : kvHead s h<s.kvHeads := by
  have he:=Nat.div_mul_cancel divides
  have hp : 0<s.heads/s.kvHeads := by nlinarith
  apply (Nat.div_lt_iff_lt_mul hp).mpr
  nlinarith [he]

#axiomsClean signed32_exact
#axiomsClean byte_allocated
#axiomsClean byte_signed64
#axiomsClean all_data_nodes_exact
#axiomsClean csr_readonly
#axiomsClean grouped_head_bound
#specNonCircular DataContract avoiding [execute,rowValue,Native.byteExecute,Spec.output]
end VeriTile.Bench.Repairs.BlockSparseAttn.Safety
