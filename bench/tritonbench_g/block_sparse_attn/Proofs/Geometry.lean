import bench.tritonbench_g.block_sparse_attn.Proofs.Spec
namespace VeriTile.Bench.Repairs.BlockSparseAttn
set_option maxHeartbeats 4000000

structure Work where
  batch : Nat
  head : Nat
  row : Nat
  col : Nat
  deriving DecidableEq
def Work.active (s : Shape) (w : Work) : Prop :=
  w.batch<s.batches ∧ w.head<s.heads ∧ w.row<s.length ∧ w.col<s.dim
instance (s : Shape) (w : Work) : Decidable (w.active s) := by unfold Work.active;infer_instance
def Work.target (s : Shape) (w : Work) := outputAddress s w.batch w.head w.row w.col
structure Lane where
  rowProgram : Nat
  headProgram : Nat
  channelProgram : Nat
  rowLane : Nat
  colLane : Nat
  deriving DecidableEq
def ceil (n b : Nat) := (n+b-1)/b
def Lane.Valid (s : Shape) (l : Lane) : Prop :=
  l.rowProgram<ceil s.length s.blockM ∧ l.headProgram<s.batches*s.heads ∧
  l.channelProgram<s.blocks ∧ l.rowLane<s.blockM ∧ l.colLane<s.blockD
def laneWork (s : Shape) (l : Lane) : Work :=
  ⟨l.headProgram/s.heads,l.headProgram%s.heads,
    l.rowProgram*s.blockM+l.rowLane,l.channelProgram*s.blockD+l.colLane⟩

theorem decode (base a b : Nat) (hb : b<base) :
    (a*base+b)/base=a ∧ (a*base+b)%base=b :=
  FlashAttn.flatten_decode base a b hb

theorem channel_coverage (s : Shape) (hp : 0<s.blockD) (col : Nat) (hc : col<s.dim) :
    col/s.blockD<s.blocks ∧ col%s.blockD<s.blockD ∧
      col/s.blockD*s.blockD+col%s.blockD=col := by
  have hd : col<s.blocks*s.blockD := by simpa [Shape.dim,Nat.mul_comm] using hc
  exact ⟨(Nat.div_lt_iff_lt_mul hp).mpr hd,Nat.mod_lt _ hp,
    by simpa only [Nat.mul_comm] using Nat.div_add_mod col s.blockD⟩

theorem lane_coverage (s : Shape) (hm : 0<s.blockM) (hd : 0<s.blockD)
    (w : Work) (active : w.active s) : ∃ l,l.Valid s ∧ laneWork s l=w := by
  rcases active with ⟨hb,hh,hr,hc⟩
  obtain ⟨hcb,hcl,hcd⟩:=channel_coverage s hd w.col hc
  have headbound : w.batch*s.heads+w.head<s.batches*s.heads := by
    have := Nat.mul_le_mul_right s.heads (show w.batch+1≤s.batches by omega)
    nlinarith
  refine ⟨⟨w.row/s.blockM,w.batch*s.heads+w.head,w.col/s.blockD,w.row%s.blockM,w.col%s.blockD⟩,?_,?_⟩
  · refine ⟨?_,headbound,hcb,Nat.mod_lt _ hm,hcl⟩
    change w.row/s.blockM<(s.length+s.blockM-1)/s.blockM
    apply Nat.lt_of_succ_le
    apply (Nat.le_div_iff_mul_le hm).mpr
    have hh:=Nat.div_mul_le_self w.row s.blockM
    have hs : s.length+s.blockM-1+1=s.length+s.blockM := by omega
    nlinarith
  · have hhdec:=decode s.heads w.batch w.head hh
    have hrdec:=Nat.div_add_mod w.row s.blockM
    cases w
    simp_all [laneWork,Nat.mul_comm]

theorem lane_injective (s : Shape) (a b : Lane) (ha : a.Valid s) (hb : b.Valid s)
    (he : laneWork s a=laneWork s b) : a=b := by
  have headsA:=Nat.div_add_mod a.headProgram s.heads
  have headsB:=Nat.div_add_mod b.headProgram s.heads
  have rowsA:=decode s.blockM a.rowProgram a.rowLane ha.2.2.2.1
  have rowsB:=decode s.blockM b.rowProgram b.rowLane hb.2.2.2.1
  have colsA:=decode s.blockD a.channelProgram a.colLane ha.2.2.2.2
  have colsB:=decode s.blockD b.channelProgram b.colLane hb.2.2.2.2
  have eq:=Work.mk.inj he
  have ehead : a.headProgram=b.headProgram := by
    rw [eq.1,eq.2.1] at headsA
    omega
  have erow : a.rowProgram=b.rowProgram ∧ a.rowLane=b.rowLane := by
    have eqr:=congrArg Work.row he
    change a.rowProgram*s.blockM+a.rowLane=b.rowProgram*s.blockM+b.rowLane at eqr
    have hd:=congrArg (fun x=>x/s.blockM) eqr
    have hm:=congrArg (fun x=>x%s.blockM) eqr
    simpa only [rowsA.1,rowsB.1,rowsA.2,rowsB.2] using And.intro hd hm
  have ecol : a.channelProgram=b.channelProgram ∧ a.colLane=b.colLane := by
    have eqc:=congrArg Work.col he
    change a.channelProgram*s.blockD+a.colLane=b.channelProgram*s.blockD+b.colLane at eqc
    have hd:=congrArg (fun x=>x/s.blockD) eqc
    have hm:=congrArg (fun x=>x%s.blockD) eqc
    simpa only [colsA.1,colsB.1,colsA.2,colsB.2] using And.intro hd hm
  cases a;cases b;simp_all

theorem target_injective (s : Shape) (a b : Work) (ha : a.active s) (hb : b.active s)
    (he : a.target s=b.target s) : a=b := by
  have da:=decode s.dim ((a.batch*s.heads+a.head)*s.length+a.row) a.col ha.2.2.2
  have db:=decode s.dim ((b.batch*s.heads+b.head)*s.length+b.row) b.col hb.2.2.2
  have e1:=congrArg (fun x=>x/s.dim) he
  have ec:=congrArg (fun x=>x%s.dim) he
  simp only [Work.target,outputAddress,da.1,db.1,da.2,db.2] at e1 ec
  have na:=decode s.length (a.batch*s.heads+a.head) a.row ha.2.2.1
  have nb:=decode s.length (b.batch*s.heads+b.head) b.row hb.2.2.1
  have e2:=congrArg (fun x=>x/s.length) e1
  have er:=congrArg (fun x=>x%s.length) e1
  simp only [na.1,nb.1,na.2,nb.2] at e2 er
  have hda:=decode s.heads a.batch a.head ha.2.1
  have hdb:=decode s.heads b.batch b.head hb.2.1
  have eb:=congrArg (fun x=>x/s.heads) e2
  have eh:=congrArg (fun x=>x%s.heads) e2
  simp only [hda.1,hdb.1,hda.2,hdb.2] at eb eh
  cases a;cases b;simp_all

theorem target_bound (s : Shape) (w : Work) (active : w.active s) :
    w.target s<s.batches*s.heads*s.length*s.dim := by
  rcases active with ⟨hb,hh,hn,hd⟩
  have hp:=Nat.mul_le_mul_right s.heads (show w.batch+1≤s.batches by omega)
  have bh : w.batch*s.heads+w.head<s.batches*s.heads := by nlinarith
  have np:=Nat.mul_le_mul_right s.length (show w.batch*s.heads+w.head+1≤s.batches*s.heads by omega)
  have bn : (w.batch*s.heads+w.head)*s.length+w.row<s.batches*s.heads*s.length := by nlinarith
  have dp:=Nat.mul_le_mul_right s.dim (show (w.batch*s.heads+w.head)*s.length+w.row+1≤s.batches*s.heads*s.length by omega)
  unfold Work.target outputAddress
  nlinarith

theorem csr_visit (s : Shape) (c : CSR) (h row visit : Nat) :
    visit<c.count s h row ↔ c.first s h row+(visit:Int)<c.last s h row := by
  unfold CSR.count;omega
theorem csr_stop (s : Shape) (c : CSR) (h row : Nat) :
    c.last s h row≤c.first s h row+(c.count s h row:Int) := by
  unfold CSR.count;omega

#axiomsClean channel_coverage
#axiomsClean lane_coverage
#axiomsClean lane_injective
#axiomsClean target_injective
#axiomsClean target_bound
#axiomsClean csr_visit
#axiomsClean csr_stop
end VeriTile.Bench.Repairs.BlockSparseAttn
