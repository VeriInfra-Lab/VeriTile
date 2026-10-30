import bench.tritonbench_g.block_sparse_attn.Compatibility

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def layoutOf (p : Parameters) (batches : Nat) (rows cols : Int → Int) : BlockSparseAttn.Layout where
  shape := ⟨batches,p.heads,p.kvHeads,p.length,16,16,16,p.blocks,p.layouts⟩
  q := ⟨p.qb,p.qh,p.qm,p.qd⟩
  k := ⟨p.kb,p.kh,p.kn,p.kd⟩
  v := ⟨p.vb,p.vh,p.vn,p.vd⟩
  csr := ⟨p.rowStride,p.colStride,rows,cols⟩
  evenM := p.evenM
  evenN := p.evenN
  reduction := BlockSparseAttn.SM90B16.plan
  scoreInstruction := .mma16n8k8
  valueInstruction := .mma16n8k8

/-- Only initial typed memory readers are related to logical views. There are
no arithmetic, loop, score, update, execution, or result fields. -/
structure Views {α : Type} [Inhabited α] (s : State (Replicas α))
    (m : BlockSparseAttn.Memory α) (rows cols : Int → Int) : Prop where
  query : ∀ address, (s.mem "Q" address).read .real=broadcast (m .q address)
  key : ∀ address, (s.mem "K" address).read .real=broadcast (m .k address)
  value : ∀ address, (s.mem "V" address).read .real=broadcast (m .v address)
  rows : ∀ address, (s.mem "rows" address).read .int=rows address
  cols : ∀ address, (s.mem "cols" address).read .int=cols address

def work {α : Type} (p : Parameters) (s : State α) (i : TileIndex [16,16]) : Work :=
  ⟨batch p s,head p s,row s i.1,channel s i.2.1⟩

theorem query_view {α : Type} [Inhabited α] (a : BlockSparseAttn.Ops α) (literal argument : ℝ → α)
    (zero : literal 0=a.zero) (p : Parameters) (batches : Nat)
    (s : State (Replicas α)) (m : BlockSparseAttn.Memory α) (rows cols : Int → Int)
    (views : Views s m rows cols) (block : Nat) (i : TileIndex [16,16]) :
    query (Arithmetic.ofLegacy a literal argument) p s block i =
      broadcast (qLoad (layoutOf p batches rows cols) a m (batch p s) (head p s)
        (row s i.1) (block*16+i.2.1.val)) := by
  simp [query,qLoad,layoutOf,Arithmetic.ofLegacy,views.query,zero,qPointer,address,
    Nat.add_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
  split_ifs <;> rfl

theorem first_view {α : Type} [Inhabited α] (p : Parameters) (batches : Nat)
    (s : State (Replicas α)) (m : BlockSparseAttn.Memory α) (rows cols : Int → Int)
    (views : Views s m rows cols) (i : Fin 16) :
    first p s=(layoutOf p batches rows cols).csr.first (layoutOf p batches rows cols).shape
      (head p s) (row s i) := by
  have row_div : row s i/16=s.pids 0 := by unfold row; omega
  simp [first,rowPointer,layout,layoutOf,CSR.first,views.rows,row_div]

theorem last_view {α : Type} [Inhabited α] (p : Parameters) (batches : Nat)
    (s : State (Replicas α)) (m : BlockSparseAttn.Memory α) (rows cols : Int → Int)
    (views : Views s m rows cols) (i : Fin 16) :
    last p s=(layoutOf p batches rows cols).csr.last (layoutOf p batches rows cols).shape
      (head p s) (row s i) := by
  have row_div : row s i/16=s.pids 0 := by unfold row; omega
  simp [last,rowPointer,layout,layoutOf,CSR.last,views.rows,row_div]

theorem count_view {α : Type} [Inhabited α] (p : Parameters) (batches : Nat)
    (s : State (Replicas α)) (m : BlockSparseAttn.Memory α) (rows cols : Int → Int)
    (views : Views s m rows cols) (i : Fin 16) :
    visitCount p s=(layoutOf p batches rows cols).csr.count (layoutOf p batches rows cols).shape
      (head p s) (row s i) := by
  simp only [visitCount,CSR.count,first_view p batches s m rows cols views i,
    last_view p batches s m rows cols views i]

theorem key_view {α : Type} [Inhabited α] (a : BlockSparseAttn.Ops α) (literal argument : ℝ → α)
    (zero : literal 0=a.zero) (p : Parameters) (batches : Nat)
    (s : State (Replicas α)) (m : BlockSparseAttn.Memory α) (rows cols : Int → Int)
    (views : Views s m rows cols) (column : Int) (nonnegative : 0≤column)
    (block : Nat) (i : TileIndex [16,16]) :
    key (Arithmetic.ofLegacy a literal argument) p s column block i =
      broadcast (kLoad (layoutOf p batches rows cols) a m (batch p s) (head p s)
        (column*16+i.2.1.val) (block*16+i.1.val)) := by
  obtain ⟨c,hc⟩ : ∃ c : Nat,column=(c : Int) :=
    ⟨column.toNat,(Int.toNat_of_nonneg nonnegative).symm⟩
  subst column
  have offset : ((c:Int)*16*(p.kn:Int)).toNat=c*16*p.kn := by norm_cast
  have key_cast : (c:Int)*16+(i.2.1.val:Int)=((c*16+i.2.1.val:Nat):Int) := by push_cast; rfl
  have addr (b h r d : Nat) : (signedAddress (layoutOf p batches rows cols).k b h (r:Int) d).toNat=
      address (layoutOf p batches rows cols).k b h r d := by
    simp only [signedAddress,address,← Int.natCast_mul,← Int.natCast_add,Int.toNat_natCast]
  have pointer : kPointer p s i+((c:Int)*16*p.kn).toNat+block*16*p.kd=
      (signedAddress (layoutOf p batches rows cols).k (batch p s)
        (kvHead (layoutOf p batches rows cols).shape (head p s))
        ((c:Int)*16+i.2.1.val) (block*16+i.1.val)).toNat := by
    rw [key_cast,addr,offset]
    simp [kPointer,address,layoutOf,kv,kvHead,Nat.add_mul,Nat.mul_add,Nat.mul_assoc,
      Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
  change (if p.evenN || decide ((i.2.1.val:Int)+(c:Int)*16<(p.length:Int)) then
      (s.mem "K" _).read .real else broadcast (literal 0)) = broadcast
    (if p.evenN || decide ((c:Int)*16+(i.2.1.val:Int)<(p.length:Int)) then m .k _ else a.zero)
  rw [pointer,views.key,zero,Int.add_comm (i.2.1.val:Int)]
  split_ifs <;> rfl

theorem value_view {α : Type} [Inhabited α] (a : BlockSparseAttn.Ops α) (literal argument : ℝ → α)
    (zero : literal 0=a.zero) (p : Parameters) (batches : Nat)
    (s : State (Replicas α)) (m : BlockSparseAttn.Memory α) (rows cols : Int → Int)
    (views : Views s m rows cols) (column : Int) (nonnegative : 0≤column)
    (i : TileIndex [16,16]) :
    value (Arithmetic.ofLegacy a literal argument) p s column i =
      broadcast (vLoad (layoutOf p batches rows cols) a m (batch p s) (head p s)
        (column*16+i.1.val) (channel s i.2.1)) := by
  obtain ⟨c,hc⟩ : ∃ c : Nat,column=(c : Int) :=
    ⟨column.toNat,(Int.toNat_of_nonneg nonnegative).symm⟩
  subst column
  have offset : ((c:Int)*16*(p.vn:Int)).toNat=c*16*p.vn := by norm_cast
  have key_cast : (c:Int)*16+(i.1.val:Int)=((c*16+i.1.val:Nat):Int) := by push_cast; rfl
  have addr (b h r d : Nat) : (signedAddress (layoutOf p batches rows cols).v b h (r:Int) d).toNat=
      address (layoutOf p batches rows cols).v b h r d := by
    simp only [signedAddress,address,← Int.natCast_mul,← Int.natCast_add,Int.toNat_natCast]
  have pointer : vPointer p s i+((c:Int)*16*p.vn).toNat=
      (signedAddress (layoutOf p batches rows cols).v (batch p s)
        (kvHead (layoutOf p batches rows cols).shape (head p s))
        ((c:Int)*16+i.1.val) (channel s i.2.1)).toNat := by
    rw [key_cast,addr,offset]
    simp [vPointer,address,layoutOf,kv,kvHead,Nat.add_mul,Nat.mul_add,Nat.mul_assoc,
      Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
  change (if p.evenN || decide ((i.1.val:Int)+(c:Int)*16<(p.length:Int)) then
      (s.mem "V" _).read .real else broadcast (literal 0)) = broadcast
    (if p.evenN || decide ((c:Int)*16+(i.1.val:Int)<(p.length:Int)) then m .v _ else a.zero)
  rw [pointer,views.value,zero,Int.add_comm (i.1.val:Int)]
  split_ifs <;> rfl

theorem column_view {α : Type} [Inhabited α] (p : Parameters) (batches : Nat)
    (s : State (Replicas α)) (m : BlockSparseAttn.Memory α) (rows cols : Int → Int)
    (views : Views s m rows cols) (nonnegative : 0≤first p s) (i : Fin 16) (visit : Nat) :
    csrColumn p s (first p s+visit) =
      (layoutOf p batches rows cols).csr.cols
        ((layoutOf p batches rows cols).csr.slot (layoutOf p batches rows cols).shape
          (head p s) (row s i) visit) := by
  have offset : 0≤((layout p s*p.colStride : Nat) : Int)+(first p s+visit) :=
    add_nonneg (Int.natCast_nonneg _) (add_nonneg nonnegative (Int.natCast_nonneg _))
  simp only [csrColumn,views.cols,Int.toNat_of_nonneg offset]
  rw [first_view p batches s m rows cols views i]
  simp [CSR.slot,layoutOf,layout,Int.add_assoc]

#axiomsClean query_view
#axiomsClean first_view
#axiomsClean last_view
#axiomsClean count_view
#axiomsClean key_view
#axiomsClean value_view
#axiomsClean column_view
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
