import bench.tritonbench_g.mixed_sparse_attention.Proofs.HostCoverage
import bench.tritonbench_g.mixed_sparse_attention.Proofs.AddressArithmetic
namespace VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Whole
open VeriTile Triton FP.Structural Source
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

structure Tensor where
  region : RegionName
  storage : ComputeDType
  shape : List Nat

def returnedTensor (c : Config) : Tensor := ⟨c.out,.fp32,[c.Z,c.H,c.T,c.BD]⟩
def tensorSize (c : Config) : Nat := c.Z*c.H*c.T*c.BD

/-- Ordinary zeros_like allocation: the fresh output's logical cells are zero;
all other cells are preserved. The scalar zero word is supplied by the dtype ABI. -/
def zeroOutput {α : Type} (c : Config) (zero : α) (s : State α) : State α :=
  {s with mem:=fun r off=>if r=c.out ∧ off<tensorSize c then
      Cell.mk .real (ofFloat .real zero)
    else s.mem r off}

structure Result (α : Type) where
  result : Tensor
  memory : State α

variable {α : Type} [Inhabited α]

def invoke (M : Program→Algebra α) (D : Program→Dot α) (c : Config) (zero : α)
    (s : State α) (work : List Program) : Option (Result α) :=
  let initial:=zeroOutput c zero s
  (if c.T=0 then some initial else runPrograms M D c (grid c) work initial).map
    (fun t=>⟨returnedTensor c,t⟩)

theorem zeroOutput_reads (c : Config) (zero : α) (s : State α) : ReadsEq c (zeroOutput c zero s) s := by
  intro r hr
  funext off
  simp [zeroOutput,hr]

theorem zeroOutput_lengths (c : Config) (zero : α) (s : State α) (sep : Separated c)
    (hl : LengthsBound c s) : LengthsBound c (zeroOutput c zero s) := by
  intro b hb
  simpa [zeroOutput,sep.inputs (Region.cast c.seq) (by simp)] using hl b hb

theorem programs_empty (c : Config) (hbm : 0<c.BM) (hz : c.T=0) : programs (grid c)=[] := by
  have hx : (c.T+c.BM-1)/c.BM=0 := Nat.div_eq_of_lt (by omega)
  simp [programs,grid,hx]

def Exterior (c : Config) (r : RegionName) (off : Nat) : Prop := r≠c.out ∨ tensorSize c≤off

def LogicalPost (M : Program→Algebra α) (D : Program→Dot α) (c : Config) (zero : α)
    (s t : State α) (hbm : 0<c.BM) : Prop :=
  (∀ hz<c.Z*c.H,∀ n<c.T,∀ d (hd : d<c.BD),
    t.mem c.out (logicalAddress c hz n d)=
      .mk .real (ofFloat .real
        (if n<(s.mem (Region.cast c.seq) (hz/c.H)).read .nat then
          answer (M (n/c.BM,hz,0)) (D (n/c.BM,hz,0)) c
            (start (grid c) (n/c.BM,hz,0) s) (lane c hbm n d hd)
         else zero))) ∧
  (∀ r off,Exterior c r off → t.mem r off=s.mem r off)

/-- Actual current-memory grid execution, complete logical output including
zero sequence tails, returned tensor descriptor, and exterior frame. -/
theorem returned_source (M : Program→Algebra α) (D : Program→Dot α) (c : Config)
    (zero : α) (s : State α) (sep : Separated c) (layout : Contiguous c)
    (hl : LengthsBound c s) (hbm : 0<c.BM) (hbn : 0<c.BN)
    (work : List Program) (legal : (programs (grid c)).Perm work) :
    ∃ result,invoke M D c zero s work=some result ∧ result.result=returnedTensor c ∧
      LogicalPost M D c zero s result.memory hbm := by
  let initial:=zeroOutput c zero s
  have hli : LengthsBound c initial:=zeroOutput_lengths c zero s sep hl
  have uniq : Unique c (grid c) work initial := by
    intro a ha b hb i j hi hj he
    exact grid_unique c layout initial hli a (legal.mem_iff.mpr ha) b (legal.mem_iff.mpr hb) i j hi hj he
  obtain ⟨final,hf,hv,hframe⟩:=run_correct M D c sep hbn (grid c) work initial uniq
  refine ⟨⟨returnedTensor c,final⟩,?_,rfl,?_,?_⟩
  · by_cases hz : c.T=0
    · have hw : work=[] := by
        have he:=programs_empty c hbm hz
        have hn:=legal.length_eq
        rw [he] at hn
        cases work <;> simp_all
      simp only [hw,runPrograms,Option.some.injEq] at hf
      subst final
      simp [invoke,hz,initial]
    · simp [invoke,hz,initial,hf]
  · intro hz hh n hn d hd
    let p : Program:=(n/c.BM,hz,0)
    let i:=lane c hbm n d hd
    obtain ⟨hp,hrow⟩:=grid_covers c hbm hz n d hh hn hd
    have haddr : address c (start (grid c) p initial) i=logicalAddress c hz n d := by
      rw [source_address c layout]
      exact congrArg (fun row=>logicalAddress c hz row d) hrow
    have refs:=references_eq (M p) (D p) c sep (start (grid c) p initial) (start (grid c) p s)
      rfl (zeroOutput_reads c zero s)
    by_cases hactive : n<(s.mem (Region.cast c.seq) (hz/c.H)).read .nat
    · have hi : active c (start (grid c) p initial) i := by
        rw [refs.1]
        change (n/c.BM)*c.BM+n%c.BM<_
        rw [hrow]
        exact hactive
      have hw:=hv p (legal.mem_iff.mp hp) i hi
      rw [haddr,refs.2.2] at hw
      simpa [hactive,p,i] using hw
    · have outside : Outside c (grid c) work initial c.out (logicalAddress c hz n d) := by
        right
        intro q hq j hj he
        have hq':=legal.mem_iff.mpr hq
        have hnj:=active_row_bound c initial hli q hq' j hj
        rw [source_address c layout] at he
        obtain ⟨ha,_⟩:=block_unique c.BD _ _ _ _ hd j.2.1.isLt he
        obtain ⟨hy,hr⟩:=block_unique c.T _ _ _ _ hn hnj ha
        have hj' : q.1*c.BM+j.1.val<(s.mem (Region.cast c.seq) (q.2.1/c.H)).read .nat := by
          have href:=references_eq (M q) (D q) c sep (start (grid c) q initial) (start (grid c) q s)
            rfl (zeroOutput_reads c zero s)
          rw [href.1] at hj
          exact hj
        rw [←hy,←hr] at hj'
        exact hactive hj'
      rw [hframe _ _ outside]
      have hb : logicalAddress c hz n d<tensorSize c := dense_bound (c.Z*c.H) c.T c.BD hz n d hh hn hd
      simp [initial,zeroOutput,hb,hactive]
  · intro r off hext
    have outside : Outside c (grid c) work initial r off := by
      rcases hext with hr|ho
      · exact Or.inl hr
      · right
        intro q hq i hi he
        have hp:=legal.mem_iff.mpr hq
        obtain ⟨_,hh,_⟩:=(mem_programs (grid c) q.1 q.2.1 q.2.2).mp hp
        have hn:=active_row_bound c initial hli q hp i hi
        have hb:=dense_bound (c.Z*c.H) c.T c.BD q.2.1 (q.1*c.BM+i.1.val) i.2.1.val hh hn i.2.1.isLt
        rw [source_address c layout] at he
        change tensorSize c≤off at ho
        change logicalAddress c q.2.1 (q.1*c.BM+i.1.val) i.2.1.val<tensorSize c at hb
        omega
    rw [hframe _ _ outside]
    rcases hext with hr|ho
    · simp [initial,zeroOutput,hr]
    · simp [initial,zeroOutput,not_lt.mpr ho]

#axiomsClean returned_source
#specNonCircular LogicalPost avoiding [invoke,runPrograms,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Whole
