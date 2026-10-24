import bench.tritonbench_g.matmul_persistent_triton.Proofs.NativeBits

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.NativePrefix
open NativeDomain NativeBits
set_option maxHeartbeats 4000000

/-- The cdiv lowering present in both recorded TTIR files: i32 add of the
constant block-1 followed by signed i32 division. -/
def ceilBits (length block : Nat) :=
  (bits (length : Int) + bits ((block-1 : Nat) : Int)).sdiv (bits (block : Int))

theorem ceil_exact (length block : Nat) (hb : 0 < block) (hblock : block < limit)
    (hadd : length+block-1 < limit) :
    Exact (ceilBits length block) ((RealMath.blocks length block : Nat) : Int) := by
  have he : length+(block-1) = length+block-1 := by omega
  have hn : length < limit := by omega
  have hm : block-1 < limit := by omega
  have ha := nat_add_exact (nat_exact hn) (nat_exact hm) (by omega)
  have hd := div_exact ha (nat_exact hblock) (by omega)
  simpa only [ceilBits,RealMath.blocks,he] using hd

structure Values where
  nr : BitVec 32
  nc : BitVec 32
  nk : BitVec 32
  total : BitVec 32
  quotient : BitVec 32
  remainder : BitVec 32
  work : BitVec 32
  tile : BitVec 32
  ki : BitVec 32
  groupSpan : BitVec 32
  trip : BitVec 32

/-- Actual prefix integer operations, preserving both the conditional work
increment and signed negative initialization. This is an expression model,
not an invented native execution relation or a proof of compiler correctness. -/
def eval (p : Parameters) (pid : Nat) : Values :=
  let nr := ceilBits p.M p.BM
  let nc := ceilBits p.N p.BN
  let nk := ceilBits p.K p.BK
  let total := nr * nc
  let quotient := total.sdiv (bits (p.SMS : Int))
  let remainder := total.srem (bits (p.SMS : Int))
  let work := if (bits (pid : Int)).slt remainder then quotient + bits 1 else quotient
  ⟨nr,nc,nk,total,quotient,remainder,work,bits (pid : Int)-bits (p.SMS : Int),
    bits (-1),nc*bits (p.GM : Int),nk*work⟩

structure Agrees (p : Parameters) (pid : Nat) (v : Values) : Prop where
  nr : Exact v.nr (rows p)
  nc : Exact v.nc (columns p)
  nk : Exact v.nk (count p)
  total : Exact v.total (NativeDomain.total p)
  quotient : Exact v.quotient (NativeDomain.total p/p.SMS)
  remainder : Exact v.remainder (NativeDomain.total p%p.SMS)
  work : Exact v.work (NativeDomain.work p pid)
  tile : Exact v.tile ((pid : Int)-(p.SMS : Int))
  ki : Exact v.ki (-1)
  groupSpan : Exact v.groupSpan (p.GM*columns p)
  trip : Exact v.trip (NativeDomain.trip p pid)

theorem agrees (p : Parameters) (h : Admissible p) (pid : Nat) (hp : pid < grid p) :
    Agrees p pid (eval p pid) := by
  have hs : p.SMS < limit := h.scalar _ (by simp)
  have hp' : pid < p.SMS := lt_of_lt_of_le hp (Nat.min_le_left _ _)
  have hpbit := nat_exact (lt_trans hp' hs)
  have hsbit := nat_exact hs
  have hnr := ceil_exact p.M p.BM h.bm (h.scalar _ (by simp)) h.ceil_m
  have hnc := ceil_exact p.N p.BN h.bn (h.scalar _ (by simp)) h.ceil_n
  have hnk := ceil_exact p.K p.BK h.bk (h.scalar _ (by simp)) h.ceil_k
  have htotal := nat_mul_exact hnr hnc h.total_bound
  have hquot := div_exact htotal hsbit h.total_bound
  have hrem := rem_exact htotal hsbit
  have hcond : (bits (pid : Int)).slt
      ((ceilBits p.M p.BM * ceilBits p.N p.BN).srem (bits (p.SMS : Int))) =
      decide (pid < NativeDomain.total p % p.SMS) := by
    simpa only [Int.ofNat_lt] using slt_exact hpbit hrem
  have hwork : Exact
      (if (bits (pid : Int)).slt ((ceilBits p.M p.BM * ceilBits p.N p.BN).srem (bits (p.SMS : Int)))
        then (ceilBits p.M p.BM * ceilBits p.N p.BN).sdiv (bits (p.SMS : Int)) + bits 1
        else (ceilBits p.M p.BM * ceilBits p.N p.BN).sdiv (bits (p.SMS : Int)))
      (NativeDomain.work p pid) := by
    rw [hcond]
    by_cases hc : pid < NativeDomain.total p % p.SMS
    · have hb : NativeDomain.total p / p.SMS + 1 < limit := by
        have hw := work_lt p h pid hp'
        simpa only [NativeDomain.work,Schedule.tilesFor,if_pos hc] using hw
      have hw := nat_add_exact hquot (nat_exact (n := 1) (by decide)) hb
      simpa only [hc,decide_true,if_true,NativeDomain.work,Schedule.tilesFor,if_pos hc] using hw
    · simpa only [hc,decide_false,Bool.false_eq_true,if_false,NativeDomain.work,Schedule.tilesFor,if_neg hc,Nat.add_zero] using hquot
  have ht : Fits ((pid : Int)-(p.SMS : Int)) := by unfold Fits limit at *; omega
  refine ⟨hnr,hnc,hnk,htotal,hquot,hrem,hwork,sub_exact hpbit hsbit ht,
    bits_exact (by norm_num [Fits]),?_,?_⟩
  · have hg := nat_mul_exact hnc (nat_exact (h.scalar p.GM (by simp)))
      (show columns p*p.GM < limit by simpa only [Nat.mul_comm] using h.group_span)
    simpa only [Nat.mul_comm] using hg
  · exact nat_mul_exact hnk hwork (trip_lt p h pid)

/-- The negative initial tile is kept as signed i32; the first native add
restores exactly the launched PID without natural-subtraction truncation. -/
theorem initial_tile_step (p : Parameters) (h : Admissible p) (pid : Nat)
    (hp : pid < grid p) :
    Exact ((eval p pid).tile + bits (p.SMS : Int)) (pid : Int) := by
  have ht := (agrees p h pid hp).tile
  have hs := nat_exact (h.scalar p.SMS (by simp))
  have hpid : pid < limit := lt_trans (lt_of_lt_of_le hp (Nat.min_le_left _ _))
    (h.scalar p.SMS (by simp))
  simpa only [sub_add_cancel] using add_exact ht hs (by simpa using nat_fits hpid)

/-- The native scf.for i32 induction increment fits even at its final step. -/
theorem loop_induction_step (p : Parameters) (h : Admissible p) (pid i : Nat)
    (hi : i < trip p pid) :
    Exact (bits (i : Int)+bits 1) ((i+1 : Nat) : Int) := by
  have hbound := trip_lt p h pid
  exact nat_add_exact (nat_exact (by omega)) (nat_exact (n := 1) (by decide)) (by omega)

end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.NativePrefix
