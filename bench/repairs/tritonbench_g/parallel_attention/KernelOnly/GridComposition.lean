import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.Grid

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
set_option maxHeartbeats 3000000

/-- Composition of proven program results. The three obligations are local
readback, input separation and unique ownership; none concerns a whole run. -/
theorem runPrograms_correct
    (launch : Program shape L → State α → Option (State α))
    (writes : Program shape L → RegionName → Nat → Prop)
    (target : Program shape L → State α → RegionName → Nat → Cell α)
    (inputSeparate : ∀ p (R : RegionName) off,R∈(["q","k","v","do","dz"] : List RegionName) → ¬writes p R off)
    (unique : ∀ p q R off,writes p R off → writes q R off → p=q)
    (stable : ∀ p s t,SameInputs s t → ∀ R off,target p s R off=target p t R off)
    (steps : ∀ p s,∃ final,launch p s=some final ∧
      (∀ R off,writes p R off → final.mem R off=target p s R off) ∧
      (∀ R off,¬writes p R off → final.mem R off=s.mem R off))
    (ps : List (Program shape L)) (s : State α) :
    ∃ final,runPrograms launch ps s=some final ∧ SameInputs s final ∧
      (∀ p∈ps,∀ R off,writes p R off → final.mem R off=target p s R off) ∧
      (∀ R off,(∀ p∈ps,¬writes p R off) → final.mem R off=s.mem R off) := by
  induction ps generalizing s with
  | nil =>
    refine ⟨s,rfl,?_,?_,?_⟩
    · intro R _; rfl
    · simp
    · intros; rfl
  | cons p rest ih =>
    obtain ⟨middle,first,firstReads,firstFrame⟩ := steps p s
    have inputs : SameInputs s middle := by
      intro R hR
      funext off
      exact (firstFrame R off (inputSeparate p R off hR)).symm
    obtain ⟨final,tail,tailInputs,tailReads,tailFrame⟩ := ih middle
    refine ⟨final,by simp only [runPrograms,first,Option.bind_some,tail],?_,?_,?_⟩
    · intro R hR
      exact (inputs R hR).trans (tailInputs R hR)
    · intro q member R off active
      have stableq := stable q s middle inputs R off
      by_cases present : q∈rest
      · exact (tailReads q present R off active).trans stableq.symm
      · have same : q=p := by simpa [present] using member
        subst q
        have unwritten : ∀ q∈rest,¬writes q R off := by
          intro q hq hwrite
          have eq := unique p q R off active hwrite
          exact present (eq.symm ▸ hq)
        rw [tailFrame R off unwritten,firstReads R off active]
    · intro R off outside
      rw [tailFrame R off (by intro q hq; exact outside q (by simp [hq])),
        firstFrame R off (outside p (by simp))]

/-- Owning a contiguous shard address fixes the head, time and feature. -/
theorem query_write_unique (shape : Shape) (p q : Program shape 64) (R : RegionName) (off : Nat)
    (hp : (queryRectangle shape p.tile).writes 64 shape.BK R off)
    (hq : (queryRectangle shape q.tile).writes 64 shape.BK R off) : p=q := by
  rcases hp.2 with ⟨⟨pr,pe,⟨⟩⟩,hp,ep⟩
  rcases hq.2 with ⟨⟨qr,qe,⟨⟩⟩,hq,eq⟩
  have same := ep.symm.trans eq
  simp only [queryRectangle,Program.tile,Rectangle.active,Nat.zero_mul,Nat.zero_add] at hp hq
  have address : dense shape.T shape.K (p.head.val+shape.B*shape.H*p.value.val)
      (p.chunk.val*64+pr.val) pe.val =
    dense shape.T shape.K (q.head.val+shape.B*shape.H*q.value.val)
      (q.chunk.val*64+qr.val) qe.val := by
    simpa only [queryRectangle,Program.tile,Rectangle.address,Nat.zero_mul,Nat.zero_add,
      dense,Nat.add_mul,Nat.mul_add,Nat.mul_assoc,Nat.add_assoc] using same
  obtain ⟨heads,rows,_⟩ := dense_injective _ _ _ _ _ _ _ _ hp.1 hq.1 hp.2 hq.2 address
  have headShards : p.value.val=q.value.val ∧ p.head.val=q.head.val :=
    tile_injective (shape.B*shape.H) p.value.val q.value.val p.head.val q.head.val
      p.head.isLt q.head.isLt (by simpa [Nat.mul_comm,Nat.add_comm] using heads)
  have chunks := tile_injective 64 p.chunk.val q.chunk.val pr.val qr.val pr.isLt qr.isLt rows
  cases p; cases q
  simp only [Program.mk.injEq]
  exact ⟨Fin.ext headShards.2,Fin.ext chunks.1,Fin.ext headShards.1⟩

theorem output_write_unique (shape : Shape) (L : Nat) (p q : Program shape L) (R : RegionName) (off : Nat)
    (hp : (outputRectangle shape p.tile L).writes L shape.BV R off)
    (hq : (outputRectangle shape q.tile L).writes L shape.BV R off) : p=q := by
  rcases hp.2 with ⟨⟨pr,pv,⟨⟩⟩,hp,ep⟩
  rcases hq.2 with ⟨⟨qr,qv,⟨⟩⟩,hq,eq⟩
  have same := ep.symm.trans eq
  have address : dense shape.T shape.V p.head.val (p.chunk.val*L+pr.val) (p.value.val*shape.BV+pv.val) =
      dense shape.T shape.V q.head.val (q.chunk.val*L+qr.val) (q.value.val*shape.BV+qv.val) := by
    simpa only [outputRectangle,Program.tile,Rectangle.address,Nat.mul_zero,Nat.add_zero,
      dense,Nat.add_mul,Nat.mul_add,Nat.mul_assoc,Nat.add_assoc] using same
  obtain ⟨heads,rows,cols⟩ := dense_injective _ _ _ _ _ _ _ _ hp.1 hq.1 hp.2 hq.2 address
  have chunks := tile_injective L p.chunk.val q.chunk.val pr.val qr.val pr.isLt qr.isLt rows
  have values := tile_injective shape.BV p.value.val q.value.val pv.val qv.val pv.isLt qv.isLt cols
  cases p; cases q
  simp only [Program.mk.injEq]
  exact ⟨Fin.ext heads,Fin.ext chunks.1,Fin.ext values.1⟩

theorem normalizer_write_unique (shape : Shape) (p q : Program shape 128) (R : RegionName) (off : Nat)
    (hp : normalizerWrites shape p.tile R off) (hq : normalizerWrites shape q.tile R off) : p=q := by
  rcases hp with ⟨pv,_,⟨pr,prBound,ep⟩⟩
  rcases hq with ⟨qv,_,⟨qr,qrBound,eq⟩⟩
  have same := ep.symm.trans eq
  simp only [normalizerAddress,Program.tile,Nat.mul_zero,Nat.add_zero] at same
  obtain ⟨heads,rows⟩ := tile_injective shape.T p.head.val q.head.val
    (p.chunk.val*128+pr.val) (q.chunk.val*128+qr.val) prBound qrBound (by simpa [Nat.add_assoc] using same)
  have chunks := tile_injective 128 p.chunk.val q.chunk.val pr.val qr.val pr.isLt qr.isLt rows
  have values : p.value.val=q.value.val := pv.trans qv.symm
  cases p; cases q
  simp only [Program.mk.injEq]
  exact ⟨Fin.ext heads,Fin.ext chunks.1,Fin.ext values⟩

def forwardWrites (shape : Shape) (p : Program shape 128) (R : RegionName) (off : Nat) : Prop :=
  (outputRectangle shape p.tile 128).writes 128 shape.BV R off ∨ normalizerWrites shape p.tile R off

def keyValueWrites (shape : Shape) (p : Program shape 64) (R : RegionName) (off : Nat) : Prop :=
  (keyRectangle shape p.tile).writes 64 shape.BK R off ∨ (valueRectangle shape p.tile).writes 64 shape.BV R off

theorem forward_write_unique (shape : Shape) (p q : Program shape 128) (R : RegionName) (off : Nat)
    (hp : forwardWrites shape p R off) (hq : forwardWrites shape q R off) : p=q := by
  rcases hp with hp|hp <;> rcases hq with hq|hq
  · exact output_write_unique shape 128 p q R off hp hq
  · exact False.elim ((by decide : ("o" : RegionName)≠"z") (hp.1.symm.trans hq.2.1))
  · exact False.elim ((by decide : ("z" : RegionName)≠"o") (hp.2.1.symm.trans hq.1))
  · exact normalizer_write_unique shape p q R off hp hq

theorem keyValue_write_unique (shape : Shape) (p q : Program shape 64) (R : RegionName) (off : Nat)
    (hp : keyValueWrites shape p R off) (hq : keyValueWrites shape q R off) : p=q := by
  rcases hp with hp|hp <;> rcases hq with hq|hq
  · exact query_write_unique shape p q "dq" off ⟨rfl,hp.2⟩ ⟨rfl,hq.2⟩
  · exact False.elim ((by decide : ("dk" : RegionName)≠"dv") (hp.1.symm.trans hq.1))
  · exact False.elim ((by decide : ("dv" : RegionName)≠"dk") (hp.1.symm.trans hq.1))
  · exact output_write_unique shape 64 p q "o" off ⟨rfl,hp.2⟩ ⟨rfl,hq.2⟩

#axiomsClean output_write_unique
#axiomsClean normalizer_write_unique
#axiomsClean forward_write_unique
#axiomsClean keyValue_write_unique
#axiomsClean runPrograms_correct
#axiomsClean query_write_unique
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel
