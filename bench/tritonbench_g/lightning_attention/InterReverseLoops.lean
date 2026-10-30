import bench.tritonbench_g.lightning_attention.InterForwardLoops
import bench.tritonbench_g.lightning_attention.InterReverseUpdate
import bench.tritonbench_g.lightning_attention.InterReverseDelta
import bench.tritonbench_g.lightning_attention.SignedRange
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]
def reverseRow (p : Parameters) (block sub : Nat) := p.NT*p.BT-(block*p.NC+sub)*p.CB

theorem reverseRow_step (p : Parameters) (hbalance : p.NC*p.CB=p.BT)
    (block sub : Nat) (hb : block<p.NT) (hs : sub<p.NC) :
    reverseRow p block (sub+1)+p.CB=reverseRow p block sub := by
  have hcoef : block*p.NC+(sub+1)≤p.NT*p.NC := by
    calc
      _ ≤ (block+1)*p.NC := by rw [Nat.add_mul]; omega
      _ ≤ p.NT*p.NC := Nat.mul_le_mul_right p.NC (by omega)
  have hsize : (block*p.NC+(sub+1))*p.CB≤p.NT*p.BT := by
    calc
      _ ≤ (p.NT*p.NC)*p.CB := Nat.mul_le_mul_right p.CB hcoef
      _ = _ := by rw [Nat.mul_assoc,hbalance]
  have hshift : (block*p.NC+(sub+1))*p.CB=(block*p.NC+sub)*p.CB+p.CB := by ring
  unfold reverseRow
  rw [hshift] at hsize ⊢
  omega

theorem reverseRow_end (p : Parameters) (block : Nat) :
    reverseRow p block p.NC=reverseRow p (block+1) 0 := by simp [reverseRow,Nat.add_mul]

theorem keyValueSubFold (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head block : Nat)
    (carry : Matrix α p.D p.E) (m : Memory α) :
    foldRange (fun sub mem => keyValueSubMemory M D p head (reverseRow p block (sub+1)) block carry mem)
      0 p.NC 1 m=keyValueBlockMemory M D p head block carry m := by
  by_cases hb : block=0
  · simp [keyValueSubMemory,keyValueBlockMemory,hb,foldRange_id]
  · simp only [keyValueSubMemory,keyValueBlockMemory,hb,if_false,reverseRow,Nat.add_assoc]

theorem interReverseUpdateLoop (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (hbalance : p.NC*p.CB=p.BT) (head block row2 : Nat) (hbnd : block<p.NT)
    (input : Memory α) (carry : Matrix α p.D p.E)
    (s : State α) (ctx : InterRReady p head (reverseRow p block 0) row2 input carry s)
    (hi : s.regs .int [] "i"=some (fun _ => (p.NT : Int)-1-block)) :
    ∃ t, FP.Structural.rangeInt (engine (withDot M D)) "__rev_j" (1-(p.NC : Int)) 1 1 (interRUpdateParts p).body s=some t ∧
      InterRReady p head (reverseRow p (block+1) 0) row2 input carry t ∧
      t.mem=keyValueBlockMemory M D p head block carry s.mem := by
  let update := fun sub mem => keyValueSubMemory M D p head (reverseRow p block (sub+1)) block carry mem
  let I := fun n (t : State α) =>
    InterRReady p head (reverseRow p block n) row2 input carry t ∧
    t.regs .int [] "i"=some (fun _ => (p.NT : Int)-1-block) ∧ t.mem=foldRange update 0 n 1 s.mem
  have hs : ∀ n, n<p.NC → ∀ t, I n t → ∃ u,
      run (engine (withDot M D)) (interRUpdateParts p).body
        (t.setReg "__rev_j" .int [] (fun _ => (n : Int)+1-p.NC))=some u ∧ I (n+1) u := by
    intro n hn t ht
    have ctx' : InterRReady p head (reverseRow p block (n+1)+p.CB) row2 input carry t := by
      rw [reverseRow_step p hbalance block n hbnd hn]; exact ht.1
    let run := interReverseUpdate M D p head (reverseRow p block (n+1)) row2 block n input carry t ctx' ht.2.1 hbnd
    have next := interReverseUpdate_ready M D p head (reverseRow p block (n+1)) row2 block n input carry t ctx' ht.2.1 hbnd
    refine ⟨run.val,run.property,next.1,next.2,?_⟩
    change (interReverseUpdate M D p head _ _ _ _ _ _ _ _ _ _).val.mem=_
    rw [interReverseUpdate_memory,ht.2.2]
    exact (foldRange_succ update n s.mem).symm
  have initial : I 0 s := ⟨ctx,hi,by simp [foldRange_zero]⟩
  obtain ⟨t,he,hf⟩ := signed_range_invariant (engine (withDot M D)) "__rev_j" (interRUpdateParts p).body p.NC I hs 0 (Nat.zero_le _) s initial
  exact ⟨t,by simpa using he,by simpa only [reverseRow_end] using hf.1,
    hf.2.2.trans (keyValueSubFold M D p head block carry s.mem)⟩

theorem interReverseDeltaLoop (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (hbalance : p.NC*p.CB=p.BT) (head row1 block : Nat) (hbnd : block<p.NT)
    (input : Memory α) (carry : Matrix α p.D p.E)
    (s : State α) (ctx : InterRReady p head row1 (reverseRow p block 0) input carry s)
    (ha : InterRDelta p (fun _ => zero M) s) :
    ∃ t, FP.Structural.rangeInt (engine (withDot M D)) "__rev_j" (1-(p.NC : Int)) 1 1 (interRDeltaParts p).body s=some t ∧
      InterRReady p head row1 (reverseRow p (block+1) 0) input carry t ∧
      InterRDelta p (keyValueBlockDelta M D p head block input) t ∧ t.mem=s.mem := by
  let update := fun sub acc => keyValueSubDelta M D p head (reverseRow p block (sub+1)) input acc
  let I := fun n (t : State α) =>
    InterRReady p head row1 (reverseRow p block n) input carry t ∧
    InterRDelta p (foldRange update 0 n 1 (fun _ => zero M)) t ∧ t.mem=s.mem
  have hs : ∀ n, n<p.NC → ∀ t, I n t → ∃ u,
      run (engine (withDot M D)) (interRDeltaParts p).body
        (t.setReg "__rev_j" .int [] (fun _ => (n : Int)+1-p.NC))=some u ∧ I (n+1) u := by
    intro n hn t ht
    have ctx' : InterRReady p head row1 (reverseRow p block (n+1)+p.CB) input carry t := by
      rw [reverseRow_step p hbalance block n hbnd hn]; exact ht.1
    let acc := foldRange update 0 n 1 (fun _ => zero M)
    let run := interReverseDelta M D p head row1 (reverseRow p block (n+1)) n input carry acc t ctx' ht.2.1
    have next := interReverseDelta_ready M D p head row1 (reverseRow p block (n+1)) n input carry acc t ctx' ht.2.1
    refine ⟨run.val,run.property,next.1,?_,?_⟩
    · rw [foldRange_succ]; exact next.2
    · exact (interReverseDelta_memory M D p head row1 _ n input carry acc t ctx' ht.2.1).trans ht.2.2
  have initial : I 0 s := ⟨ctx,by simpa [foldRange_zero] using ha,rfl⟩
  obtain ⟨t,he,hf⟩ := signed_range_invariant (engine (withDot M D)) "__rev_j" (interRDeltaParts p).body p.NC I hs 0 (Nat.zero_le _) s initial
  refine ⟨t,by simpa using he,by simpa only [reverseRow_end] using hf.1,?_,hf.2.2⟩
  simpa only [update,keyValueSubDelta,reverseRow,keyValueBlockDelta,Nat.add_assoc] using hf.2.1
#axiomsClean reverseRow_step
#axiomsClean interReverseUpdateLoop
#axiomsClean interReverseDeltaLoop
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly
