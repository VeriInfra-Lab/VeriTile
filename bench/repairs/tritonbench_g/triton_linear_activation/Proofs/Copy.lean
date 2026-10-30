import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.EventGrid
namespace VeriTile.Bench.Repairs.TritonLinearActivation
noncomputable section
set_option maxHeartbeats 4000000

def copyStep (a : Arithmetic α) (fromMode toMode : Mode) (src dst i : Nat) (m : Memory α) : Memory α :=
  writeWord m (dst,i) (a.encode toMode (a.decode fromMode (m (src,i))))
def copyRun (a : Arithmetic α) (fromMode toMode : Mode) (src dst count : Nat) (m : Memory α) : Memory α :=
  Schedule.run (copyStep a fromMode toMode src dst) (List.range count) m

def copyWrites (dst i : Nat) (p : Address) : Prop := p=(dst,i)
instance (dst i : Nat) : DecidablePred (copyWrites dst i) := by intro p; unfold copyWrites; infer_instance

theorem copy_frozen (a : Arithmetic α) (fromMode toMode : Mode) (src dst count : Nat)
    (ne : src≠dst) (entry : Memory α) :
    copyRun a fromMode toMode src dst count entry=
      Schedule.run (Schedule.put (copyWrites dst)
        (fun i _ => a.encode toMode (a.decode fromMode (entry (src,i))))) (List.range count) entry := by
  apply Schedule.freeze_reads (reads:=fun _ p => p.1=src) (writes:=copyWrites dst) (entry:=entry)
  · intro i current same
    simp only [copyStep,same (src,i) rfl]
    rfl
  · intro i current p outside
    exact if_neg outside
  · exact List.nodup_range
  · intro i hi j hj hn p read write
    apply ne
    exact read.symm.trans (congrArg Prod.fst write)
  · intro i hi p hp; rfl

theorem copy_output (a : Arithmetic α) (fromMode toMode : Mode) (src dst count : Nat)
    (ne : src≠dst) (entry : Memory α) (i : Nat) (hi : i<count) :
    copyRun a fromMode toMode src dst count entry (dst,i)=
      a.encode toMode (a.decode fromMode (entry (src,i))) := by
  rw [copy_frozen a fromMode toMode src dst count ne entry,
    Schedule.fixed_output (copyWrites dst) (fun i _ => a.encode toMode (a.decode fromMode (entry (src,i))))
      (List.range count) entry i (dst,i) rfl]
  · simp [hi]
  · intro j hj hw
    exact (congrArg Prod.snd hw).symm

theorem copy_frame (a : Arithmetic α) (fromMode toMode : Mode) (src dst count : Nat)
    (ne : src≠dst) (entry : Memory α) (p : Address) (outside : p.1≠dst ∨ count≤p.2) :
    copyRun a fromMode toMode src dst count entry p=entry p := by
  rw [copy_frozen a fromMode toMode src dst count ne entry]
  apply Schedule.fixed_frame
  intro i hi hw
  have h1 := congrArg Prod.fst hw
  have h2 := congrArg Prod.snd hw
  simp only [List.mem_range] at hi
  rcases outside with h|h
  · exact h h1
  · simp only [Prod.snd] at h2; omega

def copyCellCode (fromMode toMode : Mode) (src dst i : Nat) : Code α Unit := do
  let x ← emit (.read fromMode (src,i))
  emit (.store toMode (dst,i) x)
def copyCode (fromMode toMode : Mode) (src dst count : Nat) : Code α Unit :=
  sequenceCode (copyCellCode fromMode toMode src dst) (List.range count)

theorem eval_copyCode (a : Arithmetic α) (fromMode toMode : Mode) (src dst count : Nat) (m : Memory α) :
    (copyCode fromMode toMode src dst count).eval a m=((),copyRun a fromMode toMode src dst count m) := by
  apply eval_sequenceCode a _ _ (fun i memory => ?_) _ m
  rfl

#axiomsClean copy_output
#axiomsClean copy_frame
#axiomsClean eval_copyCode
end
end VeriTile.Bench.Repairs.TritonLinearActivation
