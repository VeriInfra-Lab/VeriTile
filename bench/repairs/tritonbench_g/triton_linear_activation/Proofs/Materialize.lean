import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.Copy
namespace VeriTile.Bench.Repairs.TritonLinearActivation
noncomputable section
set_option maxHeartbeats 4000000

 theorem copy_form (a : Arithmetic α) (fromMode toMode : Mode) (src dst count : Nat)
    (ne : src≠dst) (memory : Memory α) :
    copyRun a fromMode toMode src dst count memory=
      fun p => if p.1=dst ∧ p.2<count then
        a.encode toMode (a.decode fromMode (memory (src,p.2))) else memory p := by
  funext p
  rcases p with ⟨storage,i⟩
  by_cases h : storage=dst ∧ i<count
  · rcases h with ⟨he,hi⟩
    subst storage
    simp only [Prod.fst,Prod.snd,hi,lt_self_iff_false,true_and,and_self,ite_true]
    exact copy_output a fromMode toMode src dst count ne memory i hi
  · have ho : storage≠dst ∨ count ≤ i := by omega
    simp only [Prod.fst,Prod.snd,h,ite_false]
    exact copy_frame a fromMode toMode src dst count ne memory _ ho

structure CopyRegions (n : Nat) where
  src : Fin n → Nat
  dst : Fin n → Nat
  count : Fin n → Nat
  distinct : Function.Injective dst
  separate : ∀ i j,src i≠dst j

def CopyRegions.writes (r : CopyRegions n) (i : Fin n) (p : Address) : Prop :=
  p.1=r.dst i ∧ p.2<r.count i
instance (r : CopyRegions n) (i : Fin n) : DecidablePred (r.writes i) := by
  intro p; unfold CopyRegions.writes; infer_instance

def CopyRegions.step (r : CopyRegions n) (a : Arithmetic α) (fromMode toMode : Mode)
    (i : Fin n) (m : Memory α) : Memory α :=
  copyRun a fromMode toMode (r.src i) (r.dst i) (r.count i) m

def CopyRegions.run (r : CopyRegions n) (a : Arithmetic α) (fromMode toMode : Mode)
    (m : Memory α) : Memory α :=
  Schedule.run (r.step a fromMode toMode) (List.finRange n) m

theorem CopyRegions.frozen (r : CopyRegions n) (a : Arithmetic α) (fromMode toMode : Mode) (entry : Memory α) :
    r.run a fromMode toMode entry=
      Schedule.run (Schedule.put r.writes
        (fun i p => a.encode toMode (a.decode fromMode (entry (r.src i,p.2))))) (List.finRange n) entry := by
  apply Schedule.freeze_reads (reads:=fun _ p => ∃ i,p.1=r.src i) (writes:=r.writes) (entry:=entry)
  · intro i current same
    rw [CopyRegions.step,copy_form a fromMode toMode _ _ _ (r.separate i i)]
    funext p
    simp only [Schedule.put,CopyRegions.writes,same (r.src i,p.2) ⟨i,rfl⟩]
  · intro i current p outside
    rw [CopyRegions.step,copy_form a fromMode toMode _ _ _ (r.separate i i)]
    exact if_neg outside
  · exact List.nodup_finRange n
  · intro i hi j hj hne p read write
    obtain ⟨q,hq⟩ := read
    exact r.separate q i (hq.symm.trans write.1)
  · intro i hi p read; rfl

theorem CopyRegions.output (r : CopyRegions n) (a : Arithmetic α) (fromMode toMode : Mode)
    (entry : Memory α) (i : Fin n) (offset : Nat) (h : offset<r.count i) :
    r.run a fromMode toMode entry (r.dst i,offset)=
      a.encode toMode (a.decode fromMode (entry (r.src i,offset))) := by
  rw [r.frozen a fromMode toMode entry,
    Schedule.fixed_output r.writes
      (fun i p => a.encode toMode (a.decode fromMode (entry (r.src i,p.2))))
      (List.finRange n) entry i (r.dst i,offset) ⟨rfl,h⟩]
  · simp
  · intro j hj hw
    exact r.distinct hw.1.symm

theorem CopyRegions.frame (r : CopyRegions n) (a : Arithmetic α) (fromMode toMode : Mode)
    (entry : Memory α) (p : Address) (outside : ∀ i,¬r.writes i p) :
    r.run a fromMode toMode entry p=entry p := by
  rw [r.frozen a fromMode toMode entry]
  exact Schedule.fixed_frame _ _ _ _ _ (fun i _ => outside i)

def sequenceItems (f : β → Code α Unit) : List β → Code α Unit
  | [] => pure ()
  | q::tail => do
      let _ ← f q
      sequenceItems f tail

theorem eval_sequenceItems (a : Arithmetic α) (f : β → Code α Unit)
    (step : β → Memory α → Memory α) (localRule : ∀ q m,(f q).eval a m=((),step q m))
    (tasks : List β) (m : Memory α) :
    (sequenceItems f tasks).eval a m=((),Schedule.run step tasks m) := by
  induction tasks generalizing m with
  | nil => rfl
  | cons q tail ih =>
    simp only [sequenceItems,bind,eval_bind,localRule,Prod.fst,Prod.snd,ih]
    rfl

def CopyRegions.code (r : CopyRegions n) (fromMode toMode : Mode) : Code α Unit :=
  sequenceItems (fun i => copyCode fromMode toMode (r.src i) (r.dst i) (r.count i)) (List.finRange n)

theorem CopyRegions.eval_code (r : CopyRegions n) (a : Arithmetic α) (fromMode toMode : Mode) (m : Memory α) :
    (r.code fromMode toMode).eval a m=((),r.run a fromMode toMode m) := by
  exact eval_sequenceItems a _ _ (fun i memory => eval_copyCode a fromMode toMode _ _ _ memory) _ m

#axiomsClean copy_form
#axiomsClean CopyRegions.output
#axiomsClean CopyRegions.frame
#axiomsClean CopyRegions.eval_code
end
end VeriTile.Bench.Repairs.TritonLinearActivation
