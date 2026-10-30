import bench.tritonbench_g.fused_recurrent_retention.Proofs.Values
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention
noncomputable section

theorem observations_congr (b : InputBuffers) (s : Shape) (m m' : Memory α)
    (h : ∀ p,p.1∈b.storageIds → m p=m' p) : b.observe s m=b.observe s m' := by
  unfold InputBuffers.observe
  congr 1 <;> funext hidx t d <;> apply h <;>
    simp [InputBuffers.storageIds,InputBuffers.qAddress,InputBuffers.kAddress,
      InputBuffers.vAddress,InputBuffers.doAddress,InputBuffers.seedAddress,
      InputBuffers.finalAddress,Tensor4.address]

@[simp] theorem writeWord_here (m : Memory α) (p : Address) (x : α) :
    writeWord m p x p=x := by simp [writeWord]

theorem writeWord_frame (m : Memory α) (p q : Address) (x : α) (hne : q≠p) :
    writeWord m p x q=m q := by simp [writeWord,hne]

theorem fill_get (storage : Nat) (value : Nat → α) (items : List Nat) (m : Memory α) (p : Address) :
    fill storage value items m p=
      if p.1=storage ∧ 0≤p.2 ∧ p.2.toNat∈items then value p.2.toNat else m p := by
  induction items generalizing m with
  | nil => simp [fill]
  | cons i rest ih =>
    change fill storage value rest (writeWord m (storage,(i : Int)) (value i)) p=_
    rw [ih]
    by_cases hp : p=(storage,(i : Int))
    · subst p
      simp [writeWord]
    · have hn : ¬(p.1=storage ∧ 0≤p.2 ∧ p.2.toNat=i) := by
        intro hh
        apply hp
        apply Prod.ext hh.1
        rw [←hh.2.2,Int.toNat_of_nonneg hh.2.1]
      simp only [List.mem_cons]
      by_cases hr : p.1=storage ∧ 0≤p.2 ∧ p.2.toNat∈rest
      · simp [hr]
      · have hn' : ¬(p.1=storage ∧ 0≤p.2 ∧ (p.2.toNat=i ∨ p.2.toNat∈rest)) := by
          tauto
        simp [hr,hn',writeWord,hp]

theorem fillRange_get (storage count i : Nat) (hi : i<count) (value : Nat → α) (m : Memory α) :
    fillRange storage count value m (storage,(i : Int))=value i := by
  simp [fillRange,fill_get,hi]

theorem fillRange_frame (storage count : Nat) (value : Nat → α) (m : Memory α) (p : Address)
    (hp : p.1≠storage) : fillRange storage count value m p=m p := by
  simp [fillRange,fill_get,hp]

theorem fillRange_observations (b : InputBuffers) (s : Shape) (storage count : Nat)
    (value : Nat → α) (m : Memory α) (fresh : storage∉b.storageIds) :
    b.observe s (fillRange storage count value m)=b.observe s m := by
  apply observations_congr
  intro p hp
  apply fillRange_frame
  intro he
  exact fresh (he ▸ hp)

#axiomsClean observations_congr
#axiomsClean fill_get
#axiomsClean fillRange_get
#axiomsClean fillRange_frame
#axiomsClean fillRange_observations
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention
