import bench.tritonbench_g.flash_attn.Proofs.Geometry
import bench.tritonbench_g.flash_attn.Proofs.FlashAttn_Memory
namespace VeriTile.Bench.Repairs.FlashAttn
set_option maxHeartbeats 4000000
abbrev LaneIndex (s : Shape) := Fin (ceil s.length 128) × Fin (s.batches*s.heads) × Fin 128 × Option (Fin s.dim)
def toLane (s : Shape) (i : LaneIndex s) : Lane :=
  ⟨i.1,i.2.1,i.2.2.1,i.2.2.2.map Fin.val⟩
noncomputable def rawLanes (s : Shape) : List Lane :=
  (Finset.univ : Finset (LaneIndex s)).toList.map (toLane s)

theorem toLane_injective (s : Shape) : Function.Injective (toLane s) := by
  rintro ⟨a,b,c,d⟩ ⟨e,f,g,h⟩ he
  simp only [toLane,Lane.mk.injEq] at he
  rcases he with ⟨ha,hb,hc,hd⟩
  have aeq : a=e := Fin.ext ha
  have beq : b=f := Fin.ext hb
  have ceq : c=g := Fin.ext hc
  have deq : d=h := by
    cases d <;> cases h <;> simp_all [Fin.ext_iff]
  simp [aeq,beq,ceq,deq]

theorem raw_mem (s : Shape) (l : Lane) : l∈rawLanes s ↔ l.Valid s := by
  simp only [rawLanes,List.mem_map,Finset.mem_toList,Finset.mem_univ,true_and]
  constructor
  · rintro ⟨i,rfl⟩
    refine ⟨i.1.isLt,i.2.1.isLt,i.2.2.1.isLt,?_⟩
    cases hc : i.2.2.2 with
    | none=>simp [toLane,hc]
    | some d=>simpa [toLane,hc] using d.isLt
  · rintro ⟨hp,hh,hr,hc⟩
    cases he : l.column with
    | none=>exact ⟨(⟨l.pidRow,hp⟩,⟨l.pidHead,hh⟩,⟨l.rowLane,hr⟩,none),by cases l;simp_all [toLane]⟩
    | some d=>
      have hd : d<s.dim := by simpa [he] using hc
      exact ⟨(⟨l.pidRow,hp⟩,⟨l.pidHead,hh⟩,⟨l.rowLane,hr⟩,some ⟨d,hd⟩),by cases l;simp_all [toLane]⟩

noncomputable def sourceSchedule (s : Shape) : List Work :=
  ((rawLanes s).filter (fun l=>decide ((laneWork s l).active s))).map (laneWork s)

theorem schedule_mem (s : Shape) (hs : s.Valid) (w : Work) :
    w∈sourceSchedule s ↔ w.active s := by
  simp only [sourceSchedule,List.mem_map,List.mem_filter,decide_eq_true_eq,raw_mem]
  constructor
  · rintro ⟨l,⟨_,ha⟩,rfl⟩;exact ha
  · intro hw
    obtain ⟨l,hl,he⟩:=lane_coverage s hs w hw
    exact ⟨l,⟨hl,by simpa [he] using hw⟩,he⟩

theorem schedule_nodup (s : Shape) : (sourceSchedule s).Nodup := by
  have raw : (rawLanes s).Nodup :=
    (List.nodup_map_iff (toLane_injective s)).mpr (Finset.nodup_toList _)
  apply (List.nodup_map_iff_inj_on (raw.filter _)).mpr
  intro a ha b hb he
  exact lane_injective s a b ((raw_mem s a).mp (List.mem_filter.mp ha).1)
    ((raw_mem s b).mp (List.mem_filter.mp hb).1) he

def Complete (s : Shape) (ws : List Work) := ws.Perm (sourceSchedule s)

def OutputInjective (x : Layout) : Prop :=
  ∀ b h n d b' h' n' d', (Work.output b h n d).active x.shape →
    (Work.output b' h' n' d').active x.shape →
    address x.o b h n d=address x.o b' h' n' d' → b=b' ∧ h=h' ∧ n=n' ∧ d=d'

theorem target_injective (x : Layout) (ho : OutputInjective x) (a b : Work)
    (ha : a.active x.shape) (hb : b.active x.shape) (he : a.target x=b.target x) : a=b := by
  cases a with
  | output ba ha' na da=>
    cases b with
    | output bb hb' nb db=>
      have hd:=ho ba ha' na da bb hb' nb db ha hb (by simpa [Work.target] using he)
      simp_all
    | logsum bb hb' nb=>simp [Work.target] at he
  | logsum ba ha' na=>
    cases b with
    | output bb hb' nb db=>simp [Work.target] at he
    | logsum bb hb' nb=>
      have hl : (ba*x.shape.heads+ha')*x.shape.length+na=
          (bb*x.shape.heads+hb')*x.shape.length+nb := by simpa [Work.target,lAddress] using he
      have da:=flatten_decode x.shape.length (ba*x.shape.heads+ha') na ha.2.2
      have db:=flatten_decode x.shape.length (bb*x.shape.heads+hb') nb hb.2.2
      have hp : ba*x.shape.heads+ha'=bb*x.shape.heads+hb' := by
        have := congrArg (fun z=>z/x.shape.length) hl
        simpa [da.1,db.1] using this
      have hn : na=nb := by
        have := congrArg (fun z=>z%x.shape.length) hl
        simpa [da.2,db.2] using this
      have ea:=flatten_decode x.shape.heads ba ha' ha.2.1
      have eb:=flatten_decode x.shape.heads bb hb' hb.2.1
      have hbatch : ba=bb := by
        have:=congrArg (fun z=>z/x.shape.heads) hp
        simpa [ea.1,eb.1] using this
      have hhead : ha'=hb' := by
        have:=congrArg (fun z=>z%x.shape.heads) hp
        simpa [ea.2,eb.2] using this
      simp [hbatch,hhead,hn]

theorem call_value (x : Layout) (hs : x.shape.Valid) (ho : OutputInjective x) (p : Ops α)
    (scale : α) (ws : List Work) (complete : Complete x.shape ws) (m : Memory α)
    (w : Work) (active : w.active x.shape) :
    execute x p scale ws m (w.target x).1 (w.target x).2=w.result x p m scale := by
  apply written_value x p scale ws m m (fun _ _ _ _=>rfl) w
  · exact complete.mem_iff.mpr ((schedule_mem x.shape hs w).mpr active)
  · intro v hv he
    exact target_injective x ho v w ((schedule_mem x.shape hs v).mp (complete.mem_iff.mp hv)) active he

def footprint (x : Layout) (r : Region) (a : Nat) : Prop :=
  ∃ w : Work,w.active x.shape ∧ (r,a)=w.target x

theorem call_frame (x : Layout) (hs : x.shape.Valid) (p : Ops α) (scale : α)
    (ws : List Work) (complete : Complete x.shape ws) (m : Memory α) (r : Region) (a : Nat)
    (outside : ¬footprint x r a) : execute x p scale ws m r a=m r a := by
  apply execute_frame
  intro w hw he
  exact outside ⟨w,(schedule_mem x.shape hs w).mp (complete.mem_iff.mp hw),he⟩
#axiomsClean schedule_mem
#axiomsClean schedule_nodup
#axiomsClean target_injective
#axiomsClean call_value
#axiomsClean call_frame
end VeriTile.Bench.Repairs.FlashAttn
