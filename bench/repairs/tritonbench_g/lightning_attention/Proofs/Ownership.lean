import bench.repairs.tritonbench_g.lightning_attention.Proofs.Frame
namespace VeriTile.Bench.Repairs.LightningAttention
open VeriTile Triton FP.Structural
set_option maxHeartbeats 6000000
open VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry (packed_injective)

theorem fwd_owner (d : Host.Dimensions) (h : Host.Contract d) (a b : PID)
    (ha : ValidPID .forward d a) (hb : ValidPID .forward d b) (r : RegionName) (o : Nat)
    (wa : FwdFoot d.parameters a r o) (wb : FwdFoot d.parameters b r o) : a=b := by
  obtain ⟨_,ta,hta,fa,hfa,ea⟩ := wa
  obtain ⟨_,tb,htb,fb,hfb,eb⟩ := wb
  have ca := Geometry.channel_active d h a.channel ha.2 ⟨fa,hfa⟩
  have cb := Geometry.channel_active d h b.channel hb.2 ⟨fb,hfb⟩
  obtain ⟨hh,_,hc⟩ := Geometry.dense_collision hta htb ca cb (ea.symm.trans eb)
  obtain ⟨hv,_⟩ := packed_injective hfa hfb hc
  exact Geometry.pid_ext hh hv

theorem rows_owner (N width rows : Nat) (a b : PID) (dest other r : RegionName) (o : Nat)
    (wa : RowsFoot N width a.head (a.channel*rows) rows dest r o)
    (wb : RowsFoot N width b.head (b.channel*rows) rows other r o) : a=b := by
  obtain ⟨_,la,hla,hta,fa,hfa,ea⟩ := wa
  obtain ⟨_,lb,hlb,htb,fb,hfb,eb⟩ := wb
  obtain ⟨hh,ht,_⟩ := Geometry.dense_collision hta htb hfa hfb (ea.symm.trans eb)
  obtain ⟨hc,_⟩ := packed_injective hla hlb ht
  exact Geometry.pid_ext hh hc

theorem intra_owner (p : Parameters) (a b : PID) (r : RegionName) (o : Nat)
    (wa : IntraFoot p a.head a.channel r o) (wb : IntraFoot p b.head b.channel r o) : a=b := by
  rcases wa with wa|wa|wa <;> rcases wb with wb|wb|wb
  all_goals first
  | exact rows_owner _ _ _ _ _ _ _ _ _ wa wb
  | have hr := wa.1.symm.trans wb.1; simp at hr

theorem head_owner (N width : Nat) (a b : PID) (dest other r : RegionName) (o : Nat)
    (wa : HeadFoot N width a.head dest r o) (wb : HeadFoot N width b.head other r o) : a.head=b.head := by
  obtain ⟨_,ta,hta,fa,hfa,ea⟩ := wa
  obtain ⟨_,tb,htb,fb,hfb,eb⟩ := wb
  exact (Geometry.dense_collision hta htb hfa hfb (ea.symm.trans eb)).1

theorem inter_owner (d : Host.Dimensions) (a b : PID)
    (ha : ValidPID .inter d a) (hb : ValidPID .inter d b) (r : RegionName) (o : Nat)
    (wa : InterFoot d.parameters a.head r o) (wb : InterFoot d.parameters b.head r o) : a=b := by
  have hc : a.channel=b.channel := by
    have ha' : a.channel<1 := ha.2
    have hb' : b.channel<1 := hb.2
    omega
  apply Geometry.pid_ext _ hc
  rcases wa with wa|wa|wa <;> rcases wb with wb|wb|wb
  all_goals first
  | exact head_owner _ _ _ _ _ _ _ _ wa wb
  | have hr := wa.1.symm.trans wb.1; simp at hr

theorem footprint_unique (stage : Stage) (d : Host.Dimensions) (h : Host.Contract d) (a b : PID)
    (ha : ValidPID stage d a) (hb : ValidPID stage d b) (r : RegionName) (o : Nat)
    (wa : Footprint stage d a r o) (wb : Footprint stage d b r o) : a=b := by
  cases stage with
  | forward => exact fwd_owner _ h _ _ ha hb _ _ wa wb
  | intra => exact intra_owner _ _ _ _ _ wa wb
  | inter => exact inter_owner _ _ _ ha hb _ _ wa wb

/-- Only the four returned tensor regions may be changed. -/
def regionCapacity (d : Host.Dimensions) (r : RegionName) : Nat :=
  if r="output" ∨ r="valueGradient" then d.B*d.H*d.N*d.E
  else if r="queryGradient" ∨ r="keyGradient" then d.B*d.H*d.N*d.D else 0

theorem rows_bound (N width rows heads head start : Nat) (dest r : RegionName) (o : Nat)
    (hh : head<heads) (hw : RowsFoot N width head start rows dest r o) : o<heads*N*width := by
  obtain ⟨_,lane,_,ht,feature,hf,he⟩ := hw
  rw [he]
  exact Geometry.dense_bound hh ht hf

theorem head_bound (N width heads head : Nat) (dest r : RegionName) (o : Nat)
    (hh : head<heads) (hw : HeadFoot N width head dest r o) : o<heads*N*width := by
  obtain ⟨_,time,ht,feature,hf,he⟩ := hw
  rw [he]
  exact Geometry.dense_bound hh ht hf

theorem footprint_bound (stage : Stage) (d : Host.Dimensions) (h : Host.Contract d)
    (pid : PID) (hp : ValidPID stage d pid) (r : RegionName) (o : Nat)
    (hw : Footprint stage d pid r o) : o<regionCapacity d r := by
  cases stage with
  | forward =>
    obtain ⟨rfl,time,ht,feature,hf,he⟩ := hw
    simp only [regionCapacity,or_self,if_true]
    rw [he]
    exact Geometry.dense_bound hp.1 ht (Geometry.channel_active d h pid.channel hp.2 ⟨feature,hf⟩)
  | intra =>
    rcases hw with hw|hw|hw
    all_goals
      have bound := rows_bound _ _ _ _ _ _ _ _ _ hp.1 hw
      rcases hw with ⟨rfl,_⟩
      simpa [regionCapacity] using bound
  | inter =>
    rcases hw with hw|hw|hw
    all_goals
      have bound := head_bound _ _ _ _ _ _ _ hp.1 hw
      rcases hw with ⟨rfl,_⟩
      simpa [regionCapacity] using bound

#axiomsClean footprint_unique
#axiomsClean footprint_bound
end VeriTile.Bench.Repairs.LightningAttention
