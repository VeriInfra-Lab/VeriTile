import bench.repairs.tritonbench_g.bmm_chunk_fwd.Cleanup
import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.HostReturned
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.BmmChunkFwd.Cleanup
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
noncomputable section
open Classical
set_option maxHeartbeats 3000000

theorem config_member : config∈Host.configurations := by decide

theorem member (h : Host.Call) (a : Nat) : a∈addresses h ↔
    ∃ p i j, active h p i j ∧ a=offset h p i j := by
  simp only [addresses,List.mem_flatMap,List.mem_filterMap,List.mem_finRange,true_and,Finset.mem_toList,Finset.mem_univ]
  constructor
  · rintro ⟨p,i,j,he⟩
    split_ifs at he with valid
    · exact ⟨p,i,j,valid,Option.some.inj he |>.symm⟩
  · rintro ⟨p,i,j,valid,rfl⟩
    exact ⟨p,i,j,by simp [valid]⟩

theorem offset_bound (h : Host.Call) (p : PID h) (i : Fin 32) (j : Fin 64)
    (hi : active h p i j) : offset h p i j<h.shape.outputSize :=
  Host.mul_add_bound (Host.mul_add_bound (Host.mul_add_bound
    (show pid1 p<h.shape.batch from (p ⟨1,by change 1<3; decide⟩).isLt)
    (show pid2 p<h.shape.chunks*h.shape.groups from (p ⟨2,by change 2<3; decide⟩).isLt)) hi.1) hi.2.1

theorem image (h : Host.Call) (a : Nat) : a∈addresses h ↔ Upper h a := by
  rw [member]
  constructor
  · rintro ⟨p,i,j,hi,rfl⟩
    exact ⟨pid1 p,pid2 p,row h p i,column h p j,(p ⟨1,by change 1<3; decide⟩).isLt,
      (p ⟨2,by change 2<3; decide⟩).isLt,hi.1,hi.2.1,hi.2.2,rfl⟩
  · rintro ⟨b,ch,r,c,hb,hch,hr,hc,upper,rfl⟩
    have hg : 0<h.shape.groups := by nlinarith
    have chunks : ch/h.shape.groups<h.shape.chunks := (Nat.div_lt_iff_lt_mul hg).mpr hch
    have group := Nat.mod_lt ch hg
    obtain ⟨p,i,hb',hch',hr',hc'⟩ := Host.coverage h config config_member b (ch/h.shape.groups)
      (ch%h.shape.groups) r c hb chunks group hr hc
    have flat : Host.p2 p=ch := by rw [hch']; exact Nat.div_add_mod' _ _
    refine ⟨p,i.1,i.2.1,?_,?_⟩
    · change row h p i.1<h.shape.chunk ∧ column h p i.2.1<h.shape.chunk ∧ _
      have row' : row h p i.1=r := hr'
      have col' : column h p i.2.1=c := hc'
      rw [row',col']
      exact ⟨hr,hc,upper⟩
    · change _=((Host.p1 p*(h.shape.chunks*h.shape.groups)+Host.p2 p)*h.shape.chunk+
        RealViews.row config.M config.N h.parameters (Host.p0 p) i.1)*h.shape.chunk+
        RealViews.column config.N h.parameters (Host.p0 p) i.2.1
      rw [hb',flat,hr',hc']

theorem fold_frame {α : Type} (zero : α) (xs : List Nat) (before : Nat → α) (a : Nat) (ha : a∉xs) :
    xs.foldl (store zero) before a=before a := by
  induction xs generalizing before with
  | nil => rfl
  | cons x rest ih =>
    rw [List.foldl_cons,ih _ (by simp_all)]
    exact Function.update_of_ne (show a≠x by simp_all) zero before

theorem fold_keep {α : Type} (zero : α) (xs : List Nat) (before : Nat → α) (a : Nat) (ha : before a=zero) :
    xs.foldl (store zero) before a=zero := by
  induction xs generalizing before with
  | nil => exact ha
  | cons x rest ih =>
    rw [List.foldl_cons]
    apply ih
    by_cases eq : a=x
    · subst a; simp [store]
    · simp [store,Function.update_of_ne eq,ha]

theorem fold_read {α : Type} (zero : α) (xs : List Nat) (before : Nat → α) (a : Nat) (ha : a∈xs) :
    xs.foldl (store zero) before a=zero := by
  obtain ⟨l,r,he⟩ := List.append_of_mem ha
  rw [he,List.foldl_append,List.foldl_cons]
  apply fold_keep
  simp [store]

theorem run_value {α : Type} (h : Host.Call) (zero : α) (before : Nat → α) (a : Nat) :
    run h zero before a=if Upper h a then zero else before a := by
  by_cases hu : Upper h a
  · rw [if_pos hu]
    exact fold_read _ _ _ _ ((image h a).mpr hu)
  · rw [if_neg hu]
    exact fold_frame _ _ _ _ (fun ha => hu ((image h a).mp ha))

theorem runAt_value {α : Type} (h : Host.Call) (base : Nat) (zero : α) (before : Nat → α) (a : Nat) :
    runAt h base zero before (base+a)=if Upper h a then zero else before (base+a) := by
  have member : base+a∈(addresses h).map (base+·) ↔ Upper h a := by
    simp only [List.mem_map]
    constructor
    · rintro ⟨x,hx,eq⟩
      have eq' : x=a := Nat.add_left_cancel eq
      subst x; exact (image h a).mp hx
    · intro ha; exact ⟨a,(image h a).mpr ha,rfl⟩
  by_cases hu : Upper h a
  · rw [if_pos hu]
    exact fold_read _ _ _ _ (member.mpr hu)
  · rw [if_neg hu]
    exact fold_frame _ _ _ _ (fun hx => hu (member.mp hx))

theorem runAt_frame {α : Type} (h : Host.Call) (base : Nat) (zero : α) (before : Nat → α) (a : Nat)
    (outside : ∀ off,Upper h off → a≠base+off) : runAt h base zero before a=before a := by
  apply fold_frame
  rintro member
  obtain ⟨off,ho,eq⟩ := List.mem_map.mp member
  exact outside off ((image h off).mp ho) eq.symm

theorem upper_at (h : Host.Call) (b ch r c : Nat) (hb : b<h.shape.batch)
    (hch : ch<h.shape.chunks*h.shape.groups) (hr : r<h.shape.chunk) (hc : c<h.shape.chunk) :
    Upper h (((b*(h.shape.chunks*h.shape.groups)+ch)*h.shape.chunk+r)*h.shape.chunk+c) ↔ r<c := by
  constructor
  · rintro ⟨b',ch',r',c',hb',hch',hr',hc',upper,eq⟩
    obtain ⟨eq,col⟩ := Host.mul_add_inj hc hc' eq
    obtain ⟨eq,row⟩ := Host.mul_add_inj hr hr' eq
    omega
  · intro upper; exact ⟨b,ch,r,c,hb,hch,hr,hc,upper,rfl⟩

theorem lower_active (h : Host.Call) (c : Host.Config) (hc : c∈Host.configurations)
    (r n : Nat) (lower : n≤r) : Host.returnedActive h c r n := by
  right
  have hm := (Host.configuration_positive c hc).1
  have hn := Nat.div_mul_le_self n c.N
  have bound := Nat.div_add_mod' r c.M
  have remainder := Nat.mod_lt r hm
  nlinarith

#axiomsClean run_value
#axiomsClean lower_active
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkFwd.Cleanup
