import bench.repairs.tritonbench_g.context_attn_bloom.Contract

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.Checked
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program RealSetup
open VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host (View3)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000
abbrev bound3 := VeriTile.Bench.Repairs.ContextAttnLlama.Checked.bound3
abbrev unique3 := VeriTile.Bench.Repairs.ContextAttnLlama.Checked.unique3

theorem inside3 (v : View3) (hc : canonical v) (extent : Nat)
    (space : v.rows*v.heads*v.dims ≤ extent) (row : Int) (head dim : Nat)
    (r0 : 0 ≤ row) (hr : row < v.rows) (hh : head < v.heads) (hd : dim < v.dims) :
    AddressBounds.inside (v.address row head dim) extent := by
  rcases hc with ⟨sr,sh,sd⟩
  have hn : row.toNat < v.rows := by omega
  have hn' : (row.toNat : Int)=row := Int.toNat_of_nonneg r0
  have hb := bound3 _ _ _ _ _ _ _ space hn hh hd
  unfold AddressBounds.inside View3.address
  simp only [sr,sh,sd,Nat.cast_one,mul_one]
  constructor
  · positivity
  · rw [←hn']
    exact_mod_cast hb

theorem injective3 (v : View3) (hc : canonical v) : v.Injective := by
  intro row row' head head' dim dim' r0 hr r0' hr' hh hh' hd hd' eq
  rcases hc with ⟨sr,sh,sd⟩
  simp only [View3.address,sr,sh,sd,Nat.cast_one,mul_one] at eq
  have castrow := Int.toNat_of_nonneg r0
  have castrow' := Int.toNat_of_nonneg r0'
  rw [←castrow,←castrow'] at eq
  norm_cast at eq
  have ne : row.toNat*(v.heads*v.dims)+head*v.dims+dim=
      row'.toNat*(v.heads*v.dims)+head'*v.dims+dim' := by exact_mod_cast eq
  have he := unique3 v.rows v.heads v.dims row.toNat row'.toNat head head' dim dim'
    (by omega) (by omega) hh hh' hd hd' ne
  exact ⟨by omega,he.2⟩

theorem valid (a : Host.Arguments) (r : Regions) (A : FlatAlloc) (s : BlockState)
    (g : geometryChecks a r A.extent) (m : metadataChecks a (inputs s r))
    (labels : Labels r) : Host.Admissible a r A s := by
  let u := inputs s r
  change metadataChecks a u at m
  rcases g with ⟨strides,hq,hk,grouped,qk,kv,ok,oh,vh,orows,vrows,dlo,dhi,
    qspace,kspace,vspace,ospace,metaspace,tokenspace⟩
  rcases strides with ⟨qc,kc,vc,oc,rr,rt⟩
  have width : 0 < a.q.dims := by omega
  have kgroup : 0 < a.q.heads/a.k.heads := Nat.div_pos (Nat.le_of_dvd hq grouped) hk
  have hheads : a.k.heads*(a.q.heads/a.k.heads)=a.q.heads := Nat.mul_div_cancel' grouped
  have khBound (h : Nat) (hh : h < a.q.heads) : h/(a.q.heads/a.k.heads) < a.k.heads := by
    apply (Nat.div_lt_iff_lt_mul kgroup).mpr
    simpa only [hheads] using hh
  have metadata (b : Nat) (hb : b < a.batch) :
      0 ≤ u.start b ∧ 0 ≤ u.prompt b ∧ u.prompt b ≤ u.sequence b ∧
      u.sequence b-u.prompt b ≤ a.maxInput ∧ u.start b+(u.sequence b-u.prompt b) ≤ a.q.rows ∧
      u.sequence b ≤ a.requestColumns ∧ 0 ≤ u.request b ∧ u.request b < a.requestRows ∧
      ∀ j : Fin (u.sequence b).toNat,0 ≤ u.tokens (a.requestColumns*(u.request b).toNat+j) ∧
        u.tokens (a.requestColumns*(u.request b).toNat+j) < a.k.rows := m.1 ⟨b,hb⟩
  have pageBound (b j : Nat) (hb : b < a.batch) (hj : (j : Int) < u.sequence b) :
      0 ≤ u.tokens (a.requestColumns*(u.request b).toNat+j) ∧
      u.tokens (a.requestColumns*(u.request b).toNat+j) < a.k.rows :=
    (metadata b hb).2.2.2.2.2.2.2.2 ⟨j,by change j < (u.sequence b).toNat; omega⟩
  have packed : ∀ b c,b < a.batch → c < a.batch → ∀ i j : Int,
      0 ≤ i → i < u.sequence b-u.prompt b → 0 ≤ j → j < u.sequence c-u.prompt c →
      u.start b+i=u.start c+j → b=c ∧ i=j := by
    intro b c hb hc i j i0 hi j0 hj eq
    by_cases hbc : b=c
    · exact ⟨hbc,by subst c; omega⟩
    · have sep := m.2.1 ⟨b,hb⟩ ⟨c,hc⟩ (by change 0 < u.sequence b-u.prompt b; omega) (by change 0 < u.sequence c-u.prompt c; omega)
        (fun h => hbc (congrArg Fin.val h))
      change u.start b+(u.sequence b-u.prompt b) ≤ u.start c ∨
        u.start c+(u.sequence c-u.prompt c) ≤ u.start b at sep
      rcases sep with sep|sep <;> omega
  refine ⟨hq,hk,grouped,qk,kv,ok,oh,vh,width,?_,packed,injective3 a.o oc,?_,?_⟩
  · intro b hb
    have h := metadata b hb
    refine ⟨h.1,h.2.1,h.2.2.1,h.2.2.2.1,h.2.2.2.2.1,?_,h.2.2.2.2.2.1,h.2.2.2.2.2.2.1,h.2.2.2.2.2.2.2.1⟩
    simpa only [orows] using h.2.2.2.2.1
  · exact ⟨labels _ (by simp [readRegions]),labels _ (by simp [readRegions]),
      labels _ (by simp [readRegions]),labels _ (by simp [readRegions]),
      labels _ (by simp [readRegions]),labels _ (by simp [readRegions]),
      labels _ (by simp [readRegions]),labels _ (by simp [readRegions])⟩
  · intro idx
    let l := Host.launchOf a idx
    change AddressBounds.Valid (Host.layout a) r l u A.extent
    have hb : l.batch < a.batch := Host.batch_lt a idx
    have hh : l.head < a.q.heads := Host.head_lt a idx
    have md := metadata l.batch hb
    have lim : limit (Host.layout a) l u ≤ u.sequence l.batch := by
      exact min_le_right _ _
    have tokenEq (j : Nat) :
        token (Host.layout a) l u j=u.tokens (a.requestColumns*(u.request l.batch).toNat+j) := by
      unfold Math.token Host.layout
      rw [rr,rt]
      simp only [Nat.cast_one,mul_one]
      congr 1
      have rq := Int.toNat_of_nonneg md.2.2.2.2.2.2.1
      rw [←rq]
      simp only [←Nat.cast_mul,←Nat.cast_add,Int.toNat_natCast,Nat.mul_comm]
    have qbound (i : Nat) (hi : (queryRow (Host.layout a) l i : Int) < Math.sequence u l) :
        0 ≤ u.start l.batch+queryRow (Host.layout a) l i ∧
        u.start l.batch+queryRow (Host.layout a) l i < a.q.rows := by
      have cap := md.2.2.2.2.1
      unfold Math.sequence at hi
      constructor <;> omega
    refine ⟨lt_of_lt_of_le hb (metaspace _ (by simp)),lt_of_lt_of_le hb (metaspace _ (by simp)),
      lt_of_lt_of_le hb (metaspace _ (by simp)),lt_of_lt_of_le hb (metaspace _ (by simp)),?_,?_,?_,?_,?_⟩
    · intro i d hi hd
      exact inside3 a.q qc _ qspace _ _ _ (qbound i hi).1 (qbound i hi).2 hh hd
    · intro c hc j hj
      have jt : (c*(Host.layout a).BN+j.val : Nat) < a.requestColumns := by
        have := md.2.2.2.2.2.1
        omega
      have rq0 := md.2.2.2.2.2.2.1
      have rqt : (u.request l.batch).toNat < a.requestRows := by
        have := md.2.2.2.2.2.2.2.1
        omega
      have one := Nat.mul_le_mul_right a.requestColumns (Nat.succ_le_of_lt rqt)
      simp only [Nat.succ_mul] at one
      have bound : a.requestColumns*(u.request l.batch).toNat+(c*(Host.layout a).BN+j.val) < A.extent r.tokens := by
        rw [Nat.mul_comm a.requestColumns]
        omega
      unfold AddressBounds.inside AddressBounds.tokenAddress Host.layout
      rw [rr,rt]
      simp only [Nat.cast_one,one_mul]
      have rq := Int.toNat_of_nonneg rq0
      constructor
      · positivity
      · rw [←rq]
        exact_mod_cast bound
    · intro c hc j d hj hd
      have pb := pageBound l.batch (c*(Host.layout a).BN+j.val) hb (lt_of_lt_of_le hj lim)
      have h := inside3 a.k kc _ kspace _ _ _ pb.1 pb.2 (khBound l.head hh) (by simpa only [←qk] using hd)
      dsimp only [AddressBounds.keyAddress]
      rw [tokenEq]
      simpa only [Host.layout,View3.address,Int.natCast_div] using h
    · intro c hc j d hj hd
      have pb := pageBound l.batch (c*(Host.layout a).BN+j.val) hb (lt_of_lt_of_le hj lim)
      have h := inside3 a.v vc _ vspace _ _ _ pb.1 (by simpa only [vrows] using pb.2)
        (by simpa only [vh] using khBound l.head hh) (by simpa only [←kv,←qk] using hd)
      dsimp only [AddressBounds.valueAddress]
      rw [tokenEq]
      simpa only [Host.layout,View3.address,Int.natCast_div] using h
    · intro i d hi hd
      exact inside3 a.o oc _ ospace _ _ _ (qbound i hi).1 (by simpa only [orows] using (qbound i hi).2)
        (by simpa only [oh] using hh) (by simpa only [ok,←qk] using hd)

#axiomsClean valid
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.Checked
