import bench.repairs.tritonbench_g.flash_attn.Kernels
-- Preserve the historical Bool spellings after importing the shared Triton DSL.
local notation "true" => Bool.true
local notation "false" => Bool.false
namespace VeriTile.Bench.Repairs.FlashAttn
set_option maxHeartbeats 4000000
structure Lane where
  pidRow : Nat
  pidHead : Nat
  rowLane : Nat
  column : Option Nat
  deriving DecidableEq, Repr

def Lane.Valid (s : Shape) (l : Lane) : Prop :=
  l.pidRow<ceil s.length 128 ∧ l.pidHead<s.batches*s.heads ∧ l.rowLane<128 ∧
  match l.column with | none=>True | some d=>d<s.dim

def laneWork (s : Shape) (l : Lane) : Work :=
  match l.column with
  | some d=>.output (l.pidHead/s.heads) (l.pidHead%s.heads) (l.pidRow*128+l.rowLane) d
  | none=>.logsum (l.pidHead/s.heads) (l.pidHead%s.heads) (l.pidRow*128+l.rowLane)

theorem flatten_bounds (s : Shape) (hs : s.Valid) (b h : Nat)
    (hb : b<s.batches) (hh : h<s.heads) : b*s.heads+h<s.batches*s.heads := by
  have := Nat.mul_le_mul_right s.heads (show b+1≤s.batches by omega)
  nlinarith

theorem flatten_decode (heads b h : Nat) (hh : h<heads) :
    (b*heads+h)/heads=b ∧ (b*heads+h)%heads=h := by
  have hp : 0<heads := by omega
  simp [Nat.add_div,Nat.add_mod,Nat.mod_eq_of_lt hh,Nat.div_eq_of_lt hh,hp,hh]

theorem lane_coverage (s : Shape) (hs : s.Valid) (w : Work) (hw : w.active s) :
    ∃ l,l.Valid s ∧ laneWork s l=w := by
  cases w with
  | output b h n d =>
    rcases hw with ⟨hb,hh,hn,hd⟩
    have dec := flatten_decode s.heads b h hh
    refine ⟨⟨n/128,b*s.heads+h,n%128,some d⟩,?_,?_⟩
    · exact ⟨by dsimp [ceil];omega,flatten_bounds s hs b h hb hh,Nat.mod_lt _ (by decide),hd⟩
    · simp [laneWork,dec]; omega
  | logsum b h n =>
    rcases hw with ⟨hb,hh,hn⟩
    have dec := flatten_decode s.heads b h hh
    refine ⟨⟨n/128,b*s.heads+h,n%128,none⟩,?_,?_⟩
    · exact ⟨by dsimp [ceil];omega,flatten_bounds s hs b h hb hh,Nat.mod_lt _ (by decide),trivial⟩
    · simp [laneWork,dec]; omega

theorem lane_injective (s : Shape) (a b : Lane) (ha : a.Valid s) (hb : b.Valid s)
    (he : laneWork s a=laneWork s b) : a=b := by
  rcases a with ⟨a₀,a₁,ai,ac⟩
  rcases b with ⟨b₀,b₁,bi,bc⟩
  have hheadsA:=Nat.div_add_mod a₁ s.heads
  have hheadsB:=Nat.div_add_mod b₁ s.heads
  dsimp [Lane.Valid] at ha hb
  rcases ha with ⟨_,_,hai,_⟩
  rcases hb with ⟨_,_,hbi,_⟩
  cases ac <;> cases bc <;> simp only [laneWork,Work.output.injEq,Work.logsum.injEq] at he
  · rcases he with ⟨hd,hm,hr⟩
    have hp : a₁=b₁ := by rw [hd,hm] at hheadsA;omega
    have h0 : a₀=b₀ := by omega
    have hi : ai=bi := by omega
    simp_all
  · contradiction
  · contradiction
  · rcases he with ⟨hd,hm,hr,hc⟩
    have hp : a₁=b₁ := by rw [hd,hm] at hheadsA;omega
    have h0 : a₀=b₀ := by omega
    have hi : ai=bi := by omega
    simp_all

theorem row_envelope (s : Shape) (l : Lane) (hl : l.Valid s) :
    l.pidRow*128+l.rowLane<s.length+127 := by
  rcases hl with ⟨hp,_,hi,_⟩
  dsimp [ceil] at hp
  omega

theorem stop_le (s : Shape) (pid : Nat) : stop s pid≤s.length := by
  unfold stop
  split <;> simp

theorem rounds_envelope (s : Shape) (pid : Nat) : rounds s pid*64<s.length+64 := by
  have := stop_le s pid
  unfold rounds ceil
  omega

theorem visible_visited (s : Shape) (row key : Nat) (hr : row<s.length)
    (hk : visible s row key=true) : key<rounds s (row/128)*64 := by
  have hh : key<s.length ∧ (s.causal=false ∨ key≤row) := by simpa [visible] using hk
  have hkN := hh.1
  have hstop : key<stop s (row/128) := by
    unfold stop
    cases hc : s.causal
    · simp [hc,hkN]
    · have hkr : key≤row := by simpa [hc] using hh.2
      simp [hc]
      omega
  unfold rounds ceil
  omega

def envelope (s : Shape) (t : Strides) : Nat :=
  (s.batches-1)*t.batch+(s.heads-1)*t.head+(s.length+127)*t.row+(s.dim-1)*t.dim

theorem address_formed_bound (s : Shape) (t : Strides) (b h n d : Nat)
    (hb : b<s.batches) (hh : h<s.heads) (hn : n≤s.length+127) (hd : d<s.dim) :
    address t b h n d≤envelope s t := by
  have h₁:=Nat.mul_le_mul_right t.batch (show b≤s.batches-1 by omega)
  have h₂:=Nat.mul_le_mul_right t.head (show h≤s.heads-1 by omega)
  have h₃:=Nat.mul_le_mul_right t.row hn
  have h₄:=Nat.mul_le_mul_right t.dim (show d≤s.dim-1 by omega)
  unfold address envelope
  omega

theorem l_bound (s : Shape) (b h n : Nat) (hb : b<s.batches) (hh : h<s.heads)
    (hn : n<s.length) : lAddress s b h n<s.batches*s.heads*s.length := by
  have hp : b*s.heads+h<s.batches*s.heads := by
    have:=Nat.mul_le_mul_right s.heads (show b+1≤s.batches by omega)
    nlinarith
  have := Nat.mul_le_mul_right s.length (show b*s.heads+h+1≤s.batches*s.heads by omega)
  unfold lAddress
  nlinarith

#axiomsClean lane_coverage
#axiomsClean lane_injective
#axiomsClean address_formed_bound
#axiomsClean visible_visited
end VeriTile.Bench.Repairs.FlashAttn
