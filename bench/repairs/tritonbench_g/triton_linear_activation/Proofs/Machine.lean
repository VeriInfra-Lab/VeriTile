import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.Bounds
namespace VeriTile.Bench.Repairs.TritonLinearActivation
noncomputable section
set_option maxHeartbeats 4000000

theorem ceil_le (n block : Nat) (hb : 0<block) : ceilDiv n block≤n := by
  have mul : n≤n*block := by simpa using Nat.mul_le_mul_left n hb
  have sub : n+block-1+1=n+block := by omega
  have hp : n+block-1<(n+1)*block := by nlinarith
  exact Nat.le_of_lt_succ ((Nat.div_lt_iff_lt_mul hb).mpr hp)

theorem source_scalar_bounds (s : Shape) (c : Fin 98) (safe : SourceBounds s)
    (pid : Nat) (hp : pid<s.programs c) :
    s.m+(configs c).bm-1<2^31 ∧ s.n+(configs c).bn-1<2^31 ∧
    s.programs c<2^31 ∧ 8*s.cols c<2^31 ∧
    pid<2^31 ∧ pid/(8*s.cols c)<2^31 ∧
    groupBase (s.rows c) (s.cols c) 8 pid<s.rows c ∧
    s.rows c-groupBase (s.rows c) (s.cols c) 8 pid<2^31 ∧
    0<groupSize (s.rows c) (s.cols c) 8 pid ∧
    groupSize (s.rows c) (s.cols c) 8 pid≤8 ∧
    tileRow (s.rows c) (s.cols c) 8 pid<2^31 ∧
    tileCol (s.rows c) (s.cols c) 8 pid<2^31 := by
  have cfg := config_bounds c
  have bm : 0<(configs c).bm := by omega
  have bn : 0<(configs c).bn := by omega
  have rows : s.rows c≤s.m := ceil_le _ _ bm
  have cols : s.cols c≤s.n := ceil_le _ _ bn
  have up : s.cols c*(configs c).bn<s.n+(configs c).bn := ceil_upper _ _ bn
  have grid : s.programs c≤s.m*s.n := Nat.mul_le_mul rows cols
  have g := grouped_bounds (s.rows c) (s.cols c) 8 pid (by decide) hp
  have sn : s.n≤max s.m (max s.n s.k) := le_trans (le_max_left _ _) (le_max_right _ _)
  have sm : s.m≤max s.m (max s.n s.k) := le_max_left _ _
  have product : s.m*s.n≤(s.m+256)*max s.n 1 :=
    Nat.mul_le_mul (by omega) (le_max_left _ _)
  have pc : 0<s.cols c := by unfold Shape.programs at hp; nlinarith
  have base : groupBase (s.rows c) (s.cols c) 8 pid<s.rows c := by
    have hd := Nat.div_mul_le_self pid (8*s.cols c)
    unfold groupBase
    unfold Shape.programs at hp
    nlinarith
  have leSize := min_le_right (s.rows c-groupBase (s.rows c) (s.cols c) 8 pid) 8
  have divLe := Nat.div_le_self pid (8*s.cols c)
  unfold SourceBounds at safe
  refine ⟨by omega,by omega,by omega,?_,by omega,by omega,base,by omega,g.1,leSize,by omega,by omega⟩
  have mul : 8*s.cols c≤s.cols c*(configs c).bn := by nlinarith
  omega

theorem signed32_exact (x : Nat) (h : x<2^31) : (BitVec.ofNat 32 x).toInt=(x : Int) := by
  have hx : x<2^32 := by omega
  have hn : (BitVec.ofNat 32 x).toNat=x := by rw [BitVec.toNat_ofNat]; exact Nat.mod_eq_of_lt hx
  rw [BitVec.toInt_eq_toNat_of_lt (by rw [hn]; omega),hn]

def affine32 (r stride column : Nat) : BitVec 32 :=
  BitVec.ofNat 32 r*BitVec.ofNat 32 stride+BitVec.ofNat 32 column

theorem affine32_exact (r stride column : Nat) (bound : r*stride+column<2^31) :
    (affine32 r stride column).toInt=(r*stride+column : Nat) := by
  simpa only [affine32,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat] using signed32_exact _ bound

/-- All formed output addresses, including masked lanes, are exact signed32
affine expressions. The byte displacement after pointer extension fits signed64. -/
theorem output_machine_exact (s : Shape) (c : Fin 98) (safe : SourceBounds s)
    (pid i j : Nat) (hp : pid<s.programs c) (hi : i<(configs c).bm) (hj : j<(configs c).bn) :
    (affine32 (tileRow (s.rows c) (s.cols c) 8 pid) (configs c).bm i).toInt=(row s c pid i : Int) ∧
    (affine32 (tileCol (s.rows c) (s.cols c) 8 pid) (configs c).bn j).toInt=(col s c pid j : Int) ∧
    (affine32 (row s c pid i) (max s.n 1) (col s c pid j)).toInt=
      (row s c pid i*max s.n 1+col s c pid j : Nat) ∧
    4*(row s c pid i*max s.n 1+col s c pid j)<2^63 := by
  have bounds := formed_output s c safe pid i j hp hi hj
  exact ⟨affine32_exact _ _ _ bounds.1,affine32_exact _ _ _ bounds.2.1,
    affine32_exact _ _ _ bounds.2.2,by omega⟩

/-- The actual A/B pointer progression includes the final increment. -/
theorem input_machine_exact (rows K block lane rowIndex steps : Nat)
    (hb : 0<block) (hb128 : block≤128) (hl : lane<block) (hr : rowIndex<rows)
    (hs : steps≤ceilDiv K block) (safe : rows*max K 1+K+256<2^31) :
    (BitVec.ofNat 32 rowIndex*BitVec.ofNat 32 (max K 1)+
      BitVec.ofNat 32 steps*BitVec.ofNat 32 block+BitVec.ofNat 32 lane).toInt=
      (rowIndex*max K 1+steps*block+lane : Nat) ∧
    4*(rowIndex*max K 1+steps*block+lane)<2^63 := by
  have bound := formed_input rows K block lane rowIndex steps hb hb128 hl hr hs safe
  constructor
  · simpa only [BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat] using signed32_exact _ bound
  · omega

theorem loaded_input_bound (rows K rowIndex block iteration lane : Nat)
    (hb : 0<block) (hr : rowIndex<rows) (it : iteration<ceilDiv K block) (hl : lane<block)
    (load : K%block=0 ∨ lane<K-iteration*block) :
    rowIndex*max K 1+iteration*block+lane<rows*K := by
  have hk := (source_k_mask K block iteration lane hb it hl).mp load
  have positive : 1≤K := by omega
  rw [max_eq_left positive]
  have := Nat.mul_le_mul_right K (Nat.succ_le_of_lt hr)
  nlinarith

#axiomsClean source_scalar_bounds
#axiomsClean output_machine_exact
#axiomsClean input_machine_exact
#axiomsClean loaded_input_bound
end
end VeriTile.Bench.Repairs.TritonLinearActivation
