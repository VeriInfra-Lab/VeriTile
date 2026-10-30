import bench.tritonbench_g.triton_conv2d_fwd.Proofs.Native
namespace VeriTile.Bench.Repairs.Conv2d
set_option maxRecDepth 16384
set_option maxHeartbeats 1600000
/-- SM90 m64n32 WGMMA output register geometry: four warps, 16 registers
per lane. These coordinates are also checked against every pinned PTX store. -/
def outputRow (thread reg : Nat) := thread/32*16+thread%32/4+reg%4/2*8
def outputCol (thread reg : Nat) := thread%4*2+reg%2+reg/4*8
def outputThread (row col : Nat) := row/16*32+row%8*4+col%8/2
def outputReg (row col : Nat) := col/8*4+row%16/8*2+col%2

theorem output_layout_bounds (thread reg : Nat) (ht : thread<128) (hr : reg<16) :
    outputRow thread reg<64 ∧ outputCol thread reg<32 := by
  unfold outputRow outputCol
  omega

theorem output_layout_covers (row col : Nat) (hr : row<64) (hc : col<32) :
    outputThread row col<128 ∧ outputReg row col<16 ∧
      outputRow (outputThread row col) (outputReg row col)=row ∧
      outputCol (outputThread row col) (outputReg row col)=col := by
  have checked : ∀ r : Fin 64,∀ c : Fin 32,
      outputThread r.val c.val<128 ∧ outputReg r.val c.val<16 ∧
      outputRow (outputThread r.val c.val) (outputReg r.val c.val)=r.val ∧
      outputCol (outputThread r.val c.val) (outputReg r.val c.val)=c.val := by decide
  exact checked ⟨row,hr⟩ ⟨col,hc⟩

theorem output_layout_unique (thread reg : Nat) (ht : thread<128) (hr : reg<16) :
    outputThread (outputRow thread reg) (outputCol thread reg)=thread ∧
      outputReg (outputRow thread reg) (outputCol thread reg)=reg := by
  have checked : ∀ t : Fin 128,∀ r : Fin 16,
      outputThread (outputRow t.val r.val) (outputCol t.val r.val)=t.val ∧
      outputReg (outputRow t.val r.val) (outputCol t.val r.val)=r.val := by decide
  exact checked ⟨thread,ht⟩ ⟨reg,hr⟩
/-- FP16 A register geometry: four packed pairs in each lane. -/
def a16Row (thread reg : Nat) := thread/32*16+thread%32/4+reg%2*8
def a16K (thread reg half : Nat) := thread%4*2+reg/2*8+half

theorem a16_layout_bounds (thread reg half : Nat) (ht : thread<128) (hr : reg<4) (hh : half<2) :
    a16Row thread reg<64 ∧ a16K thread reg half<16 := by
  unfold a16Row a16K
  omega

theorem a16_layout_covers (row k : Nat) (hr : row<64) (hk : k<16) :
    let thread := row/16*32+row%8*4+k%8/2
    let reg := k/8*2+row%16/8
    thread<128 ∧ reg<4 ∧ k%2<2 ∧ a16Row thread reg=row ∧ a16K thread reg (k%2)=k := by
  dsimp only
  unfold a16Row a16K
  omega

theorem fragment_anchor (pid : Program) (lane : Lane) (hl : lane∈lanes) :
    (cellOf pid lane).spatial/64*64=pid.1*128+lane.1/64*64 ∧
    (cellOf pid lane).feature/32*32=pid.2.1*32 ∧ lane.1/64<2 := by
  have h : lane.1<128 ∧ lane.2<32 := by
    rcases lane with ⟨a,b⟩;simpa only [lanes,List.mem_product,List.mem_range] using hl
  simp only [cellOf]
  omega

#axiomsClean output_layout_covers
#axiomsClean output_layout_unique
#axiomsClean a16_layout_covers
#axiomsClean fragment_anchor
end VeriTile.Bench.Repairs.Conv2d
