import bench.tritonbench_g.matmul_persistent_triton.Proofs.NativeCoordinates

/-! Integer output/mask agreement at actual reached iterations of the source
control projection. The source loop is not replaced by independently prepared
ki/row/column values. Floating operations and native memory remain outside
this integer-expression checkpoint. -/
namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.NativeIteration
open NativeDomain NativeBits
set_option maxHeartbeats 3000000

theorem load_tail_mask (p : Parameters) (h : Admissible p) (pid : Nat)
    (hp : pid < grid p) (hk : 0 < count p) (i : Nat) (hi : i < trip p pid)
    (lane : Nat) (hl : lane < p.BK) :
    (bits (lane : Int)).slt
      (NativeAddresses.tailBits p.K (NativeLoop.run p pid (i+1)).ki p.BK) =
        decide ((i%count p)*p.BK+lane < p.K) :=
  (NativeAddresses.tail_exact p h _ lane (Nat.mod_lt i hk) hl _
    (NativeCoordinates.control_after p h pid hp hk i hi).2).2

/-- The actual signed equality controlling the store/reset branch matches the
mathematical last-K predicate in every reached persistent loop iteration. -/
theorem last_k_branch (p : Parameters) (h : Admissible p) (pid : Nat)
    (hp : pid < grid p) (hk : 0 < count p) (i : Nat) (hi : i < trip p pid) :
    ((NativeLoop.run p pid (i+1)).ki == (NativePrefix.eval p pid).nk-bits 1) =
      decide (i%count p+1 = count p) := by
  have hc := (NativePrefix.agrees p h pid hp).nk
  have hn := count_lt p h
  have hone : Exact (bits 1) 1 := bits_exact (by norm_num [Fits])
  have hlast := sub_exact hc hone (by unfold Fits limit at *; omega)
  have hki := (NativeCoordinates.control_after p h pid hp hk i hi).2
  rw [eq_exact hki hlast]
  congr 1
  apply propext
  omega

/-- The two native C products are exact separately. The generic
`two_addresses_exact` theorem then composes their actual pair of 64-bit pointer
advances under an explicit base-plus-byte-span no-wrap bound. -/
theorem store_products (p : Parameters) (h : Admissible p) (pid : Nat)
    (hp : pid < grid p) (hk : 0 < count p) (i : Nat) (hi : i < trip p pid)
    (rl cl : Nat) (hrl : rl < p.BM) (hcl : cl < p.BN) :
    let row := Schedule.row (rows p) (columns p) p.GM (Schedule.tileId p.SMS pid (i/count p))
    let col := Schedule.column (rows p) (columns p) p.GM (Schedule.tileId p.SMS pid (i/count p))
    let coords := NativeCoordinates.run p pid (i+1)
    Exact (bits (p.scm : Int)*NativeAddresses.laneBits coords.row p.BM rl)
      (p.scm*(row*p.BM+rl)) ∧
    Exact (bits (p.scn : Int)*NativeAddresses.laneBits coords.column p.BN cl)
      (p.scn*(col*p.BN+cl)) := by
  have hc := NativeCoordinates.agrees p h pid hp hk i hi
  have hg := Schedule.grouped_bounds (rows p) (columns p) p.GM _ h.gm
    (NativeCoordinates.current_tile_bound p h pid hp hk i hi)
  exact NativeAddresses.c_products_exact p h _ _ rl cl hg.2.1 hg.2.2 hrl hcl _ _ hc.1 hc.2

theorem store_mask (p : Parameters) (h : Admissible p) (pid : Nat)
    (hp : pid < grid p) (hk : 0 < count p) (i : Nat) (hi : i < trip p pid)
    (rl cl : Nat) (hrl : rl < p.BM) (hcl : cl < p.BN) :
    let row := Schedule.row (rows p) (columns p) p.GM (Schedule.tileId p.SMS pid (i/count p))
    let col := Schedule.column (rows p) (columns p) p.GM (Schedule.tileId p.SMS pid (i/count p))
    let coords := NativeCoordinates.run p pid (i+1)
    ((NativeAddresses.laneBits coords.row p.BM rl).slt (bits (p.M : Int)) &&
      (NativeAddresses.laneBits coords.column p.BN cl).slt (bits (p.N : Int))) =
        decide (row*p.BM+rl < p.M ∧ col*p.BN+cl < p.N) := by
  have hc := NativeCoordinates.agrees p h pid hp hk i hi
  have hg := Schedule.grouped_bounds (rows p) (columns p) p.GM _ h.gm
    (NativeCoordinates.current_tile_bound p h pid hp hk i hi)
  have hr := (NativeAddresses.lane_exact _ p.BM rl _ hc.1 (h.scalar _ (by simp)) hrl
    (row_lane_lt p h _ rl hg.2.1 hrl)).2
  have hn := (NativeAddresses.lane_exact _ p.BN cl _ hc.2 (h.scalar _ (by simp)) hcl
    (column_lane_lt p h _ cl hg.2.2 hcl)).2
  exact NativeAddresses.output_mask_exact p h _ _ _ _ hr hn

end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.NativeIteration
