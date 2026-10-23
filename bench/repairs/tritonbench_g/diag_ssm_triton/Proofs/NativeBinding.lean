import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.Tensors

/-! Native integer binding for the actual source's left-associated products,
reverse-loop arithmetic, remainder, padded lanes and component offsets. Byte
capacity and signed metadata come from live tensors, rather than a new logical
shape restriction. A runtime must deliver the stated logical program id; the
separate grid-product/segmentation edge is not silently asserted here. -/
namespace VeriTile.Bench.Repairs.DiagSsmTriton.NativeBinding
open Kernels ByteMemory Execution Tensors NativeIndex
set_option autoImplicit false

structure Metadata (tensor : Tensor) : Prop where
  sizes : ∀ n ∈ tensor.shape, n < 2 ^ 63
  nonnull : 0 < tensor.view.words → 0 < tensor.view.backing.base

structure Shape (route : Route) (d : Dimensions) : Prop where
  allocated : AllocatedShape d.length d.batch d.dim route.components
  batch : d.batch < 2 ^ 63
  dim : d.dim < 2 ^ 63

theorem from_tensors (route : Route) (d : Dimensions) (s x : Tensor)
    (sr : s.route = route) (xr : x.route = route)
    (ss : s.shape = [d.batch, d.dim]) (xs : x.shape = [d.length, d.batch, d.dim])
    (sd : Dense s) (xd : Dense x) (sm : Metadata s) (xm : Metadata x) : Shape route d := by
  have sb := s.view.byte_count_lt sm.nonnull
  have xb := x.view.byte_count_lt xm.nonnull
  have sw : s.view.words = route.components * (d.batch * d.dim) := by
    simpa [Tensor.count, ss, sr] using sd.2.1
  have xw : x.view.words = route.components * (d.length * (d.batch * d.dim)) := by
    simpa [Tensor.count, xs, xr] using xd.2.1
  rw [sw] at sb
  rw [xw] at xb
  refine ⟨⟨?_, xm.sizes d.length (by simp [xs]), ?_, ?_⟩,
    sm.sizes d.batch (by simp [ss]), sm.sizes d.dim (by simp [ss])⟩
  · cases route <;> simp [Route.components]
  · nlinarith
  · nlinarith

def inRange (n : Nat) : Prop := n < 2 ^ 63

theorem exact_nat (n : Nat) (h : inRange n) : i64 (n : Int) = n := by
  apply i64_exact
  · omega
  · exact_mod_cast h

theorem exact_add (a b : Nat) (h : inRange (a + b)) :
    i64 ((a : Int) + b) = (a + b : Nat) := by
  simpa using exact_nat (a + b) h

theorem exact_mul (a b : Nat) (h : inRange (a * b)) :
    i64 ((a : Int) * b) = (a * b : Nat) := by
  simpa using exact_nat (a * b) h

/-- Every nonconstant integer intermediate actually evaluated by a launched
lane, including inactive padded lanes. The time-dependent fields apply only
inside an entered loop. Static loads remain covered when length is zero. -/
structure StaticBounds (route : Route) (d : Dimensions) (pid lane : Nat) : Prop where
  length : inRange d.length
  batch : inRange d.batch
  dim : inRange d.dim
  pidBound : inRange pid
  laneBound : inRange lane
  width : inRange (d.batch * d.dim)
  block : inRange (pid * 128)
  columnBound : inRange (column pid lane)
  componentBlock : inRange (route.components * (pid * 128))
  componentColumn : ∀ k < route.components, inRange (route.components * column pid lane + k)
  componentBatch : inRange (route.components * d.batch)
  componentWidth : inRange (route.components * (d.batch * d.dim))
  remainder : ∀ k < route.components, inRange (route.components * (column pid lane % d.dim) + k)
  positiveDim : 0 < d.dim

structure TimeBounds (route : Route) (d : Dimensions) (pid lane t : Nat) : Prop where
  time : inRange t
  timeBatch : inRange (t * d.batch)
  timeWidth : inRange (t * d.batch * d.dim)
  offset : inRange (t * d.batch * d.dim + column pid lane)
  componentOffset : ∀ k < route.components,
    inRange (route.components * (t * d.batch * d.dim + column pid lane) + k)
  previous : 0 < t → ∀ k < route.components,
    inRange (route.components * ((t - 1) * d.batch * d.dim + column pid lane) + k)

theorem static_bounds (route : Route) (d : Dimensions) (h : Shape route d)
    (pid lane : Nat) (hp : pid < grid d.width) (hl : lane < 128) : StaticBounds route d pid lane := by
  have counts := (allocated_counts h.allocated).1
  have padded := padded_column d.width pid lane hp hl
  have positive : 0 < d.batch * d.dim := by
    by_contra hz
    have hz : d.width = 0 := by dsimp [Dimensions.width]; omega
    simp [hz, grid] at hp
  have hd : 0 < d.dim := by nlinarith
  have hb : 0 < d.batch := by nlinarith
  have col : ∀ k < route.components, route.components * column pid lane + k < 2 ^ 63 :=
    fun k hk => padded_static_i64 h.allocated pid lane k hp hl hk
  have hc : 0 < route.components := by cases route <;> decide
  have bn : d.batch ≤ d.batch * d.dim := by nlinarith
  have cn := Nat.mul_le_mul_left route.components bn
  have c0 := col 0 hc
  refine ⟨h.allocated.time_size, h.batch, h.dim, ?_, ?_, ?_, ?_, ?_, ?_, col, ?_, ?_, ?_, hd⟩
  all_goals unfold inRange
  · unfold column at c0; nlinarith
  · omega
  · nlinarith
  · unfold column at c0; nlinarith
  · nlinarith
  · unfold column at c0; nlinarith
  · omega
  · omega
  · intro k hk
    have rem := Nat.mod_lt (column pid lane) hd
    have dimle : d.dim ≤ d.batch * d.dim := by nlinarith
    have remle : column pid lane % d.dim + 1 ≤ d.batch * d.dim := by omega
    have cr := Nat.mul_le_mul_left route.components remle
    nlinarith

theorem time_bounds (route : Route) (d : Dimensions) (h : Shape route d)
    (pid lane t : Nat) (hp : pid < grid d.width) (hl : lane < 128) (ht : t < d.length) :
    TimeBounds route d pid lane t := by
  have sb := static_bounds route d h pid lane hp hl
  have c : 0 < route.components := by cases route <;> decide
  have component : ∀ k < route.components,
      route.components * (t * d.batch * d.dim + column pid lane) + k < 2 ^ 63 := by
    intro k hk
    simpa [offset, Nat.mul_assoc] using padded_time_i64 h.allocated t pid lane k ht hp hl hk
  have c0 := component 0 c
  have lower : t * d.batch * d.dim + column pid lane ≤
      route.components * (t * d.batch * d.dim + column pid lane) := by
    simpa using Nat.mul_le_mul_right (t * d.batch * d.dim + column pid lane) c
  have first : t * d.batch ≤ t * d.batch * d.dim := by
    simpa using Nat.mul_le_mul_left (t * d.batch) sb.positiveDim
  refine ⟨?_, ?_, ?_, ?_, component, ?_⟩
  · exact lt_trans ht h.allocated.time_size
  · unfold inRange; omega
  · unfold inRange; nlinarith
  · unfold inRange; nlinarith
  · intro ht0 k hk
    simpa [offset, Nat.mul_assoc] using
      padded_time_i64 h.allocated (t - 1) pid lane k (by omega) hp hl hk

/-- The actual source order t*batch*dim is retained, rather than assuming
only the final expression fits. The component scale occurs after addition. -/
theorem source_offsets (route : Route) (d : Dimensions) (h : Shape route d)
    (pid lane t k : Nat) (hp : pid < grid d.width) (hl : lane < 128)
    (ht : t < d.length) (hk : k < route.components) :
    let b := i64 d.batch
    let dim := i64 d.dim
    let col := i64 (i64 (i64 pid * 128) + lane)
    let off := i64 (i64 (i64 (i64 t * b) * dim) + col)
    b = d.batch ∧ dim = d.dim ∧ col = column pid lane ∧
      off = (t * d.batch * d.dim + column pid lane : Nat) ∧
      i64 (off * route.components + k) =
        (route.components * (t * d.batch * d.dim + column pid lane) + k : Nat) := by
  have sb := static_bounds route d h pid lane hp hl
  have tb := time_bounds route d h pid lane t hp hl ht
  dsimp only
  have bp : i64 ((pid : Int) * 128) = (pid * 128 : Nat) := by
    simpa using exact_mul pid 128 sb.block
  rw [exact_nat _ sb.batch, exact_nat _ sb.dim, exact_nat _ sb.pidBound,
    exact_nat _ tb.time, bp]
  have cc : i64 ((pid * 128 : Nat) + (lane : Int)) = column pid lane := by
    simpa [column] using exact_add (pid * 128) lane sb.columnBound
  rw [cc, exact_mul t d.batch tb.timeBatch, exact_mul (t * d.batch) d.dim tb.timeWidth]
  have off : i64 ((t * d.batch * d.dim : Nat) + (column pid lane : Int)) =
      (t * d.batch * d.dim + column pid lane : Nat) := by
    simpa using exact_nat _ tb.offset
  rw [off]
  refine ⟨rfl, rfl, rfl, rfl, ?_⟩
  simpa [Nat.mul_comm, Int.mul_comm] using exact_nat _ (tb.componentOffset k hk)

/-- Reverse-loop subtraction and previous-state subtraction are exact on
entered iterations. All dereferenced bytes use the actual nonzero view offset. -/
theorem source_reverse (route : Route) (d : Dimensions) (h : Shape route d)
    (i : Nat) (hi : i < d.length) :
    i64 (i64 (i64 d.length - 1) - i64 i) = (d.length - 1 - i : Nat) ∧
      i64 (i64 i + 1) = (i + 1 : Nat) := by
  have li : inRange i := lt_trans hi h.allocated.time_size
  have lm : inRange (d.length - 1) := by unfold inRange; have := h.allocated.time_size; omega
  rw [exact_nat _ h.allocated.time_size, exact_nat _ li]
  have first : i64 ((d.length : Int) - 1) = (d.length - 1 : Nat) := by
    have eq : (d.length : Int) - 1 = (d.length - 1 : Nat) := by omega
    rw [eq, exact_nat _ lm]
  rw [first]
  constructor
  · have eq : ((d.length - 1 : Nat) : Int) - i = (d.length - 1 - i : Nat) := by omega
    rw [eq]
    apply exact_nat
    unfold inRange
    have := h.allocated.time_size
    omega
  · exact exact_add i 1 (by unfold inRange; have := h.allocated.time_size; omega)

theorem active_addresses (v : View) (components count index k : Nat)
    (size : v.words = components * count) (hi : index < count) (hk : k < components) :
    v.address (components * index + k) + 4 ≤ 2 ^ 64 ∧
      ∀ byte : Fin 4, v.backing.contains (v.address (components * index + k) + byte.val) := by
  have bound : components * index + k < v.words := by rw [size]; nlinarith
  exact ⟨v.address_native _ bound, fun byte => v.byte_inside _ bound byte⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DiagSsmTriton.NativeBinding
