import bench.tritonbench_g.kcache_copy_triton.Proofs.HostPrepare

/-! Complete execution of the logical host preparation and every program of
the original three-dimensional grid, for both cache-layout branches and all
natural n. This preserves source behavior, including ignored cache-last and
length strides. It does not establish native bounds, physical pointer alias
mapping, race freedom, Torch primitive refinement, or unspecified dtypes.

The six floating constructors here are inherited logical Storage profiles.
In particular fp64 still has the old opaque legacy carrier; this module adds
no sixth finite-width raw machine format and no native fp64 certification. -/
namespace VeriTile.Bench.TritonBenchG.KcacheCopyTriton.Host
open VeriTile.Triton FP.Structural HostGrid
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 16000
variable {α : Type} [Inhabited α]

/-- Address of a 4D source element in flattened row-major logical order.
The division factors refer to tensor dimensions, not storage strides. -/
def reshapeSourceOffset (key : Tensor) (o : Nat) : Nat :=
  match key.shape, key.strides with
  | [_,b,h,d], [sa,sb,sh,sd] =>
      (o / d / h / b) * sa + (o / d / h % b) * sb + (o / d % h) * sh + (o % d) * sd
  | _, _ => 0

/-- A required reshape copy snapshots all input cells. A view leaves memory
unchanged and retains the original pointer region and computed view strides.
The allocation region is a logical input; native allocation is not proved. -/
def reshapeMemory (h : Inputs) (p : Launch) (m : HostGrid.Memory α) : HostGrid.Memory α :=
  if p.reshaped.copied then fun r o =>
    if r = p.reshaped.tensor.region ∧ o < h.key.shape.prod
    then m h.key.region (reshapeSourceOffset h.key o) else m r o
  else m

def initialState (h : Inputs) (p : Launch) (s : State α) : State α :=
  { s with mem := reshapeMemory h p s.mem }

def expectedMemory (h : Inputs) (p : Launch) (s : State α) : HostGrid.Memory α :=
  HostGrid.reference h.keyProfile p.parameters (HostGrid.programs p.grid)
    (reshapeMemory h p s.mem)

def run (rest : Algebra α) (h : Inputs) (s : State α) : Except Error (Option (State α)) :=
  match prepare h with
  | .error e => .error e
  | .ok p => .ok (HostGrid.run rest h.keyProfile p.parameters p.grid (initialState h p s))

/-- No cell of either the reshape allocation or any program in the grid
writes this address. Program destinations use current reference memory. -/
def Unwritten (h : Inputs) (p : Launch) (s : State α) (r : RegionName) (o : Nat) : Prop :=
  (p.reshaped.copied = Bool.false ∨ r ≠ p.reshaped.tensor.region ∨ h.key.shape.prod ≤ o) ∧
  HostGrid.Unwritten h.keyProfile p.parameters (HostGrid.programs p.grid)
    (reshapeMemory h p s.mem) r o

theorem reshape_frame (h : Inputs) (p : Launch) (m : HostGrid.Memory α)
    (r : RegionName) (o : Nat)
    (miss : p.reshaped.copied = Bool.false ∨ r ≠ p.reshaped.tensor.region ∨ h.key.shape.prod ≤ o) :
    reshapeMemory h p m r o = m r o := by
  rcases miss with noCopy | noRegion | noOffset
  · simp [reshapeMemory,noCopy]
  · cases hc : p.reshaped.copied <;> simp [reshapeMemory,hc,noRegion]
  · cases hc : p.reshaped.copied <;> simp [reshapeMemory,hc,show ¬o < h.key.shape.prod by omega]

private theorem flat_div (outer size inner : Nat) (hi : inner < size) :
    (outer * size + inner) / size = outer := by
  rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega : 0 < size), Nat.div_eq_of_lt hi]
  simp

private theorem flat_lt (outerSize size outer inner : Nat)
    (ho : outer < outerSize) (hi : inner < size) :
    outer * size + inner < outerSize * size := by
  have bound := Nat.mul_le_mul_right size (Nat.succ_le_of_lt ho)
  rw [Nat.succ_mul] at bound
  omega

theorem reshape_source_coordinates (region : RegionName) (a b h d sa sb sh sd : Nat)
    (ia : Fin a) (ib : Fin b) (ih : Fin h) (id : Fin d) :
    reshapeSourceOffset ⟨region,[a,b,h,d],[sa,sb,sh,sd]⟩
      (((ia.val*b+ib.val)*h+ih.val)*d+id.val) =
      ia.val*sa + ib.val*sb + ih.val*sh + id.val*sd := by
  simp only [reshapeSourceOffset, flat_div _ _ _ id.isLt,
    flat_div _ _ _ ih.isLt, flat_div _ _ _ ib.isLt,
    Nat.mul_add_mod_of_lt id.isLt, Nat.mul_add_mod_of_lt ih.isLt, Nat.mul_add_mod_of_lt ib.isLt]

/-- The modeled materialized reshape preserves every original element with
its original four strides, including noncontiguous, zero-stride and aliased
reads. It does not assume the value at a copied destination. -/
theorem reshape_copy_values (inputs : Inputs) (p : Launch) (m : HostGrid.Memory α)
    (a b h d sa sb sh sd : Nat) (shape : inputs.key.shape = [a,b,h,d])
    (strides : inputs.key.strides = [sa,sb,sh,sd]) (copy : p.reshaped.copied = Bool.true)
    (ia : Fin a) (ib : Fin b) (ih : Fin h) (id : Fin d) :
    reshapeMemory inputs p m p.reshaped.tensor.region
      (((ia.val*b+ib.val)*h+ih.val)*d+id.val) =
      m inputs.key.region (ia.val*sa + ib.val*sb + ih.val*sh + id.val*sd) := by
  have first := flat_lt a b ia.val ib.val ia.isLt ib.isLt
  have second := flat_lt (a*b) h _ ih.val first ih.isLt
  have third := flat_lt (a*b*h) d _ id.val second id.isLt
  have bound : ((ia.val*b+ib.val)*h+ih.val)*d+id.val < inputs.key.shape.prod := by
    simpa [shape, Nat.mul_assoc] using third
  simp only [reshapeMemory,copy,↓reduceIte,bound,and_self,↓reduceIte]
  have offset := reshape_source_coordinates inputs.key.region a b h d sa sb sh sd ia ib ih id
  simpa only [reshapeSourceOffset,shape,strides] using congrArg (m inputs.key.region) offset

/-- Total logical wrapper contract. Original modeled host errors remain
errors; every successful preparation executes every grid program, yielding
the complete independent memory reference and frame. No successful run,
post-reshape stride, n=1 restriction, no-alias condition, or numerical callback
law is supplied as a correctness hypothesis. -/
specification kcache_host_grid_correctness (rest : Algebra α) (h : Inputs) (s : State α) :
    match prepare h with
    | .error e => run rest h s = .error e
    | .ok p => ∃ final, run rest h s = .ok (some final) ∧
        final.mem = expectedMemory h p s ∧
        (∀ r o, Unwritten h p s r o → final.mem r o = s.mem r o) := by
  cases hp : prepare h with
  | error e => simp only [run,hp]
  | ok p =>
    obtain ⟨final,hr,hm,hf⟩ := HostGrid.grid_correct rest h.keyProfile p.parameters p.grid (initialState h p s)
    refine ⟨final,?_,hm,?_⟩
    · simp only [run,hp,hr]
    · intro r o miss
      rw [hf r o miss.2]
      exact reshape_frame h p s.mem r o miss.1

#specNonCircular expectedMemory avoiding [run, HostGrid.run, HostGrid.runPrograms,
  FP.Structural.exec, SignedCopy.execution, SignedDiscreteCopy.execution]
#axiomsClean reshape_frame
#axiomsClean reshape_source_coordinates
#axiomsClean reshape_copy_values
#axiomsClean kcache_host_grid_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.KcacheCopyTriton.Host
