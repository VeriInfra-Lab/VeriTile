import bench.tritonbench_g.uniform_sampling.Proofs.KernelOffsets

namespace VeriTile.Bench.Repairs.UniformSampling.KernelStores
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.UniformSampling
open KernelOffsets (output offsets)
noncomputable section
set_option maxHeartbeats 2500000
set_option maxRecDepth 20000

def pointer (N B : Nat) (name : RegName) : Op .nat [B] :=
  .castIntToNat (.where
    (.lt .int .scalarR (.ref .int [B] name)
      (.castNatToInt (.sub .nat .nil (.constNat N) (.constNat 1))))
    (.ref .int [B] name)
    (.broadcast (.castNatToInt (.sub .nat .nil (.constNat N) (.constNat 1))) [B]))

def stmt (y : RegionName) (N B : Nat) (valueName offsetName : RegName) : ComputeStmt :=
  .store .real [B] (.region y (pointer N B offsetName))
    (.compute (.alg .fp32 (.castFloat .real .real (.ref .real [B] valueName))))
    (.mask (.lt .int .scalarR (.ref .int [B] offsetName) (.castNatToInt (.constNat N))))

def body (y : RegionName) (N B : Nat) : List ComputeStmt := [
  stmt y N B "r0" "off_0",stmt y N B "r1" "off_1",
  stmt y N B "r2" "off_2",stmt y N B "r3" "off_3"]

def field {α : Type} (y : RegionName) (N pid B : Nat) (f : Fin 4)
    (values : TileIndex [B] → α) (s : State α) : State α :=
  (TileShape.allIndices [B]).foldl (fun t i =>
    if output pid B f.val i.1.val < N then
      t.write y (output pid B f.val i.1.val) (.mk .real (values i)) else t) s

@[simp] theorem field_regs {α : Type} (y : RegionName) (N pid B : Nat) (f : Fin 4)
    (values : TileIndex [B] → α) (s : State α) :
    (field y N pid B f values s).regs = s.regs := by
  rw [field,State.masked_scatter_eq_filter,State.scatter_regs]

theorem step_exec {α : Type} [Inhabited α] (M : Algebra α) (y : RegionName)
    (N pid B : Nat) (f : Fin 4) (valueName offsetName : RegName)
    (values : Values α .real [B]) (s : State α)
    (hv : s.regs .real [B] valueName = some values)
    (ho : s.regs .int [B] offsetName = some (offsets pid B f.val)) :
    step M (stmt y N B valueName offsetName) s =
      some (field y N pid B f (fun i => M.cast (some .fp32) .real .real (values i)) s) := by
  simp [stmt,pointer,step,evalExpr,evalComputeOp,evalOp_unfold,hv,ho,store,
    numeric,numericLt,bop,ofFloat,toFloat,Region.cast,offsets,ComputeDType.eraseDType,Option.bind]
  unfold field
  congr 1
  funext t i
  by_cases active : output pid B f.val i.1.val < N
  · simp only [active,if_true]
    congr 1
    split <;> omega
  · simp [active]

def result {α : Type} (y : RegionName) (N pid B : Nat)
    (values : Fin 4 → TileIndex [B] → α) (s : State α) : State α :=
  field y N pid B 3 (values 3)
    (field y N pid B 2 (values 2)
      (field y N pid B 1 (values 1) (field y N pid B 0 (values 0) s)))

def fieldName (f : Fin 4) : RegName := ["r0","r1","r2","r3"][f]
def offsetName (f : Fin 4) : RegName := ["off_0","off_1","off_2","off_3"][f]

theorem exec {α : Type} [Inhabited α] (M : Algebra α) (y : RegionName)
    (N pid B : Nat) (values : Fin 4 → Values α .real [B]) (s : State α)
    (hv : ∀ f,s.regs .real [B] (fieldName f) = some (values f))
    (ho : ∀ f,s.regs .int [B] (offsetName f) = some (offsets pid B f.val)) :
    run M (body y N B) s = some (result y N pid B
      (fun f i => M.cast (some .fp32) .real .real (values f i)) s) := by
  let v := fun f i => M.cast (some .fp32) .real .real (values f i)
  let s0 := field y N pid B 0 (v 0) s
  let s1 := field y N pid B 1 (v 1) s0
  let s2 := field y N pid B 2 (v 2) s1
  exact TypedAffine.four_steps M
    (step_exec M y N pid B 0 "r0" "off_0" (values 0) s (hv 0) (ho 0))
    (step_exec M y N pid B 1 "r1" "off_1" (values 1) s0
      (by simpa [s0,fieldName] using hv 1) (by simpa [s0,offsetName] using ho 1))
    (step_exec M y N pid B 2 "r2" "off_2" (values 2) s1
      (by simpa [s1,s0,fieldName] using hv 2) (by simpa [s1,s0,offsetName] using ho 2))
    (step_exec M y N pid B 3 "r3" "off_3" (values 3) s2
      (by simpa [s2,s1,s0,fieldName] using hv 3) (by simpa [s2,s1,s0,offsetName] using ho 3))

theorem source_slice (swapped : Bool) (y : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : Float32Bits) (B : Nat) (hB : Random.ArangeSizeDomain B) :
    (KernelSource.source swapped y N seed offset lower upper B hB).surfaceBody.drop 23 = body y N B := by
  cases swapped <;> rfl

def Outside (y : RegionName) (N pid B : Nat) (r : RegionName) (a : Nat) : Prop :=
  r ≠ y ∨ ∀ f : Fin 4,∀ i : Fin B,output pid B f.val i.val < N → a ≠ output pid B f.val i.val

theorem field_frame {α : Type} (y : RegionName) (N pid B : Nat) (f : Fin 4)
    (values : TileIndex [B] → α) (s : State α) (r : RegionName) (a : Nat)
    (outside : r ≠ y ∨ ∀ i : Fin B,output pid B f.val i.val < N → a ≠ output pid B f.val i.val) :
    (field y N pid B f values s).mem r a = s.mem r a := by
  apply State.masked_scatter_frame
  rcases outside with hr | ha
  · exact Or.inl hr
  · exact Or.inr (fun i _ hi => ha i.1 hi)

theorem frame {α : Type} (y : RegionName) (N pid B : Nat)
    (values : Fin 4 → TileIndex [B] → α) (s : State α) (r : RegionName) (a : Nat)
    (outside : Outside y N pid B r a) :
    (result y N pid B values s).mem r a = s.mem r a := by
  have h (f : Fin 4) : r ≠ y ∨ ∀ i : Fin B,output pid B f.val i.val < N → a ≠ output pid B f.val i.val := by
    rcases outside with h | h
    · exact Or.inl h
    · exact Or.inr (h f)
  simp only [result,field_frame y N pid B _ _ _ r a (h _)]

theorem field_readback {α : Type} (y : RegionName) (N pid B : Nat) (f : Fin 4)
    (values : TileIndex [B] → α) (s : State α) (i : Fin B)
    (active : output pid B f.val i.val < N) :
    (field y N pid B f values s).mem y (output pid B f.val i.val) =
      .mk .real (values (i,PUnit.unit)) := by
  unfold field
  apply State.masked_scatter_readback (shape := [B]) y
    (fun j => output pid B f.val j.1.val) (fun j => Cell.mk .real (values j))
    (fun j => output pid B f.val j.1.val < N) s (i,PUnit.unit) active
  intro a b h
  apply Prod.ext
  · apply Fin.ext
    simpa [output] using h
  · exact Subsingleton.elim _ _

theorem other_field {α : Type} (y : RegionName) (N pid B : Nat) (hB : 0<B)
    (f g : Fin 4) (hne : f ≠ g) (values : TileIndex [B] → α) (s : State α) (i : Fin B) :
    (field y N pid B g values s).mem y (output pid B f.val i.val) =
      s.mem y (output pid B f.val i.val) := by
  apply field_frame
  right
  intro j _ he
  exact hne (SourceDomain.output_unique hB f g i j he).2.1

theorem readback {α : Type} (y : RegionName) (N pid B : Nat) (hB : 0<B)
    (values : Fin 4 → TileIndex [B] → α) (s : State α) (f : Fin 4) (i : Fin B)
    (active : output pid B f.val i.val < N) :
    (result y N pid B values s).mem y (output pid B f.val i.val) = .mk .real (values f (i,PUnit.unit)) := by
  fin_cases f <;> dsimp only at active ⊢
  · erw [result,other_field y N pid B hB 0 3 (by decide) (values 3) _ i,
      other_field y N pid B hB 0 2 (by decide) (values 2) _ i,
      other_field y N pid B hB 0 1 (by decide) (values 1) _ i]
    exact field_readback y N pid B 0 _ _ i active
  · erw [result,other_field y N pid B hB 1 3 (by decide) (values 3) _ i,
      other_field y N pid B hB 1 2 (by decide) (values 2) _ i]
    exact field_readback y N pid B 1 _ _ i active
  · erw [result,other_field y N pid B hB 2 3 (by decide) (values 3) _ i]
    exact field_readback y N pid B 2 _ _ i active
  · exact field_readback y N pid B 3 _ _ i active

#axiomsClean step_exec
#axiomsClean exec
#axiomsClean source_slice
#axiomsClean frame
#axiomsClean readback
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.UniformSampling.KernelStores
