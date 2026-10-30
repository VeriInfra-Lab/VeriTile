import bench.tritonbench_g.index_select_bwd.Proofs.Native

namespace VeriTile.Bench.Repairs.IndexSelectBwd.KernelProof
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000

/-- Selected primitive semantics: this source executes one FP32 add per active
lane. Other operations retain the supplied backend, without numerical laws. -/
def arithmetic {A : Type} (rest : Algebra A) (add : A → A → A) : Algebra A :=
  {rest with binary := fun p d op => match p,d,op with
    | some .fp32,.real,.add => add
    | _,_,_ => rest.binary p d op}

def InputValues {A : Type} [Inhabited A] (p : Launch) (s : State A)
    (current incoming : Nat → A) : Prop :=
  ∀ i,i<p.count → (s.mem p.destination i).read .real=current i ∧
    (s.mem p.incoming i).read .real=incoming i

/-- A direct pointwise specification of the entire kernel memory. It uses no
executor, store fold, source scalar helper, or final-state observation. -/
def reference {A : Type} (add : A → A → A) (p : Launch) (s : State A)
    (current incoming : Nat → A) (region : RegionName) (offset : Nat) : Cell A :=
  if region=p.destination ∧ offset<p.count then
    .mk .real (sliceTarget add current incoming offset)
  else s.mem region offset

/-- Successful complete execution, independent output formula, full memory
equality, and exterior frame for the actual unit-stride slice kernel. -/
theorem successful {A : Type} [Inhabited A] (rest : Algebra A) (add : A → A → A)
    (p : Launch) (s : State A) (current incoming : Nat → A)
    (stride : p.stride=1) (count : p.count≤p.block)
    (inputs : InputValues p s current incoming) :
    ∃ t,exec (Source.backend (arithmetic rest add)) (kernel p) s=some t ∧
      t.mem=reference add p s current incoming ∧
      (∀ i,i<p.count → t.mem p.destination i=.mk .real (sliceTarget add current incoming i)) ∧
      (∀ region offset,region≠p.destination ∨ p.count≤offset →
        t.mem region offset=s.mem region offset) := by
  let t := (Source.execution (arithmetic rest add) p s).val
  have values : ∀ i,i<p.count → t.mem p.destination i=
      .mk .real (sliceTarget add current incoming i) := by
    intro i hi
    let lane : TileIndex [p.block] := (⟨i,lt_of_lt_of_le hi count⟩,PUnit.unit)
    have value := Source.readback (arithmetic rest add) p s stride lane hi
    change t.mem p.destination i=_ at value
    rw [value,(inputs i hi).1,(inputs i hi).2]
    rfl
  have frame : ∀ region offset,region≠p.destination ∨ p.count≤offset →
      t.mem region offset=s.mem region offset := by
    intro region offset outside
    apply Source.frame
    rcases outside with different | outside
    · exact Or.inl different
    · right
      intro lane active
      simpa only [stride,Nat.mul_one] using (show offset≠lane.1.val by omega)
  refine ⟨t,(Source.execution (arithmetic rest add) p s).property,?_,values,frame⟩
  funext region offset
  by_cases active : region=p.destination ∧ offset<p.count
  · rw [reference,if_pos active,active.1,values offset active.2]
  · rw [reference,if_neg active]
    apply frame
    by_cases same : region=p.destination
    · right
      have notlt : ¬offset<p.count := fun h => active ⟨same,h⟩
      omega
    · exact Or.inl same

theorem argument_checks_iff (a : Arguments) :
    checkArguments a=Bool.true ↔ AdmittedArguments a := by
  cases a with
  | tensors r => exact Native.checkInputs_iff r
  | other => simp [checkArguments,AdmittedArguments]

#specNonCircular sliceTarget avoiding [kernel,Source.scalar,Source.execution]
#specNonCircular reference avoiding [kernel,Source.expected,Source.execution]
#specNonCircular gradientTarget avoiding [kernel,Program.run,Source.execution]
#specNonCircular orderedGradientTarget avoiding [kernel,Program.run,Source.execution]
#axiomsClean successful
#axiomsClean argument_checks_iff
end
end VeriTile.Bench.Repairs.IndexSelectBwd.KernelProof
