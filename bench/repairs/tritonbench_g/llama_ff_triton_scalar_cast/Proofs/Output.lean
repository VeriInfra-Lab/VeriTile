import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.Reference

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.Execution
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 6000000

def steps (p : Parameters) := (p.inner+63)/64
def outputRow {α : Type} (p : Parameters) (s : State α) (i : Fin 16) :=
  rowBlock p s*16+i.val
def outputColumn {α : Type} (p : Parameters) (s : State α) (i : Fin 16) :=
  columnBlock p s*16+i.val
def active {α : Type} (p : Parameters) (s : State α) (i : TileIndex [16,16]) :=
  outputRow p s i.1<p.rows ∧ outputColumn p s i.2.1<p.columns
def outputOffset {α : Type} (p : Parameters) (s : State α) (i : TileIndex [16,16]) :=
  p.columns*outputRow p s i.1+outputColumn p s i.2.1
def normalization {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Parameters) (s : State α) (i : Fin 16) : α :=
  a.unary (some .fp32) .rsqrt (a.binary (some .fp32) .real .add
    (a.binary (some .fp32) .real .div
      (reduceLine a (some .fp32) (fun k => moment a p s (steps p) (i,k,PUnit.unit)))
      (a.fromInt (some .fp32) p.inner)) (a.literal (some .fp32) .real p.epsilon))
def projection {α : Type} [Inhabited α] (a : Arithmetic α) (native : Bool)
    (p : Parameters) (s : State α) (weight : RegionName) (i : TileIndex [16,16]) : α :=
  a.binary (some .fp32) .real .mul (accumulation a native p s weight (steps p) i)
    (normalization a p s i.1)
def sourceOutput {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Parameters) (s : State α) (i : TileIndex [16,16]) : α :=
  let u := projection a Bool.false p s "W1" i
  let v := projection a Bool.false p s "W3" i
  a.cast none .real .fp16
    (a.binary (some .fp32) .real .mul
      (a.binary (some .fp32) .real .mul u (a.unary (some .fp32) .sigmoid u)) v)

theorem active_injective {α : Type} (p : Parameters) (s : State α)
    (i j : TileIndex [16,16]) (hi : active p s i) (hj : active p s j)
    (equal : outputOffset p s i=outputOffset p s j) : i=j := by
  have hi' := hi.2
  have hj' := hj.2
  have hn : 0<p.columns := Nat.zero_lt_of_lt hi'
  have hcol := congrArg (fun n => n%p.columns) equal
  have hrow := congrArg (fun n => n/p.columns) equal
  simp [outputOffset,Nat.add_mod,Nat.mod_eq_of_lt hi',Nat.mod_eq_of_lt hj'] at hcol
  have hd (x y : Nat) (hy : y<p.columns) : (p.columns*x+y)/p.columns=x := by
    rw [Nat.mul_add_div hn,Nat.div_eq_of_lt hy,Nat.add_zero]
  simp only [outputOffset,hd _ _ hi',hd _ _ hj'] at hrow
  rcases i with ⟨i,k,⟨⟩⟩
  rcases j with ⟨j,l,⟨⟩⟩
  simp only [outputRow,outputColumn] at hrow hcol
  have hij : i=j := Fin.ext (by omega)
  have hkl : k=l := Fin.ext (by omega)
  subst j; subst l; rfl

#axiomsClean active_injective
#specNonCircular sourceOutput avoiding [source,body,prelude,compiledBody,run,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.Execution
