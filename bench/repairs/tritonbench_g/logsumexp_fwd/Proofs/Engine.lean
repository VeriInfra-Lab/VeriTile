import bench.repairs.tritonbench_g.logsumexp_fwd.Proofs.Source

namespace VeriTile.Bench.Repairs.LogsumexpFwd
open VeriTile.Triton FP.Structural
noncomputable section
open Classical

/-- Scalar semantics and explicit reduction engines. Native construction
below expands reductions into the actual selected trees; no reduction law
is a deferred numerical primitive. -/
structure Engine (A : Type) where
  zero : A
  negInf : A
  bits : Float32Bits → A
  multiply : A → A → A
  subtract : A → A → A
  add : A → A → A
  exponential : A → A
  logarithm : A → A
  equal : A → A → Bool
  maximum : (width : Nat) → (Fin width → A) → A
  sum : (width : Nat) → (Fin width → A) → A

def reduceWith (operation : (width : Nat) → (Fin width → A) → A)
    {shape : TileShape} (ax : Fin shape.length) (keep : Bool) (xs : Values A .real shape) :
    Values A .real (TileShape.reduceShape shape ax keep) := by
  cases keep
  · exact fun out => operation (TileShape.axisDim shape ax)
      (fun i => xs (TileShape.insertAxisIndex shape ax out i))
  · exact fun out => operation (TileShape.axisDim shape ax)
      (fun i => xs (TileShape.replaceAxisIndex shape ax out i))

def Engine.algebra (engine : Engine A) : Algebra A where
  literal := fun _ _ _ => engine.zero
  negInf := engine.negInf
  binary := fun _ _ op => match op with
    | .mul => engine.multiply | .sub => engine.subtract | .add => engine.add
    | _ => fun _ _ => engine.zero
  unary := fun _ op => match op with
    | .exp => engine.exponential | .log => engine.logarithm | _ => fun _ => engine.zero
  compareEq := fun _ _ => some engine.equal
  cast := fun _ _ _ value => value
  fromNat := fun _ _ => engine.zero
  fromInt := fun _ _ => engine.zero
  fp32Bits := engine.bits
  fp32Load := id
  reduceMax := fun _ => reduceWith engine.maximum
  reduceSum := fun _ => reduceWith engine.sum

/-- Pure ordered operation graph; input coordinates are logical row-major
snapshot positions. Padded lanes are reset after optional scaling. -/
def blockValue (engine : Engine A) (input : Nat → A) (scale : A)
    (row block D B : Nat) (hasScale : Bool) : A :=
  let values := fun i : Fin B => if block*B+i.val<D then
    let x := input (row*D+(block*B+i.val))
    if hasScale then engine.multiply x scale else x
    else engine.negInf
  let maximum := engine.maximum B values
  let shift := if engine.equal maximum (engine.bits ⟨BitVec.ofNat 32 2139095040⟩) ||
      engine.equal maximum engine.negInf then engine.zero else maximum
  engine.add (engine.logarithm (engine.sum B (fun i =>
    engine.exponential (engine.subtract (values i) shift)))) shift

theorem source_value [Inhabited A] (engine : Engine A) (input : Nat → A) (scale : A)
    (row block D B : Nat) (hasScale : Bool) :
    Source.rowValue engine.algebra engine.equal input scale row block D B hasScale=
      blockValue engine input scale row block D B hasScale := by
  cases hasScale <;> simp (config := {contextual := Bool.true}) [Source.rowValue,Source.inputAddress,Engine.algebra,reduceWith,
    blockValue,TileShape.axisDim,TileShape.insertAxisIndex]

theorem block_local (engine : Engine A) (f g : Nat → A) (scale : A)
    (row block D B : Nat) (hasScale : Bool)
    (same : ∀ i,i<D → f (row*D+i)=g (row*D+i)) :
    blockValue engine f scale row block D B hasScale=blockValue engine g scale row block D B hasScale := by
  have values : (fun i : Fin B => if block*B+i.val<D then
      if hasScale then engine.multiply (f (row*D+(block*B+i.val))) scale else f (row*D+(block*B+i.val))
      else engine.negInf)=
    (fun i : Fin B => if block*B+i.val<D then
      if hasScale then engine.multiply (g (row*D+(block*B+i.val))) scale else g (row*D+(block*B+i.val))
      else engine.negInf) := by
    funext i
    by_cases h : block*B+i.val<D
    · simp [h,same _ h]
    · simp [h]
  unfold blockValue
  rw [values]

theorem block_exec [Inhabited A] (engine : Engine A) (x z : RegionName)
    (D B : Nat) (hasScale : Bool) (scale : A) (s : State A)
    (hB : 0<B) (parameter : s.regs .real [] "scale"=some (fun _ => scale)) :
    ∃ final,FP.Structural.exec engine.algebra (rowBlock x z D B hasScale) s=some final ∧
      final.mem=(s.write z (Source.outputAddress (s.pids 0) (s.pids 1) D B)
        (.mk .real (blockValue engine (fun i => (s.mem x i).read .real) scale
          (s.pids 0) (s.pids 1) D B hasScale))).mem := by
  simpa only [source_value] using Source.row_exec engine.algebra engine.equal (fun _ _ => rfl)
    x z D B hasScale scale s hB parameter

#axiomsClean source_value
#axiomsClean block_local
#axiomsClean block_exec
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LogsumexpFwd
