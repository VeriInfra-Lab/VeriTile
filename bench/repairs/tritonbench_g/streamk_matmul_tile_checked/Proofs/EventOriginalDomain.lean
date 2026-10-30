import bench.repairs.tritonbench_g.streamk_matmul_tile_checked.Proofs.EventImplementation

/-! Every original raw finite state is represented, with arbitrary background
outside its named arrays. No Tensor-static guard, physical A/B binding, new
alignment condition or finite-only FP32 condition is introduced. -/
namespace VeriTile.Bench.Repairs.StreamK.TileChecked.EventIO
open Whole001 Source VeriTile.Triton Concurrent
noncomputable section

def finiteRead {n : Nat} (values : Fin n → V) (fallback : Int → V) (address : Int) : V :=
  if h : 0≤address ∧ address.toNat<n then values ⟨address.toNat,h.2⟩ else fallback address

theorem finiteRead_at {n : Nat} (values : Fin n → V) (fallback : Int → V) (i : Fin n) :
    finiteRead values fallback i.val=values i := by
  simp [finiteRead,i.isLt]

def lift (p : Parameters) (s : Source.State V p exterior) (background : Concurrent.State V) : Concurrent.State V :=
  {background with
    input:=fun r a=>
      if r=0 then finiteRead s.inputA (background.input r) a
      else if r=1 then finiteRead s.inputB (background.input r) a else background.input r a
    mem:=fun r a=>
      if r=0 then finiteRead s.output (background.mem r) a
      else if r=1 then finiteRead s.scratch (background.mem r) a
      else if r=2 then finiteRead s.exterior (background.mem r) a else background.mem r a}

theorem extract_lift (p : Parameters) (s : Source.State V p exterior) (background : Concurrent.State V) :
    ClosedSource.extract p exterior (lift p s background)=s := by
  apply ClosedSource.state_ext
  · funext i; simp [ClosedSource.extract,ClosedSource.viewA,lift,finiteRead_at]
  · funext i; simp [ClosedSource.extract,ClosedSource.viewB,lift,finiteRead_at]
  · funext i; simp [ClosedSource.extract,lift,finiteRead_at]
  · funext i; simp [ClosedSource.extract,lift,finiteRead_at]
  · funext i; simp [ClosedSource.extract,lift,finiteRead_at]

def snapshot (p : Parameters) (s : Concurrent.State V) : Inputs V p where
  scalars:=fun port lane=>
    (match (scalarInputs p)[port].space with | .input=>s.input | .mem=>s.mem)
      (scalarInputs p)[port].window.region ((scalarInputs p)[port].window.address lane)
  words:=fun port=>Fin.elim0 port

theorem snapshot_binding (p : Parameters) (observe : V → O) (s : Concurrent.State V) :
    (io p observe).InputBinding (snapshot p s) s := by
  refine ⟨?_,?_⟩
  · intro port lane _; rfl
  · intro port; exact Fin.elim0 port

/-- The original initial values and arbitrary exterior can be recovered from
one state admitted by the public IO, without changing any original premise. -/
theorem original_state_admitted (p : Parameters) (observe : V → O)
    (s : Source.State V p exterior) (background : Concurrent.State V) :
    (io p observe).InputBinding (snapshot p (lift p s background)) (lift p s background) ∧
      ClosedSource.extract p exterior (lift p s background)=s :=
  ⟨snapshot_binding p observe _,extract_lift p s background⟩

#axiomsClean extract_lift
#axiomsClean original_state_admitted
end
end VeriTile.Bench.Repairs.StreamK.TileChecked.EventIO
