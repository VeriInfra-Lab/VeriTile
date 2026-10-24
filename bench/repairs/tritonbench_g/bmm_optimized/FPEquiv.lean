import bench.repairs.tritonbench_g.bmm_optimized.Contract
namespace VeriTile.Bench.Repairs.BmmOptimized
theorem correct_native (call : Host.Call) (c : Config) (p : Native.Instructions)
    (callerDevice : Nat) (initial : Native.Bytes) (x : Cell) (hx : call.shape.Active x) (i : Fin 4) :
    (Host.invoke call c p callerDevice initial).memory .output (4*call.shape.o x.batch x.row x.column+i.val)=
      Host.expected p call initial x i := Host.whole_native call c p callerDevice initial x hx i
theorem fp_equivalent (call : Host.Call) (c : Config) (actual target : Native.Instructions)
    (ports : Native.Obligations actual target) (callerDevice : Nat) (initial : Native.Bytes) :
    Host.invoke call c actual callerDevice initial=Host.invoke call c target callerDevice initial := by
  rw [Native.instructions_eq actual target ports]
#axiomsClean correct_native
#axiomsClean fp_equivalent
end VeriTile.Bench.Repairs.BmmOptimized

namespace VeriTile.Bench.Repairs.BmmOptimized.Checked
open Host Native

def runNative (p : Metadata) (c : Config) (instructions : Instructions) (device : Nat) (m : Bytes) := do
  checkInputs p
  return invoke p.call c instructions device m

specification checked_native (p : Metadata) (admitted : InputContract p) (ordinary : Ordinary p.call)
    (c : Config) (instructions : Instructions) (device : Nat) (m : Bytes) :
    runNative p c instructions device m=.ok (invoke p.call c instructions device m) ∧
      Public.NativeClaim p.call c instructions instructions device m (invoke p.call c instructions device m) ∧
      IndexABI p.call.shape := by
  refine ⟨by simp [runNative,(checkInputs_iff p).2 admitted],?_,admitted.indices⟩
  refine ⟨rfl,⟨ordinary.freshA,ordinary.freshB⟩,rfl,rfl,rfl,⟨rfl,rfl⟩,?_,?_,?_⟩
  · rw [invoke_memory,load_selected,load_prepare]
  · exact Host.whole_native p.call c instructions device m
  · exact whole_frame p.call c instructions device m

/-- Only one scalar FMA numerical family remains; lazy negation is rejected. -/
specification checked_floating (p : Metadata) (admitted : InputContract p) (ordinary : Ordinary p.call)
    (c : Config) (actual target : Instructions)
    (fma : ∀ a b carry,actual.fmaRN a b carry=target.fmaRN a b carry)
    (device : Nat) (m : Bytes) :
    Public.NativeClaim p.call c actual target device m (invoke p.call c actual device m) := by
  let h := (checked_native p admitted ordinary c actual device m).2.1
  refine { h with bytes := ?_ }
  intro x hx i
  rw [h.bytes x hx i]
  exact congrFun (chain_raw_equiv p admitted actual target fma (load m) x _) i

#axiomsClean checked_native
#axiomsClean checked_floating
end VeriTile.Bench.Repairs.BmmOptimized.Checked
