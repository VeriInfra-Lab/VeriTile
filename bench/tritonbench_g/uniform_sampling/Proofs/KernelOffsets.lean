import bench.tritonbench_g.uniform_sampling.IO
import bench.tritonbench_g.uniform_sampling.Proofs.KernelAffine

namespace VeriTile.Bench.Repairs.UniformSampling.KernelOffsets
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.UniformSampling
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

def offsets (pid B f : Nat) : TileIndex [B] → Int := fun i => (output pid B f i.1.val : Int)

def firstOp (B : Nat) : Op .int [B] :=
  .add .int .scalarL (.mul .int .nil (.mul .int .nil (.ref .int [] "pid")
    (.castNatToInt (.ref .nat [] "BLOCK_arg"))) (.constInt 4)) (.ref .int [B] "lane")

def bumpOp (B : Nat) (name : RegName) : Op .int [B] :=
  .add .int .scalarR (.ref .int [B] name) (.castNatToInt (.ref .nat [] "BLOCK_arg"))

def body (B : Nat) : List ComputeStmt := [
  .assign .int [B] "off_0" (.alg (firstOp B)),
  .assign .int [B] "off_1" (.alg (bumpOp B "off_0")),
  .assign .int [B] "off_2" (.alg (bumpOp B "off_1")),
  .assign .int [B] "off_3" (.alg (bumpOp B "off_2"))]

def result {α : Type} (pid B : Nat) (s : State α) : State α :=
  let s := s.setReg "off_0" .int [B] (offsets pid B 0)
  let s := s.setReg "off_1" .int [B] (offsets pid B 1)
  let s := s.setReg "off_2" .int [B] (offsets pid B 2)
  s.setReg "off_3" .int [B] (offsets pid B 3)

theorem first_eval {α : Type} [Inhabited α] (M : Algebra α) (pid B : Nat) (s : State α)
    (hb : s.regs .nat [] "BLOCK_arg" = some (fun _ => B))
    (hp : s.regs .int [] "pid" = some (fun _ => (pid : Int)))
    (hl : s.regs .int [B] "lane" = some (fun i => (i.1.val : Int))) :
    evalOp M none (firstOp B) s = some (offsets pid B 0) := by
  simp [firstOp,evalOp_unfold,hb,hp,hl,numeric]
  funext i
  simp only [offsets,output,bop,Broadcast.leftIndex,Broadcast.rightIndex,Nat.cast_add,Nat.cast_mul,Nat.cast_ofNat]
  ring

theorem bump_eval {α : Type} [Inhabited α] (M : Algebra α) (pid B f : Nat)
    (name : RegName) (s : State α)
    (hb : s.regs .nat [] "BLOCK_arg" = some (fun _ => B))
    (ho : s.regs .int [B] name = some (offsets pid B f)) :
    evalOp M none (bumpOp B name) s = some (offsets pid B (f+1)) := by
  simp [bumpOp,evalOp_unfold,hb,ho,numeric]
  funext i
  simp only [offsets,output,bop,Broadcast.leftIndex,Broadcast.rightIndex,Nat.cast_add,Nat.cast_mul,Nat.cast_ofNat]
  ring

theorem exec {α : Type} [Inhabited α] (M : Algebra α) (pid B : Nat) (s : State α)
    (hb : s.regs .nat [] "BLOCK_arg" = some (fun _ => B))
    (hp : s.regs .int [] "pid" = some (fun _ => (pid : Int)))
    (hl : s.regs .int [B] "lane" = some (fun i => (i.1.val : Int))) :
    run M (body B) s = some (result pid B s) := by
  let s0 := s.setReg "off_0" .int [B] (offsets pid B 0)
  let s1 := s0.setReg "off_1" .int [B] (offsets pid B 1)
  let s2 := s1.setReg "off_2" .int [B] (offsets pid B 2)
  exact TypedAffine.four_steps M
    (TypedBindings.assign_step_of_eval M "off_0" _ _ _ (first_eval M pid B s hb hp hl))
    (TypedBindings.assign_step_of_eval M "off_1" _ _ _ (bump_eval M pid B 0 "off_0" s0
      (by simpa [s0] using hb) (by simp [s0])))
    (TypedBindings.assign_step_of_eval M "off_2" _ _ _ (bump_eval M pid B 1 "off_1" s1
      (by simpa [s1,s0] using hb) (by simp [s1])))
    (TypedBindings.assign_step_of_eval M "off_3" _ _ _ (bump_eval M pid B 2 "off_2" s2
      (by simpa [s2,s1,s0] using hb) (by simp [s2])))

theorem source_slice (swapped : Bool) (y : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : Float32Bits) (B : Nat) (hB : Random.ArangeSizeDomain B) :
    ((KernelSource.source swapped y N seed offset lower upper B hB).surfaceBody.drop 19).take 4 = body B := by
  cases swapped <;> rfl

#axiomsClean first_eval
#axiomsClean bump_eval
#axiomsClean exec
#axiomsClean source_slice
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.UniformSampling.KernelOffsets
