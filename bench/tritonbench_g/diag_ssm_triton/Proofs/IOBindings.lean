import bench.tritonbench_g.diag_ssm_triton.Proofs.IOIndex

namespace VeriTile.Bench.Repairs.DiagSsmTriton.Interface
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def forwardComponent (complex : Bool) (p : ForwardLayout) (length : Nat)
    (v : (forwardIO complex p length).Ctx) (role : Fin 3) (c : Fin (components complex)) (k : Nat) : ℝ :=
  extend (v (pair role c)) k

def backwardComponent (complex : Bool) (p : BackwardLayout) (length : Nat)
    (v : (backwardIO complex p length).Ctx) (role : Fin 4) (c : Fin (components complex)) (k : Nat) : ℝ :=
  extend (v (pair role c)) k

theorem forward_pin (complex : Bool) (p : ForwardLayout) (length : Nat)
    (u : BlockState) (v : (forwardIO complex p length).Ctx)
    (hp : (forwardIO complex p length).RegionalInputs u v)
    (role : Fin 3) (c : Fin (components complex)) (k : Nat)
    (hk : k<forwardArity p.B length role)
    (ha : active p.batch p.dim p.B (u.pids 0) (forwardLane p.B role k)) :
    u.readMem (forwardBuffer p role)
      (forwardAddress p (u.pids 0) role k*components complex+c.val) =
      forwardComponent complex p length v role c k := by
  have hik : k<(forwardIO complex p length).iarity (pair role c) := by
    simpa only [forwardIO,divNat_pair] using hk
  have hm : (forwardIO complex p length).imask (pair role c) v (u.pids 0) (u.pids 1) (u.pids 2) ⟨k,hik⟩ := by
    simpa only [forwardIO,divNat_pair] using ha
  have h := hp (pair role c) ⟨k,hik⟩ hm
  simpa only [forwardIO,divNat_pair,modNat_pair,ChanTy.read,forwardComponent,extend,hik,hk,↓reduceDIte] using h

theorem backward_pin (complex : Bool) (p : BackwardLayout) (length : Nat)
    (u : BlockState) (v : (backwardIO complex p length).Ctx)
    (hp : (backwardIO complex p length).RegionalInputs u v)
    (role : Fin 4) (c : Fin (components complex)) (k : Nat)
    (hk : k<backwardArity p.B length role)
    (hz : role.val=0 → 0<length)
    (ha : active p.batch p.dim p.B (u.pids 0) (backwardLane p.B role k)) :
    u.readMem (backwardBuffer p role)
      (backwardAddress p (u.pids 0) role k*components complex+c.val) =
      backwardComponent complex p length v role c k := by
  have hik : k<(backwardIO complex p length).iarity (pair role c) := by
    simpa only [backwardIO,divNat_pair] using hk
  have hm : (backwardIO complex p length).imask (pair role c) v (u.pids 0) (u.pids 1) (u.pids 2) ⟨k,hik⟩ := by
    simpa only [backwardIO,divNat_pair] using And.intro hz ha
  have h := hp (pair role c) ⟨k,hik⟩ hm
  simpa only [backwardIO,divNat_pair,modNat_pair,ChanTy.read,backwardComponent,extend,hik,hk,↓reduceDIte] using h

theorem forward_component_real (p : ForwardLayout) (length : Nat)
    (v : (forwardIO Bool.false p length).Ctx) (role : Fin 3) (k : Nat) :
    forwardComponent Bool.false p length v role (0 : Fin 1) k=forwardRealInputs p length v role k := by
  fin_cases role <;> rfl

theorem backward_component_real (p : BackwardLayout) (length : Nat)
    (v : (backwardIO Bool.false p length).Ctx) (role : Fin 4) (k : Nat) :
    backwardComponent Bool.false p length v role (0 : Fin 1) k=backwardRealInputs p length v role k := by
  fin_cases role <;> rfl

theorem forward_component_complex_re (p : ForwardLayout) (length : Nat)
    (v : (forwardIO Bool.true p length).Ctx) (role : Fin 3) (k : Nat) :
    forwardComponent Bool.true p length v role (0 : Fin 2) k=(forwardComplexInputs p length v role k).re := by
  fin_cases role <;> rfl

theorem forward_component_complex_im (p : ForwardLayout) (length : Nat)
    (v : (forwardIO Bool.true p length).Ctx) (role : Fin 3) (k : Nat) :
    forwardComponent Bool.true p length v role (1 : Fin 2) k=(forwardComplexInputs p length v role k).im := by
  fin_cases role <;> rfl

theorem backward_component_complex_re (p : BackwardLayout) (length : Nat)
    (v : (backwardIO Bool.true p length).Ctx) (role : Fin 4) (k : Nat) :
    backwardComponent Bool.true p length v role (0 : Fin 2) k=(backwardComplexInputs p length v role k).re := by
  fin_cases role <;> rfl

theorem backward_component_complex_im (p : BackwardLayout) (length : Nat)
    (v : (backwardIO Bool.true p length).Ctx) (role : Fin 4) (k : Nat) :
    backwardComponent Bool.true p length v role (1 : Fin 2) k=(backwardComplexInputs p length v role k).im := by
  fin_cases role <;> rfl

end
end VeriTile.Bench.Repairs.DiagSsmTriton.Interface
