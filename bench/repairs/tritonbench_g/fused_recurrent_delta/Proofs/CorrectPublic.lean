import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.HostVector
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.HostFrames
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta
noncomputable section
open scoped BigOperators
set_option maxHeartbeats 10000000
/-- Independent primal specification: output, required v mutation, and optional HT. -/
structure ForwardClaim (s : Shape) (f : Flags) (scale : ℝ) (input result : Memory ℝ) : Prop where
  output : ∀ b t v,b<s.BH → t<s.T → v<s.V →
    result ⟨.o,b,t,v⟩=Math.output scale (Real.column s f input b v) t
  mutation : ∀ b t v,b<s.BH → t<s.T → v<s.V →
    result ⟨.v,b,t,v⟩=Math.residual (Math.state (Real.column s f input b v) t)
      ((Real.column s f input b v).k t) ((Real.column s f input b v).x t)
  terminal : f.final=Bool.true → ∀ b v (i : Fin s.K),b<s.BH → v<s.V →
    result ⟨.ht,b,i.val,v⟩=Math.state (Real.column s f input b v) s.T i
/-- Independent complete VJP specification. Shared q/k and scalar beta are
summed over all value columns, not over source tiles. -/
structure BackwardClaim (s : Shape) (f : Flags) (scale : ℝ) (input result : Memory ℝ) : Prop where
  query : ∀ b t (i : Fin s.K),b<s.BH → t<s.T →
    result ⟨.retq,b,t,i.val⟩=∑ v : Fin s.V,(RealBackward.grad s f scale input b v.val t).q i
  key : ∀ b t (i : Fin s.K),b<s.BH → t<s.T →
    result ⟨.retk,b,t,i.val⟩=∑ v : Fin s.V,(RealBackward.grad s f scale input b v.val t).k i
  value : ∀ b t v,b<s.BH → t<s.T → v<s.V →
    result ⟨.retv,b,t,v⟩=(RealBackward.grad s f scale input b v t).x
  vectorBeta : f.vectorBeta=Bool.true → ∀ b t v,b<s.BH → t<s.T → v<s.V →
    result ⟨.retbeta,b,t,v⟩=(RealBackward.grad s f scale input b v t).beta
  scalarBeta : f.vectorBeta=Bool.false → ∀ b t,b<s.BH → t<s.T →
    result ⟨.retbeta,b,t,0⟩=∑ v : Fin s.V,(RealBackward.grad s f scale input b v.val t).beta
  initial : f.initialGrad=Bool.true → ∀ b v (i : Fin s.K),b<s.BH → v<s.V →
    result ⟨.dh0,b,i.val,v⟩=Math.carry scale (Real.column s f input b v) (Real.incoming input b v)
      (Real.terminal s f input b v) s.T 0 i

theorem forward_correct (s : Shape) (f : Flags) (g : Trees)
    (hk : (g.key s.BK).Legal s.BK) (ho : (g.readout s.BK).Legal s.BK) (scale : ℝ) (m : Memory ℝ) :
    ForwardClaim s f scale (prepareBeta s f Real.ops m) (forwardCall s f Real.ops g scale m) := by
  constructor
  · exact ForwardResults.output_real s f g hk ho scale _
  · exact ForwardResults.residual_real s f g hk scale _
  · intro hf b v i hb hv
    exact ForwardResults.final_real s f g hk scale _ b v i hb hv hf

theorem backward_correct (s : Shape) (f : Flags) (gf gb : Trees) (h : RealBackward.Legal s gf gb)
    (scale : ℝ) (m : Memory ℝ) :
    BackwardClaim s f scale (prepareBeta s f Real.ops m)
      (backwardCall s f Real.ops gb scale (forwardCall s f Real.ops gf scale m)) := by
  let m0:=prepareBeta s f Real.ops m
  change BackwardClaim s f scale m0 (RealBackward.finish s f gf gb scale m0)
  constructor
  · exact RealBackward.return_q s f gf gb h scale m0
  · exact RealBackward.return_k s f gf gb h scale m0
  · exact RealBackward.return_v s f gf gb h scale m0
  · intro hf b t v hb ht hv
    exact RealBackward.return_beta_vector s f gf gb h scale m0 b t v hb ht hv hf
  · intro hf b t hb ht
    exact RealBackward.return_beta_scalar s f gf gb h scale m0 b t hb ht hf
  · intro hf b v i hb hv
    exact RealBackward.return_h0 s f gf gb h scale m0 b v i hb hv hf

theorem correct (s : Shape) (f : Flags) (gf gb : Trees) (h : RealBackward.Legal s gf gb)
    (scale : ℝ) (m : Memory ℝ) :
    ForwardClaim s f scale (prepareBeta s f Real.ops m) (forwardCall s f Real.ops gf scale m) ∧
    BackwardClaim s f scale (prepareBeta s f Real.ops m)
      (backwardCall s f Real.ops gb scale (forwardCall s f Real.ops gf scale m)) ∧
    MemoryProof.Preserves (MemoryProof.ForwardCallWrites s f) m (forwardCall s f Real.ops gf scale m) ∧
    MemoryProof.Preserves (MemoryProof.BackwardWrites s f) (forwardCall s f Real.ops gf scale m)
      (backwardCall s f Real.ops gb scale (forwardCall s f Real.ops gf scale m)) :=
  ⟨forward_correct s f gf h.fkey h.freadout scale m,backward_correct s f gf gb h scale m,
    MemoryProof.forward_call_frame s f Real.ops gf scale m,MemoryProof.backward_call_frame s f Real.ops gb scale _⟩

/-- The target gradients used in BackwardClaim are actual derivatives of the
independent full sequence loss, including an arbitrary terminal cotangent. -/
theorem vjp_is_derivative {BH K V : Nat} (scale : ℝ) (a d : Math.TensorInput BH K V)
    (do_ : Fin BH → Nat → Fin V → ℝ) (dht : Fin BH → Fin V → Math.Vec K) (T : Nat) :
    HasDerivAt (fun e => Math.tensorLoss scale (Math.tensorPerturb a d e) do_ dht T)
      (Math.tensorPair (Math.tensorGradient scale a do_ dht T) d T) 0 :=
  Math.tensor_loss_derivative scale a d do_ dht T

#axiomsClean correct
#axiomsClean vjp_is_derivative
#specNonCircular ForwardClaim avoiding [forwardCall,forwardGrid,forwardLoop,forwardStep]
#specNonCircular BackwardClaim avoiding [backwardCall,backwardGrid,backwardProgram,reverseLoop,recomputeLoop,hostReduce]
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta
