# Is `reversible=True` safe for a loop that may not be reversible?

Asked 2026-09-18 by Sungyeon Hong; measured with
`analysis/reversible_estimator.py` (numbers in
`analysis/reversible_estimator.json`). No new data and no GPU: a synthetic
chain, fitted with the pipeline's own crossing guard.

The question (TASK-76, TASK-102): `analysis/msm_pipeline.py` fits every
transition matrix with `MaximumLikelihoodMSM(reversible=True)` (lines 225 and
253) and the Bayesian posterior with `BayesianMSM(..., reversible=True)` (line
369). Detailed balance is therefore imposed by construction and never tested.
The Markov state model literature the horizon argument rests on is molecular:
molecular dynamics trajectories are Boltzmann-weighted samples of a system in
thermal equilibrium, so enforcing reversibility there is variance reduction on
a property the system actually has. A text-to-image-to-text loop has no such
guarantee, and there is a mechanism pointing the other way --- compression
pressure under a transmission bottleneck (Kirby et al. 2015, in
`research-programme.md`) is directional by construction. `escape_time_prior.py`
validated the guards on `synthetic_chain`, whose wells all exchange at the same
rate, so the assumption has never been off while the guards were being checked.
This asks what it costs when it is wrong.

METHOD. The same three-well shape as `escape_time_prior.synthetic_chain`, but
the leaving probability is split unevenly between the two destinations: a share
`f` of departures go forward around the cycle 0->1->2->0 and (1-f) go back.
`f`=0.5 is the reversible chain already in use; `f`>0.5 is a directed cycle that
breaks detailed balance while leaving the stationary distribution uniform over
the wells, so any departure from a third is estimator error rather than
structure. Flux asymmetry below is the worst well pair's
|pi_i T_ij - pi_j T_ji| / (pi_i T_ij + pi_j T_ji): 0 is detailed balance, 1 is
one-way flow. Sampled at the committed per-cell budget (40 trajectories x 100
text states, all starting in well 0, as the panel's runs all start at a
prompt), 25 draws, scored against exact mean first passage times from the true
transition matrix.

## Verdict

**The escape-time guard already covers this, and that is worth saying first.**
Every pair whose reversible estimate is badly biased is also a pair the
crossing guard marks unresolved. The mechanism is not luck: a directed cycle
starves the against-the-flow direction of crossings, and the crossing count is
exactly what the guard measures. At the strongest asymmetry the 1->0 estimate
is 2.3 times the true value --- and rests on a median of zero observed
crossings, so the pipeline refuses to report it. Every pair that passes the
guard is within 6% of exact under either estimator.

| flux asym | pair | exact | `rev=True` | `rev=False` | crossings | guard      |
| --------- | ---- | ----- | ---------- | ----------- | --------- | ---------- |
| 0.00      | 0->1 | 200   | 188 (0.94) | 191 (0.96)  | 14        | resolved   |
| 0.00      | 1->0 | 200   | 212 (1.06) | 206 (1.03)  | 3         | unresolved |
| 0.40      | 0->1 | 165   | 151 (0.92) | 158 (0.96)  | 20        | resolved   |
| 0.40      | 0->2 | 215   | 263 (1.22) | 247 (1.15)  | 7         | unresolved |
| 0.80      | 0->1 | 121   | 125 (1.04) | 122 (1.01)  | 24        | resolved   |
| 0.80      | 1->0 | 209   | 334 (1.60) | 197 (0.94)  | 1         | unresolved |
| 0.96      | 0->1 | 104   | 110 (1.06) | 95 (0.92)   | 27        | resolved   |
| 0.96      | 1->0 | 202   | 469 (2.32) | 181 (0.90)  | 0         | unresolved |

**The stationary distribution is not guarded, and that is where the assumption
bites.** There is no crossing threshold on it, so it is reported whatever the
flux asymmetry. Under `reversible=True` the error grows monotonically with
asymmetry; under `reversible=False` it stays at the sampling floor.

| flux asym | `rev=True` weights  | max error | `rev=False` weights | max error |
| --------- | ------------------- | --------- | ------------------- | --------- |
| 0.00      | 0.301 0.333 0.338   | 0.033     | 0.314 0.320 0.326   | 0.019     |
| 0.40      | 0.273 0.432 0.254   | 0.098     | 0.304 0.372 0.290   | 0.044     |
| 0.80      | 0.236 0.507 0.270   | 0.174     | 0.330 0.374 0.283   | 0.050     |
| 0.96      | 0.144 0.408 0.369   | 0.190     | 0.352 0.285 0.331   | 0.048     |

True weight is a third everywhere. At the strongest asymmetry `reversible=True`
reports one set holding 0.144 of the stationary distribution and another 0.408
--- a 2.8-fold spread between two sets that in truth hold the same share ---
while `reversible=False` returns 0.352 and 0.285 at the same budget. The
reversible error at asymmetry 0.80 is five times the sampling floor the same
estimator shows on the reversible chain, so this is the constraint and not the
budget.

**Whether this loop is irreversible is still unmeasured.** Three properties are
easy to run together and the repository has different evidence for each.
_Stationarity_ is measured: step size and distance from t0 plateau by
invocation 100--150, which is what the burn-in rests on. _Ergodicity_ the
5,000-step data actively contradicts for three of four SMC networks ---
satellite sets visited by one to four runs each, implied timescales climbing
without converging out to lag 50. _Reversibility_ is neither measured nor
implied by the other two; it is imposed by a keyword argument.

## What this suggests

- **Report the flux asymmetry per cell.** It is a diagnostic computed from the
  same count matrix the pipeline already builds, it costs nothing, and it says
  whether the constraint is doing harm. Near zero, `reversible=True` is free
  variance reduction and should stay.
- **Fit the stationary distribution over metastable sets both ways** and report
  the pair where they disagree, at least for TASK-90's data. This is
  TASK-102's first per-tier observable and RQ2's object --- "whose prior does
  the stationary distribution sample from?" is a question about that vector, so
  a systematic distortion of it is not a methodological footnote.
- **Leave the escape-time guard alone.** It handles the escape times, for a
  reason that generalises past this chain.
- **Consider whether irreversibility is a result rather than a nuisance.** A
  measured detailed-balance violation says the loop has a direction --- that the
  captioner and the generator are not inverses --- which is a claim about the
  system and not about the estimator.

## Limits

A synthetic chain with one geometry, three equal wells, fast internal mixing
and a single cycle; real metastable sets differ in size and internal relaxation
and may break detailed balance in messier ways. Estimates are at lag 1 only.
The stationary distribution is uniform by construction, which makes the error
easy to read but is not the shape of a real one, where one dominant set holds
most of the weight. Runs all start in well 0, so the non-equilibrium-start bias
already noted in `escape-time-resolvability.md` is present in both columns and
is part of why `reversible=False` is not exactly a third either. Nothing here
measures the loop's actual flux asymmetry, which needs TASK-90's data.
