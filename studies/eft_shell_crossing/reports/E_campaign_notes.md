# Report E: campaign notes

Running log with commands and times: `CAMPAIGN_LEDGER.md` in the scratch directory.

## Settings

All GPU runs used the original full-resolution settings: 256³ particles, 512³ force mesh, 200 BullFrog steps from D = 0.02 with 3LPT ICs, 4th-order deposit, 2³ sheet resampling, 8 snapshot epochs. Stage 4, testing cheaper settings and code speedups, was skipped: the GPU runs were started at the user's request before it, and about 4 hours of remaining CPU work would not have repaid half a day of tests. All 9 configs are complete: 16/16 members each, 32/32 stress measurements.

## Machine

Two unexplained reboots preceded the campaign (Oct 31 about 16:25 PDT, Nov 1 about 08:52 PST), both 7–12 minutes after two GPU queues, a tl_measure batch and run_fixedmass overlapped. Their cause is still unknown.

* **No reboot since Nov 1, 08:55,** including one GPU process running for 4.5 h, overlapped with run_fixedmass (-t 96) for about the first hour, with this campaign's σ(M) jobs (-t 32) later, and with other sessions' CPU jobs throughout (about 55 cores at one check). This was a smaller load than the crash mix, so it does not show the machine is safe at that mix.
* **GPU:** median 285 W and 86 °C when busy, peaks 300 W and 87 °C; software thermal slowdown in about two-thirds of busy samples and the software power cap in 35 %. No hardware slowdown or power brake was reported.
* **UPS:** APC Smart-UPS SRT. `battery.runtime` reads 0 or −1 throughout at 100 % charge. It carried the machine through a 20 s discharge on Nov 2, 14:25:48 PST (55.2 → 48.0 V, 98 %), with no job affected; the cause of that event is not known. So the battery carried that moderate load, with the GPU idle, and the runtime value is very likely a reading problem. A self-test could look the same. Whether it can carry the crash-time load is untested.
* **Journal:** made persistent on Nov 1 by the user, so kernel messages before any later crash would survive.

**Not checked, for the user:** UPS event log around the two crash times; card fans and airflow; `upsmon` failed its login at boot on Nov 1 (ERR ACCESS-DENIED), and that was not followed up, so shutdown on low battery may not work.

## Errors made and corrected during the campaign

| what | how it was found | effect |
|---|---|---|
| README applied the displacement C₂ to P(k): wrong sign | review of line A | published, then corrected |
| Finite-ε smearing missed; explained wrongly at first; "clean" estimator misjudged | independent audit | C₂,δ 1.97 → 1.935; one-loop agreement 10 % → 1.2 % |
| Wave × band noise, mismatched realizations in a series comparison | own check | corrected before reporting |
| tl_fit3 closure forced c_r1 = c_d1 (11.14 for a true 10) | independent audit, then synthetic test | fixed before any fit used it |
| d2 + dd1 = 0.81 ± 0.02 reported as a discrepancy | own epoch-by-epoch test | not a stable measurement; withdrawn |
| "Contraction is not the main explanation" | spherical-collapse test | withdrawn: required for fixed-radius σ |
| Axis-pairing bug in `patch_anisotropy.py`; a correct Jacobian was wrongly called buggy | curl/div test | fixed; statement retracted |
| "Response scales with added power" | full set of pairs | withdrawn: varies 3.4× |
| dc̄/dI across Λ called the clearest UV test | independent audit | withdrawn: those modes are shared, so the agreement is built in; the band fits are the test |
| Switching Pages from `main` to `gh-pages` was recommended without checking which public URLs depended on the old source; `studies/lpt_pdf/report.html` went 404 | user report | link fixed and old URL redirected |
| A crashed job went unnoticed for 2.5 h; `pkill`/`pgrep -f` matched my own shell twice; PID of a `setsid` wrapper; global-scope loop ran about 15× slower | status checks | time lost, no data lost |

## What remains

All three branches are merged into `main`; the additive conflicts were in `scripts/run_jobs.jl`, `src/ics.jl` and `src/runners.jl`. After the merge, 66 study, 32 sheet-moment and 15 fixed-mass tests pass, and both branches' ICs reproduce stored runs to 3 × 10⁻⁷. Scratch holds about 400 GB of snapshots and is worth keeping for further analysis.
