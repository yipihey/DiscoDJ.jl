"""Build the GitHub Pages page of this study:  README.md → index.html (+ figures).

    python3 studies/eft_shell_crossing/build_site.py <outdir>

Called by .github/workflows/report-pages.yml, which publishes it at
https://yipihey.github.io/DiscoDJ.jl/studies/eft_shell_crossing/ .  Needs the `markdown` package.
Links to repository files (configs, scripts, results) are pointed at GitHub; figures are copied and also shown in a
gallery at the end of the page.  reports/*.md are rendered to reports/*.html.
"""
import os, re, sys, shutil, glob
import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "https://github.com/yipihey/DiscoDJ.jl/blob/main/studies/eft_shell_crossing/"
REPO_ROOT = "https://github.com/yipihey/DiscoDJ.jl/blob/main/"

CSS = """
body{font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;color:#1b1b1a;margin:0;background:#fcfcfb}
main{max-width:1080px;margin:0 auto;padding:28px 22px 80px}
h1{font-size:1.75em;line-height:1.25;margin:.2em 0 .6em} h2{margin-top:2.1em;padding-bottom:.25em;border-bottom:1px solid #e4e3df}
h3{margin-top:1.6em} a{color:#2a78d6;text-decoration:none} a:hover{text-decoration:underline}
table{border-collapse:collapse;margin:1em 0;font-size:.9em;display:block;overflow-x:auto}
th,td{border:1px solid #e4e3df;padding:5px 9px;text-align:left;vertical-align:top} th{background:#f3f2ef}
code{background:#f1f0ec;padding:1px 4px;border-radius:3px;font-size:.88em} pre{background:#f1f0ec;padding:12px;overflow-x:auto;border-radius:4px}
pre code{background:none;padding:0} blockquote{border-left:3px solid #e4e3df;margin-left:0;padding-left:14px;color:#52514e}
.gallery figure{margin:28px 0} .gallery img{max-width:100%;border:1px solid #e4e3df;background:#fff}
.gallery figcaption{font-size:.88em;color:#52514e;margin-top:4px} .meta{color:#52514e;font-size:.9em}
"""

CAPTIONS = {
    "figP8_density_response": "Eulerian density response: C₂,δ/D² against D (solid corrected, dotted raw, dashed one loop), against k_split with displacement, and density shells relative to one loop.",
    "figT1_running": "Linear coefficient against I(Λ) at D = 0.1. Those modes are shared by every spectrum, so agreement across spectra is built in.",
    "figT2_stress_time": "Linear stress coefficient against D for PM and LPT, Λ = 0.1.",
    "figT3_time_local": "Density and momentum responses against the prediction from a stress local in time.",
    "figU1_uv_response": "Suppression of variance by UV modes against R at D = 1, and against D at R = 20 h⁻¹Mpc.",
    "figM1_sigma_fixed_mass_tl_full": "σ(M): fixed radius, random and mass-drawn centres at fixed mass, Lagrangian patches, and linear.",
    "figM2_sigma_spherical_collapse_tl_full": "σ(M) relative to linear, PM against spherical collapse, with and without contraction.",
    "figM3_patch_anisotropy_tl_full": "ln V of Lagrangian patches at fixed δ_L against spherical collapse, and the scatter at fixed δ_L.",
    "figM4_lnV_pdf_tl_full": "Distribution of ln V at fixed mass, D = 1: PM, Gaussian δ_L through spherical collapse, and a plain Gaussian.",
    "figM4_lnV_pdf_tl_div": "The same in the UV-divergent spectrum.",
    "figM5_k_of_R_tl_full": "Scale mapping from Δ²(k, D) = σ²(R, D): k(R), M(k) and kR(M). Dips in kR are BAO.",
    "figM6_sigma_lpt_tl_full": "nLPT/PM for each σ(M) statistic, ±5 % band. Patch curves above the axis are 5LPT and 7LPT, dominated by inverted patches.",
    "figM6_sigma_lpt_tl_div": "The same in the UV-divergent spectrum (1, 3, 7LPT).",
    "figM7_sigma_sc_reach_tl_full": "Spherical collapse/PM for each statistic, ±5 % band.",
    "figM7_sigma_sc_reach_tl_div": "The same in the UV-divergent spectrum.",
    "figM8_sigma_reach_tl_full": "Smallest mass matched to 5 % against D, for nLPT and spherical collapse. Dotted lines are the grid floor; missing points fail at every scale.",
    "figM8_sigma_reach_tl_div": "The same in the UV-divergent spectrum.",
    "figF1_fixedmass": "Fixed-mass statistics of the fiducial pair: coherent suppression, added power against linear, and F(R, μ).",
    "fig1_power_ratio": "T(k, D) = P_UV/P_IR − 1 of the Eulerian density at several epochs, PM and 7LPT (k < k_split: identical initial modes).",
    "fig2_response_k": "Field-level response R(k)/k² of the long-mode displacement at three epochs (3LPT, 7LPT, PM) with the leading-order fits.",
    "fig3_c2_vs_time": "Central figure: C₂(D)/D² for 1, 3, 5, 7LPT, the extrapolated LPT∞ and PM, with the shell-crossed (inverted) mass fraction.",
    "fig4_lpt_convergence": "|PM − LPT_n| for n = 1, 3, 5, 7 and LPT∞ vs D, and the convergence ratio of the odd LPT series.",
    "fig5_kmax_stability": "C₂(k_max) and (C₂, C₄)(k_max) at epochs before, near and after crossing.",
    "fig6_deterministic_vs_stochastic": "Deterministic R²P_L vs stochastic (conditional) P_ε for displacement and density; P_ε ∝ k² and k⁴.",
    "fig7_pm_vs_lpt": "R(k) of PM, 4LPT, 7LPT across crossing, and the realization-level departure (R_PM − R_7LPT)/k² (flat ⇒ pure k²).",
    "fig8_density_vs_displacement": "Density vs displacement response: the density is dominated by kinematic smearing σ²_ψ/6.",
    "fig9_velocity_response": "Velocity response vs d(D C₂)/dD from the displacement (kinematic consistency).",
    "fig10_pm_convergence": "PM numerical convergence: PM − 7LPT for plain CIC at three resolutions, its Richardson limit, and the discreteness-suppressed PM.",
    "fig11_ic_controls": "IC controls: Gaussian-filter pairs (PreCrossedSheet-style) carry a large IC-level 'C₂'; the sharp split and null pair give zero.",
    "fig12_band_variants": "Band variants (amplitude ×0.7, band 0.2–0.6): the fractional departure collapses onto one function of the crossed mass.",
    "fig13_kinetic_stress": "Experiment 2: response implied by PM's own multistream velocity-dispersion stress vs the measured departure.",
    "fig14_pk_convergence": "Small-scale power-spectrum convergence of the PM runs (density and displacement).",
    "figR1_chs_fig3_comparison": "Cutoff running vs CHS (2012) Fig. 3 (reconstructed and digitized) in response units: PM, 3LPT, 4LPT, one-loop PT.",
    "figR1_chs_fig3_comparison_c2units": "The same comparison in c² units (normalisation carries f₀²).",
    "figR2_running_shape": "Normalization-free running C₂(k_split) − C₂(0.5) at D = 0.3, 0.5, 1.",
    "figR3_c2_map": "C₂,δ(k_split, D)/D² for PM and 4LPT and their difference, with the IR members' shell-crossing contours.",
    "figR4_band_contribution": "Contribution of each k-shell to C₂ and its ratio to one-loop PT vs k/k_NL (decoupling beyond the nonlinear scale).",
    "figR5_bao_consequence": "Coherent ΔP/P and stochastic P_ε/P_L at BAO wavenumbers vs k_split.",
    "figR6_counts_in_spheres": "Counts in spheres: variance decomposition, real-space EFT slope vs weighted Fourier prediction, Δδ_R vs ∇²δ_R scatter.",
    "figR7_flux_decomposition": "Mass flux across the sphere boundary: diffusion (1LPT), + infall (3LPT/7LPT), + multistream (PM).",
    "figP1_cs_vs_time": "Plane-wave (ε → 0) coefficient vs D at k_split = 0.2: LPT orders, LPT∞, PM, PM on a long background (a = 0.5, 1) and the random-long-field ensemble of §6; (b) departure PM − 7LPT and the crossed mass.",
    "figP2_cs_running": "ε → 0 coefficient vs the cutoff k_split (PM solid, one loop dashed) and the departure from the pressureless series, at D = 0.3–1.",
    "figP3_loop_ladder": "Every contribution to the k² response at k_split = 0.2, isolated by construction: one loop, two-loop and higher short×short, long×short (perturbative and beyond PT), and the beyond-PT counterterm.",
    "figP4_background_scan": "Response vs long-background power a²: one loop is independent of the long modes; PM departs from the two-loop a² scaling after D ≈ 0.6 and reaches the random-ensemble value at a = 1.",
    "figP5_second_order_kernel": "Second-order (A×B) plane-wave kernel relative to the 2LPT tree: pressureless enhancement vs PM's suppressive counterterm, mean over 70 pairs and per pair.",
    "figP6_per_wave_fits": "Per-wave −R/k² with the two- and three-term fits (k_split = 0.2, 0.5): fit adequacy, and the low-k PM floor R₀ at k_split = 0.5 (§14.2).",
    "figP7_step_convergence": "BullFrog time-step convergence of C₂ against 400 steps, for the early (D = 0.02, 3LPT) and late (D = 0.15, 7LPT) starts.",
}


def relink(html):
    def fix(m):
        url = m.group(1)
        if url.startswith(("http://", "https://", "#", "mailto:", "figures/")):
            return m.group(0)
        rep = re.fullmatch(r"reports/(\w+)\.md(#.*)?", url)
        if rep:
            return f'href="reports/{rep.group(1)}.html{rep.group(2) or ""}"'
        if url.startswith("../../"):
            return f'href="{REPO_ROOT}{url[6:]}"'
        return f'href="{REPO}{url}"'
    return re.sub(r'href="([^"]+)"', fix, html)


def report_links(html):
    """Links inside reports/*.md: other reports stay on the site, figures are siblings of reports/, anything else goes to GitHub."""
    def fix(m):
        url = m.group(1)
        if url.startswith(("http://", "https://", "#", "mailto:", "../figures/")):
            return m.group(0)
        rep = re.fullmatch(r"(\w+)\.md(#.*)?", url)
        if rep:
            return f'href="{rep.group(1)}.html{rep.group(2) or ""}"'
        if url.startswith("../"):
            return f'href="{REPO}{url[3:]}"'
        return f'href="{REPO}reports/{url}"'
    return re.sub(r'href="([^"]+)"', fix, html)


def build_reports(out):
    os.makedirs(os.path.join(out, "reports"), exist_ok=True)
    paths = sorted(glob.glob(os.path.join(HERE, "reports", "*.md")))
    for p in paths:
        md = open(p, encoding="utf-8").read()
        body = report_links(markdown.markdown(md, extensions=["tables", "fenced_code", "toc", "sane_lists"]))
        title = re.search(r"^# (.+)$", md, re.M).group(1)
        name = os.path.basename(p)[:-3]
        open(os.path.join(out, "reports", name + ".html"), "w", encoding="utf-8").write(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — DiscoDJ.jl</title>
<style>{CSS} main img{{max-width:100%;border:1px solid #e4e3df;background:#fff}}</style></head>
<body><main>
<p class="meta"><a href="../">EFT / shell-crossing study</a> · <a href="00_synthesis.html">synthesis</a> · <a href="{REPO}reports/{name}.md">source on GitHub</a></p>
{body}
</main></body></html>
""")
    print(f"{len(paths)} reports written to {out}/reports")


def main(out):
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    md = open(os.path.join(HERE, "README.md"), encoding="utf-8").read()
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "toc", "sane_lists"])
    body = relink(body)
    figs = sorted(glob.glob(os.path.join(HERE, "figures", "*.png")),
                  key=lambda p: ({"fig": 0, "figR": 1, "figP": 2}[re.match(r"fig[RP]?", os.path.basename(p)).group(0)],
                                 int(re.sub(r"\D", "", os.path.basename(p).split("_")[0]) or 0)))
    gallery = ['<h2 id="figures">Figures</h2><div class="gallery">']
    for p in figs:
        name = os.path.basename(p); stem = name[:-4]
        shutil.copy(p, os.path.join(out, "figures", name))
        pdf = f"{REPO}figures/{stem}.pdf"
        gallery.append(f'<figure id="{stem}"><a href="figures/{name}"><img src="figures/{name}" alt="{stem}" loading="lazy"></a>'
                       f'<figcaption><b>{stem}</b> — {CAPTIONS.get(stem, "")} (<a href="{pdf}">PDF</a>)</figcaption></figure>')
    gallery.append("</div>")
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EFT / shell-crossing study — DiscoDJ.jl</title>
<style>{CSS}</style></head>
<body><main>
<p class="meta"><a href="../../">DiscoDJ.jl study reports</a> · <a href="{REPO}">source on GitHub</a> ·
<a href="#figures">all figures</a></p>
{body}
{"".join(gallery)}
</main></body></html>
"""
    open(os.path.join(out, "index.html"), "w", encoding="utf-8").write(html)
    print(f"site page written to {out}/index.html ({len(figs)} figures)")
    build_reports(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "_site/studies/eft_shell_crossing")
