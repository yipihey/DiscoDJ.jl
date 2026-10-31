"""Build the GitHub Pages page of this study:  README.md → index.html (+ figures).

    python3 studies/eft_shell_crossing/build_site.py <outdir>

Called by .github/workflows/report-pages.yml, which publishes it at
https://yipihey.github.io/DiscoDJ.jl/studies/eft_shell_crossing/ .  Needs the `markdown` package.
Links to repository files (configs, scripts, results) are pointed at GitHub; figures are copied and also shown in a
gallery at the end of the page.
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
}


def relink(html):
    def fix(m):
        url = m.group(1)
        if url.startswith(("http://", "https://", "#", "mailto:", "figures/")):
            return m.group(0)
        if url.startswith("../../"):
            return f'href="{REPO_ROOT}{url[6:]}"'
        return f'href="{REPO}{url}"'
    return re.sub(r'href="([^"]+)"', fix, html)


def main(out):
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    md = open(os.path.join(HERE, "README.md"), encoding="utf-8").read()
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "toc", "sane_lists"])
    body = relink(body)
    figs = sorted(glob.glob(os.path.join(HERE, "figures", "*.png")),
                  key=lambda p: (os.path.basename(p).startswith("figR"),
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


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "_site/studies/eft_shell_crossing")
