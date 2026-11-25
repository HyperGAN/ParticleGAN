/* ParticleGAN Research Portal — Interactive Leaderboard, Energy Pareto,
 * Evolving Laws Matrix & Live-Streamed Hugging Face Training GIF Comparison.
 * Zero external JS dependencies; works in both light and dark Material themes.
 */
((globalScope) => {
  "use strict";

  const DEFAULT_HF_REPO = "HyperGAN/particlegan-forge-evidence";
  const GITHUB_REPO = "HyperGAN/ParticleGAN";
  const DEFAULT_BRANCH = "develop";

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function toNum(value) {
    if (value == null || value === "" || typeof value === "boolean") return null;
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  }

  function formatNumber(value, digits = 2, suffix = "") {
    const n = toNum(value);
    if (n === null) return "—";
    const text =
      Math.abs(n) >= 1000
        ? n.toLocaleString("en-US", {
            minimumFractionDigits: digits,
            maximumFractionDigits: digits,
          })
        : n.toFixed(digits);
    return suffix ? `${text}${suffix}` : text;
  }

  function formatJoules(value) {
    const n = toNum(value);
    if (n === null) return "—";
    if (n >= 1000) {
      return `${n.toLocaleString("en-US", { maximumFractionDigits: 0 })} J`;
    }
    return `${n.toFixed(1)} J`;
  }

  function shortHash(value, len = 10) {
    if (!value) return "—";
    const s = String(value).trim();
    return s.length > len ? s.slice(0, len) : s;
  }

  function hfGifUrl(repo, rev, relpath) {
    if (!relpath) return null;
    if (/^https?:\/\//i.test(relpath)) return relpath;
    const cleaned = String(relpath).replace(/^\/+/, "");
    const safeRepo = repo || DEFAULT_HF_REPO;
    const safeRev = rev || "main";
    return `https://huggingface.co/datasets/${safeRepo}/resolve/${encodeURIComponent(safeRev)}/${cleaned}`;
  }

  function detectSiteRootUrl() {
    if (typeof document === "undefined") return "./";
    const scripts = Array.from(document.querySelectorAll("script[src]"));
    for (const script of scripts) {
      const src = script.getAttribute("src") || "";
      const marker = "assets/javascripts/portal-leaderboard.js";
      const idx = src.indexOf(marker);
      if (idx !== -1) {
        return src.slice(0, idx) || "./";
      }
    }
    return "../";
  }

  function statusBadgeClass(status) {
    const norm = String(status || "").trim().toLowerCase();
    if (norm === "pass" || norm === "passed" || norm === "addressed") return "pg-badge-pass";
    if (norm === "fail" || norm === "failed" || norm === "unaddressed") return "pg-badge-fail";
    if (norm === "partial" || norm === "blocked" || norm === "stale") return "pg-badge-warn";
    if (norm === "required" || norm === "active") return "pg-badge-info";
    return "pg-badge-muted";
  }

  function sortFamilies(families, sortKey, sortDir) {
    const dir = sortDir === "asc" ? 1 : -1;
    const getter = {
      label: (f) => String(f.label || f.id || "").toLowerCase(),
      full_pass: (f) => toNum(f.full_suite?.pass) ?? -1,
      full_fail: (f) => toNum(f.full_suite?.fail) ?? 1e9,
      tier1: (f) => toNum(f.tiers?.["1"]?.pass) ?? -1,
      tier2: (f) => toNum(f.tiers?.["2"]?.pass) ?? -1,
      total_view: (f) => toNum(f.total_view_requirements?.pass) ?? -1,
      stretch: (f) => toNum(f.stretch?.total?.pass) ?? -1,
      r256_ms: (f) => toNum(f.scaling_r256_ms ?? f.r256_ms_per_step) ?? 1e12,
    }[sortKey] || ((f) => toNum(f.full_suite?.pass) ?? -1);

    return [...families].sort((a, b) => {
      const av = getter(a);
      const bv = getter(b);
      if (typeof av === "string" || typeof bv === "string") {
        const cmp = String(av).localeCompare(String(bv));
        if (cmp !== 0) return cmp * dir;
      } else if (av !== bv) {
        return (av < bv ? -1 : 1) * dir;
      }
      const pCmp = (toNum(b.full_suite?.pass) ?? 0) - (toNum(a.full_suite?.pass) ?? 0);
      if (pCmp !== 0) return pCmp;
      return String(a.label || a.id).localeCompare(String(b.label || b.id));
    });
  }

  function computeParetoFrontier(points) {
    // Minimize x (energy/latency), maximize y (pass rate / passes).
    const valid = points
      .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y))
      .sort((a, b) => (a.x !== b.x ? a.x - b.x : b.y - a.y));
    const frontier = [];
    let bestY = -Infinity;
    for (const pt of valid) {
      if (pt.y > bestY) {
        frontier.push(pt);
        bestY = pt.y;
      }
    }
    return frontier;
  }

  function cohortCategory(row) {
    if (row.contended === true) return "contended";
    const gpu = String(row.hardware_gpu || "").toLowerCase();
    if (gpu.includes("a6000")) return "idle_a6000";
    if (gpu) return "idle_other";
    return "unspecified";
  }

  function cohortColor(cat) {
    if (cat === "idle_a6000") return "#4f46e5";
    if (cat === "idle_other") return "#0891b2";
    if (cat === "contended") return "#d97706";
    return "#64748b";
  }

  function cohortLabel(cat) {
    if (cat === "idle_a6000") return "Idle RTX A6000";
    if (cat === "idle_other") return "Idle (Other GPU)";
    if (cat === "contended") return "Contended GPU";
    return "Unspecified Cohort";
  }

  function createPortalApp(rootEl) {
    const siteRoot = detectSiteRootUrl();
    const dataUrl =
      rootEl.getAttribute("data-json-url") || `${siteRoot}assets/data/leaderboard.json`;

    const state = {
      data: null,
      filterMode: "all",
      search: "",
      sortKey: "full_pass",
      sortDir: "desc",
      expandedFamilyId: null,
      paretoX: "r256_joules_per_step",
      paretoIdleOnly: true,
      selectedEnergyIdx: null,
      selectedLawCell: null,
      gifTask: "",
      gifFamilyA: "",
      gifFamilyB: "",
    };

    function familyPageHref(fam) {
      const id = String(fam.id || "").trim();
      return `${siteRoot}families/${encodeURIComponent(id)}/`;
    }

    function lawDocHref(anchor) {
      const frag = anchor ? `#${encodeURIComponent(anchor)}` : "";
      return `${siteRoot}laws/four-laws/${frag}`;
    }

    function renderKpis(data) {
      const s = data.summary || {};
      const commit = shortHash(data.git_commit, 9);
      const genDate = (data.generated_at || "").slice(0, 10) || "—";
      const lead = [...(data.families || [])].sort(
        (a, b) => (b.full_suite?.pass || 0) - (a.full_suite?.pass || 0)
      )[0];
      const leadText = lead
        ? `${lead.label}: ${lead.full_suite?.pass || 0}/${lead.full_suite?.total || s.full_suite_tests || 0}`
        : "—";

      return `
        <div class="portal-kpi-grid">
          <div class="portal-kpi">
            <div class="portal-kpi-label">Selected Families</div>
            <div class="portal-kpi-val">${escapeHtml(s.selected_families ?? (data.families || []).length)}</div>
            <div class="portal-kpi-sub">Screening leader: ${escapeHtml(leadText)}</div>
          </div>
          <div class="portal-kpi">
            <div class="portal-kpi-label">Full-Suite Denominator</div>
            <div class="portal-kpi-val">${escapeHtml(s.full_suite_tests ?? data.full_suite_list?.count ?? "—")}</div>
            <div class="portal-kpi-sub">Fixed list (${escapeHtml(data.full_suite_list?.dated || "pinned")})</div>
          </div>
          <div class="portal-kpi">
            <div class="portal-kpi-label">View Requirements</div>
            <div class="portal-kpi-val">${escapeHtml(s.counted_view_requirements ?? "—")}</div>
            <div class="portal-kpi-sub">${escapeHtml(s.distinct_required_tests ?? "—")} distinct required tests</div>
          </div>
          <div class="portal-kpi">
            <div class="portal-kpi-label">NVML Energy Rows</div>
            <div class="portal-kpi-val">${escapeHtml(s.energy_benchmark_rows ?? (data.energy_benchmarks || []).length)}</div>
            <div class="portal-kpi-sub">Matched-cohort GPU telemetry</div>
          </div>
          <div class="portal-kpi">
            <div class="portal-kpi-label">Snapshot Provenance</div>
            <div class="portal-kpi-val"><code>${escapeHtml(commit)}</code></div>
            <div class="portal-kpi-sub">Built ${escapeHtml(genDate)} · Calibration BLOCKED</div>
          </div>
        </div>
      `;
    }

    function renderStackBar(fs) {
      const total = Math.max(1, toNum(fs?.total) || 1);
      const pPct = ((toNum(fs?.pass) || 0) / total) * 100;
      const fPct = ((toNum(fs?.fail) || 0) / total) * 100;
      const oPct = ((toNum(fs?.other) || 0) / total) * 100;
      const uPct = Math.max(0, 100 - pPct - fPct - oPct);
      return `
        <div class="pg-stack-bar" title="PASS ${fs?.pass || 0} · FAIL ${fs?.fail || 0} · Other ${fs?.other || 0} · Not measured ${fs?.not_measured || 0}">
          <span class="pg-stack-pass" style="width:${pPct.toFixed(1)}%"></span>
          <span class="pg-stack-fail" style="width:${fPct.toFixed(1)}%"></span>
          <span class="pg-stack-other" style="width:${oPct.toFixed(1)}%"></span>
          <span class="pg-stack-unmeasured" style="width:${uPct.toFixed(1)}%"></span>
        </div>
      `;
    }

    function renderLeaderboardPanel(data) {
      const allFamilies = data.families || [];
      const q = state.search.trim().toLowerCase();
      const filtered = allFamilies.filter((f) => {
        if (state.filterMode === "active" && f.stale) return false;
        if (state.filterMode === "stale" && !f.stale) return false;
        if (q) {
          const hay = `${f.id} ${f.label} ${f.selected_recipe || ""} ${f.candidate_id || ""}`.toLowerCase();
          if (!hay.includes(q)) return false;
        }
        return true;
      });
      const sorted = sortFamilies(filtered, state.sortKey, state.sortDir);

      const sortArrow = (key) =>
        state.sortKey === key ? (state.sortDir === "asc" ? " ▲" : " ▼") : "";

      const rowsHtml = sorted
        .map((f) => {
          const fs = f.full_suite || {};
          const t1 = f.tiers?.["1"] || {};
          const t2 = f.tiers?.["2"] || {};
          const tot = f.total_view_requirements || {};
          const st = f.stretch?.total || {};
          const speedVal = toNum(f.scaling_r256_ms ?? f.r256_ms_per_step);
          const energyVal = toNum(f.r256_joules_per_step);
          const speedText =
            speedVal !== null
              ? `${formatNumber(speedVal, 1, " ms")}${energyVal !== null ? ` · ${formatNumber(energyVal, 1, " J")}` : ""}`
              : energyVal !== null
                ? formatNumber(energyVal, 1, " J/step")
                : "—";
          const expanded = state.expandedFamilyId === f.id;
          const statusBadge = f.stale
            ? `<span class="pg-badge pg-badge-warn">Stale</span>`
            : `<span class="pg-badge pg-badge-info">Current</span>`;

          const mainRow = `
            <tr class="portal-row-clickable" data-family-toggle="${escapeHtml(f.id)}">
              <td>
                <a class="portal-family-name" href="${escapeHtml(familyPageHref(f))}" onclick="event.stopPropagation()">${escapeHtml(f.label)}</a>
                <span class="portal-family-sub"><code>${escapeHtml(f.candidate_id || f.id)}</code></span>
              </td>
              <td>
                <strong>${escapeHtml(fs.pass ?? 0)} / ${escapeHtml(fs.total ?? 0)}</strong>
                ${renderStackBar(fs)}
              </td>
              <td>
                <span class="pg-badge ${(fs.fail || 0) > 0 ? "pg-badge-fail" : "pg-badge-muted"}">${escapeHtml(fs.fail ?? 0)} FAIL</span>
                <span class="portal-family-sub">${escapeHtml(fs.not_measured ?? 0)} unmeas.${(fs.other || 0) > 0 ? ` · ${escapeHtml(fs.other)} other` : ""}</span>
              </td>
              <td>${escapeHtml(t1.pass ?? 0)}/${escapeHtml(t1.total ?? 0)}</td>
              <td>${escapeHtml(t2.pass ?? 0)}/${escapeHtml(t2.total ?? 0)}</td>
              <td><strong>${escapeHtml(tot.pass ?? 0)}/${escapeHtml(tot.total ?? 0)}</strong></td>
              <td>${escapeHtml(st.pass ?? 0)}/${escapeHtml(st.total ?? 0)}</td>
              <td>${escapeHtml(speedText)}</td>
              <td>${statusBadge}</td>
            </tr>
          `;

          if (!expanded) return mainRow;
          return (
            mainRow +
            `
            <tr>
              <td colspan="9">
                <div class="portal-detail-box">
                  <div class="portal-detail-grid">
                    <div>
                      <strong>Selected Recipe:</strong> ${escapeHtml(f.selected_recipe || f.label)}<br/>
                      <strong>Recipe SHA:</strong> <code>${escapeHtml(shortHash(f.recipe_sha256, 12))}</code><br/>
                      <strong>Source Commit:</strong> <code>${escapeHtml(shortHash(f.source_origin_commit, 12))}</code>
                    </div>
                    <div>
                      <strong>Held Breakdown:</strong> ${escapeHtml(fs.held?.pass ?? 0)} PASS / ${escapeHtml(fs.held?.fail ?? 0)} FAIL / ${escapeHtml(fs.held?.not_measured ?? 0)} unmeasured (of ${escapeHtml(fs.held?.total ?? 0)})<br/>
                      <strong>Backend:</strong> ${escapeHtml(f.backend || "cuda")}<br/>
                      ${f.qualification_note ? `<strong>Policy Note:</strong> ${escapeHtml(f.qualification_note)}` : ""}
                    </div>
                    <div>
                      <strong>Tuning Budget:</strong> ${escapeHtml(f.tuning_budget || "—")}<br/>
                      ${f.holdout_claim ? `<strong>Holdout:</strong> ${escapeHtml(f.holdout_claim)}<br/>` : ""}
                      <a href="${escapeHtml(familyPageHref(f))}">Open full family report →</a>
                    </div>
                  </div>
                  ${f.overview ? `<p class="portal-muted" style="margin:0.4rem 0 0">${escapeHtml(f.overview.slice(0, 360))}${f.overview.length > 360 ? "…" : ""}</p>` : ""}
                </div>
              </td>
            </tr>
          `
          );
        })
        .join("");

      return `
        <section class="portal-panel">
          <div class="portal-panel-head">
            <div>
              <h2 class="portal-panel-title">Selected Family Screening Leaderboard</h2>
              <p class="portal-panel-sub">Click column headers to sort; click any row to inspect provenance, tuning budget, and held-task breakdown.</p>
            </div>
            <div class="portal-controls">
              <div class="portal-btn-group" role="group" aria-label="Filter families">
                <button class="portal-btn ${state.filterMode === "all" ? "is-active" : ""}" data-filter="all">All (${allFamilies.length})</button>
                <button class="portal-btn ${state.filterMode === "active" ? "is-active" : ""}" data-filter="active">Current</button>
                <button class="portal-btn ${state.filterMode === "stale" ? "is-active" : ""}" data-filter="stale">Stale</button>
              </div>
              <input class="portal-input" type="search" id="pg-lb-search" placeholder="Filter family or candidate…" value="${escapeHtml(state.search)}" />
            </div>
          </div>
          <div class="portal-table-wrap">
            <table class="portal-table">
              <thead>
                <tr>
                  <th class="is-sortable" data-sort="label">Family / Candidate${sortArrow("label")}</th>
                  <th class="is-sortable" data-sort="full_pass">Full-Suite PASS${sortArrow("full_pass")}</th>
                  <th class="is-sortable" data-sort="full_fail">FAIL &amp; Coverage${sortArrow("full_fail")}</th>
                  <th class="is-sortable" data-sort="tier1">Tier 1${sortArrow("tier1")}</th>
                  <th class="is-sortable" data-sort="tier2">Tier 2${sortArrow("tier2")}</th>
                  <th class="is-sortable" data-sort="total_view">View Total${sortArrow("total_view")}</th>
                  <th class="is-sortable" data-sort="stretch">Stretch${sortArrow("stretch")}</th>
                  <th class="is-sortable" data-sort="r256_ms">r256 Speed / Energy${sortArrow("r256_ms")}</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                ${rowsHtml || `<tr><td colspan="9" class="portal-muted">No families match the current filter.</td></tr>`}
              </tbody>
            </table>
          </div>
        </section>
      `;
    }

    function renderParetoPanel(data) {
      const benchmarks = data.energy_benchmarks || [];
      const xOptions = [
        { key: "r256_joules_per_step", label: "r256 J/step (NVML)", fmt: (v) => formatNumber(v, 2, " J/step") },
        { key: "r256_ms_per_step", label: "r256 ms/step", fmt: (v) => formatNumber(v, 2, " ms") },
        { key: "grid100_j_to_pass", label: "grid100 J-to-PASS", fmt: formatJoules },
        { key: "ring16_j_to_pass", label: "ring16 J-to-PASS", fmt: formatJoules },
      ];
      const activeOpt = xOptions.find((o) => o.key === state.paretoX) || xOptions[0];

      const points = [];
      benchmarks.forEach((row, idx) => {
        if (state.paretoIdleOnly && row.contended === true) return;
        const x = toNum(row[activeOpt.key]);
        const op = toNum(row.occam_pass);
        const oe = toNum(row.occam_executed);
        if (x === null || op === null || oe === null || oe <= 0) return;
        const y = (op / oe) * 100;
        points.push({
          idx,
          row,
          x,
          y,
          cat: cohortCategory(row),
        });
      });

      const idlePoints = points.filter((p) => p.cat === "idle_a6000");
      const frontier = computeParetoFrontier(idlePoints.length ? idlePoints : points);
      const frontierSet = new Set(frontier.map((p) => p.idx));

      if (points.length && !points.some((p) => p.idx === state.selectedEnergyIdx)) {
        const bestFrontier = frontier.length ? frontier[frontier.length - 1] : points[0];
        state.selectedEnergyIdx = bestFrontier.idx;
      }
      const selectedPoint = points.find((p) => p.idx === state.selectedEnergyIdx) || points[0] || null;

      // SVG dimensions
      const W = 680;
      const H = 360;
      const pad = { l: 58, r: 26, t: 24, b: 48 };
      const plotW = W - pad.l - pad.r;
      const plotH = H - pad.t - pad.b;

      let svgBody = `<text x="${W / 2}" y="${H / 2}" text-anchor="middle" fill="currentColor" opacity="0.6" font-size="13">No matched benchmarks with both ${escapeHtml(activeOpt.label)} and Occam pass count.</text>`;

      if (points.length) {
        const xs = points.map((p) => p.x);
        const xMinRaw = Math.min(...xs);
        const xMaxRaw = Math.max(...xs);
        const xSpan = Math.max(1e-6, xMaxRaw - xMinRaw);
        const xMin = Math.max(0, xMinRaw - xSpan * 0.08);
        const xMax = xMaxRaw + xSpan * 0.1;
        const yMin = 0;
        const yMax = 105;

        const sx = (v) => pad.l + ((v - xMin) / Math.max(1e-6, xMax - xMin)) * plotW;
        const sy = (v) => pad.t + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

        const yTicks = [0, 25, 50, 75, 100];
        const xTicks = [0, 0.25, 0.5, 0.75, 1].map((t) => xMin + t * (xMax - xMin));

        const gridLines = [
          ...yTicks.map(
            (yv) => `
              <line x1="${pad.l}" y1="${sy(yv).toFixed(1)}" x2="${W - pad.r}" y2="${sy(yv).toFixed(1)}" stroke="var(--pg-grid-line)" stroke-width="1" />
              <text x="${pad.l - 8}" y="${(sy(yv) + 4).toFixed(1)}" text-anchor="end" fill="var(--pg-text-muted)" font-size="11">${yv}%</text>
            `
          ),
          ...xTicks.map(
            (xv) => `
              <line x1="${sx(xv).toFixed(1)}" y1="${pad.t}" x2="${sx(xv).toFixed(1)}" y2="${H - pad.b}" stroke="var(--pg-grid-line)" stroke-width="1" />
              <text x="${sx(xv).toFixed(1)}" y="${H - pad.b + 18}" text-anchor="middle" fill="var(--pg-text-muted)" font-size="11">${xv >= 1000 ? Math.round(xv).toLocaleString("en-US") : xv.toFixed(1)}</text>
            `
          ),
        ].join("");

        const frontierLine =
          frontier.length > 1
            ? `<polyline fill="none" stroke="var(--pg-accent-2)" stroke-width="2.2" stroke-dasharray="6 4" points="${frontier
                .map((p) => `${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`)
                .join(" ")}" />`
            : "";

        const dots = points
          .map((p) => {
            const cx = sx(p.x).toFixed(1);
            const cy = sy(p.y).toFixed(1);
            const isSel = selectedPoint && p.idx === selectedPoint.idx;
            const isFront = frontierSet.has(p.idx);
            const col = cohortColor(p.cat);
            const label = isFront ? p.row.candidate_id : "";
            return `
              <g class="pg-pareto-pt" data-energy-idx="${p.idx}" style="cursor:pointer">
                ${isFront ? `<circle cx="${cx}" cy="${cy}" r="${isSel ? 11 : 9}" fill="none" stroke="var(--pg-accent-2)" stroke-width="1.8" opacity="0.85" />` : ""}
                <circle cx="${cx}" cy="${cy}" r="${isSel ? 6.5 : 5}" fill="${col}" stroke="var(--pg-surface)" stroke-width="1.5">
                  <title>${escapeHtml(p.row.candidate_id)}: ${escapeHtml(activeOpt.fmt(p.x))}, Occam ${escapeHtml(p.row.occam_pass)}/${escapeHtml(p.row.occam_executed)} (${p.y.toFixed(0)}%)</title>
                </circle>
                ${label ? `<text x="${(Number(cx) + 9).toFixed(1)}" y="${(Number(cy) - 6).toFixed(1)}" fill="var(--pg-text)" font-size="10" font-weight="600">${escapeHtml(label)}</text>` : ""}
              </g>
            `;
          })
          .join("");

        svgBody = `
          ${gridLines}
          <line x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${H - pad.b}" stroke="var(--pg-border-strong)" stroke-width="1.2" />
          <line x1="${pad.l}" y1="${H - pad.b}" x2="${W - pad.r}" y2="${H - pad.b}" stroke="var(--pg-border-strong)" stroke-width="1.2" />
          ${frontierLine}
          ${dots}
          <text x="${pad.l + plotW / 2}" y="${H - 8}" text-anchor="middle" fill="var(--pg-text)" font-size="11.5" font-weight="600">${escapeHtml(activeOpt.label)} (lower is better ←)</text>
          <text x="16" y="${pad.t + plotH / 2}" text-anchor="middle" fill="var(--pg-text)" font-size="11.5" font-weight="600" transform="rotate(-90 16 ${pad.t + plotH / 2})">Occam Pass Rate (higher is better ↑)</text>
        `;
      }

      const sel = selectedPoint ? selectedPoint.row : null;
      const detailCard = sel
        ? `
          <div class="portal-pareto-card">
            <div style="display:flex;justify-content:space-between;align-items:center;gap:0.5rem;margin-bottom:0.5rem">
              <strong><code>${escapeHtml(sel.candidate_id)}</code></strong>
              <span class="pg-badge ${sel.contended ? "pg-badge-warn" : "pg-badge-pass"}">${sel.contended ? "Contended" : "Idle GPU"}</span>
            </div>
            <p class="portal-muted" style="margin:0 0 0.7rem">
              Optimizer: <code>${escapeHtml(sel.optimizer_family || "—")}</code> · GPU: ${escapeHtml(sel.hardware_gpu || "—")}
            </p>
            <table class="portal-table" style="font-size:0.76rem">
              <tbody>
                <tr><td>Occam Pass Rate</td><td><strong>${escapeHtml(sel.occam_pass ?? "—")}/${escapeHtml(sel.occam_executed ?? "—")}</strong></td></tr>
                <tr><td>r256 ms/step</td><td>${escapeHtml(formatNumber(sel.r256_ms_per_step, 2, " ms"))}</td></tr>
                <tr><td>r256 Power / Energy</td><td>${escapeHtml(formatNumber(sel.r256_watts, 1, " W"))} · ${escapeHtml(formatNumber(sel.r256_joules_per_step, 2, " J"))}</td></tr>
                <tr><td>Optimizer State</td><td>${escapeHtml(formatNumber(sel.state_bytes_per_param, 3, " B/param"))}</td></tr>
                <tr><td>ring16 J-to-PASS</td><td>${escapeHtml(formatJoules(sel.ring16_j_to_pass))}${sel.ring16_first_pass_step ? ` (step ${sel.ring16_first_pass_step})` : ""}</td></tr>
                <tr><td>grid100 J-to-PASS</td><td>${escapeHtml(formatJoules(sel.grid100_j_to_pass))}${sel.grid100_first_pass_step ? ` (step ${sel.grid100_first_pass_step})` : ""}</td></tr>
                <tr><td>Source Commit</td><td><code>${escapeHtml(shortHash(sel.source_commit, 10))}</code></td></tr>
              </tbody>
            </table>
            ${
              sel.source_file
                ? `<p style="margin:0.65rem 0 0;font-size:0.75rem"><a href="https://github.com/${GITHUB_REPO}/blob/${DEFAULT_BRANCH}/${escapeHtml(sel.source_file)}" target="_blank" rel="noopener">Inspect benchmark receipt on GitHub ↗</a></p>`
                : ""
            }
          </div>
        `
        : `<div class="portal-pareto-card"><p class="portal-muted">Click any point on the Pareto plot to inspect its NVML energy telemetry.</p></div>`;

      return `
        <section class="portal-panel">
          <div class="portal-panel-head">
            <div>
              <h2 class="portal-panel-title">Hardware NVML Energy &amp; Latency Pareto Frontier</h2>
              <p class="portal-panel-sub">Top-left is optimal (higher Occam pass rate at lower GPU energy/latency). Contended runs are never pooled into the idle Pareto curve.</p>
            </div>
            <div class="portal-controls">
              <div class="portal-btn-group" role="group" aria-label="Pareto X-axis">
                ${xOptions
                  .map(
                    (o) =>
                      `<button class="portal-btn ${state.paretoX === o.key ? "is-active" : ""}" data-pareto-x="${escapeHtml(o.key)}">${escapeHtml(o.label)}</button>`
                  )
                  .join("")}
              </div>
              <button class="portal-btn ${state.paretoIdleOnly ? "is-active" : ""}" style="border:1px solid var(--pg-border);border-radius:8px" data-pareto-idle="toggle">
                ${state.paretoIdleOnly ? "Idle Cohorts Only" : "Showing Contended Too"}
              </button>
            </div>
          </div>
          <div class="portal-pareto-layout">
            <div>
              <div class="portal-svg-frame">
                <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Pareto scatter plot of Occam pass rate versus GPU energy or latency">
                  ${svgBody}
                </svg>
              </div>
              <div class="portal-legend">
                <span class="portal-legend-item"><span class="portal-dot" style="background:${cohortColor("idle_a6000")}"></span>${cohortLabel("idle_a6000")}</span>
                <span class="portal-legend-item"><span class="portal-dot" style="background:${cohortColor("idle_other")}"></span>${cohortLabel("idle_other")}</span>
                <span class="portal-legend-item"><span class="portal-dot" style="background:${cohortColor("contended")}"></span>${cohortLabel("contended")}</span>
                <span class="portal-legend-item"><span class="portal-dot" style="background:transparent;border:2px solid var(--pg-accent-2)"></span>Pareto Frontier (${frontier.length} pts)</span>
              </div>
            </div>
            ${detailCard}
          </div>
        </section>
      `;
    }

    function renderLawsMatrixPanel(data) {
      const laws = Array.isArray(data.laws) ? data.laws : [];
      const lawsMeta = data.laws_meta || {};
      const families = data.families || [];
      if (!laws.length) return "";

      const headerCols = laws
        .map(
          (law) => `
            <th>
              <a href="${escapeHtml(lawDocHref(law.doc_anchor))}">${escapeHtml(law.short_label || law.id)}</a>
              <span class="portal-family-sub">${escapeHtml(law.title || "")}</span>
            </th>
          `
        )
        .join("");

      const rows = families
        .map((fam) => {
          const mechs = fam.law_mechanisms || {};
          const cells = laws
            .map((law) => {
              const entry = mechs[law.id];
              if (entry && typeof entry === "object") {
                const st = String(entry.status || "partial");
                const cls = statusBadgeClass(st);
                return `
                  <td>
                    <button class="portal-law-cell ${cls}" data-law-fam="${escapeHtml(fam.id)}" data-law-id="${escapeHtml(law.id)}" title="${escapeHtml(entry.mechanism || st)}">
                      ${escapeHtml(st)}
                    </button>
                  </td>
                `;
              }
              return `
                <td>
                  <button class="portal-law-cell pg-badge-muted" data-law-fam="${escapeHtml(fam.id)}" data-law-id="${escapeHtml(law.id)}">
                    inspect
                  </button>
                </td>
              `;
            })
            .join("");
          return `
            <tr>
              <td><a class="portal-family-name" href="${escapeHtml(familyPageHref(fam))}">${escapeHtml(fam.label)}</a></td>
              ${cells}
            </tr>
          `;
        })
        .join("");

      let detailHtml = `<p class="portal-muted" style="margin:0">Click any cell above to inspect how that family addresses the selected law (and its known limits).</p>`;
      if (state.selectedLawCell) {
        const fam = families.find((f) => f.id === state.selectedLawCell.famId);
        const law = laws.find((l) => l.id === state.selectedLawCell.lawId);
        if (fam && law) {
          const entry = (fam.law_mechanisms || {})[law.id];
          if (entry) {
            detailHtml = `
              <div>
                <strong>${escapeHtml(fam.label)} × ${escapeHtml(law.short_label)}: ${escapeHtml(law.title)}</strong>
                <span class="pg-badge ${statusBadgeClass(entry.status)}" style="margin-left:0.5rem">${escapeHtml(entry.status)}</span>
                <p style="margin:0.45rem 0 0.3rem"><strong>Mechanism:</strong> ${escapeHtml(entry.mechanism || "—")}</p>
                <p style="margin:0 0 0.3rem"><strong>How it works:</strong> ${escapeHtml(entry.how_it_works || "—")}</p>
                <p style="margin:0"><strong>Known limits:</strong> ${escapeHtml(entry.known_limits || "—")}</p>
              </div>
            `;
          } else {
            detailHtml = `
              <div>
                <strong>${escapeHtml(fam.label)} × ${escapeHtml(law.short_label)}: ${escapeHtml(law.title)}</strong>
                <p class="portal-muted" style="margin:0.35rem 0 0.35rem"><em>Law question:</em> ${escapeHtml(law.question || "")}</p>
                <p style="margin:0 0 0.35rem">${escapeHtml((fam.overview || "See the dedicated family page for mathematical formulation and mechanism breakdown.").slice(0, 340))}</p>
                <a href="${escapeHtml(familyPageHref(fam))}">Open ${escapeHtml(fam.label)} family report →</a> ·
                <a href="${escapeHtml(lawDocHref(law.doc_anchor))}">Read ${escapeHtml(law.short_label)} formalization →</a>
              </div>
            `;
          }
        }
      }

      return `
        <section class="portal-panel">
          <div class="portal-panel-head">
            <div>
              <h2 class="portal-panel-title">Evolving Laws × Family Coverage Matrix</h2>
              <p class="portal-panel-sub">Rendered dynamically from the repository's evolving laws registry (provisional scientific framework).</p>
            </div>
            <a class="md-button" href="${escapeHtml(siteRoot)}laws/">Open Laws Hub</a>
          </div>
          ${
            lawsMeta.epistemic_note
              ? `<div class="portal-law-banner"><strong>Provisional Framework:</strong> ${escapeHtml(lawsMeta.epistemic_note)}</div>`
              : ""
          }
          <div class="portal-table-wrap">
            <table class="portal-table">
              <thead>
                <tr>
                  <th>Family</th>
                  ${headerCols}
                </tr>
              </thead>
              <tbody>${rows}</tbody>
            </table>
          </div>
          <div class="portal-law-detail">${detailHtml}</div>
        </section>
      `;
    }

    function renderGifPanel(data) {
      const tasksWithGifs = (data.tasks || []).filter((t) => t.gif_relpath);
      if (!tasksWithGifs.length) return "";

      const byTask = new Map();
      for (const row of tasksWithGifs) {
        if (!byTask.has(row.task_id)) byTask.set(row.task_id, []);
        byTask.get(row.task_id).push(row);
      }

      const taskNames = Array.from(byTask.keys()).sort((a, b) => {
        const diff = byTask.get(b).length - byTask.get(a).length;
        return diff !== 0 ? diff : a.localeCompare(b);
      });

      const preferred = ["ring16_acquisition", "grid100", "ae_gan_hold", "mode_hold"];
      if (!state.gifTask || !byTask.has(state.gifTask)) {
        state.gifTask = preferred.find((name) => byTask.has(name)) || taskNames[0];
      }

      const rowsForTask = byTask.get(state.gifTask) || [];
      const famIds = rowsForTask.map((r) => r.family_id);
      if (!state.gifFamilyA || !famIds.includes(state.gifFamilyA)) {
        state.gifFamilyA = famIds.includes("unified") ? "unified" : famIds[0] || "";
      }
      if (!state.gifFamilyB || !famIds.includes(state.gifFamilyB) || state.gifFamilyB === state.gifFamilyA) {
        state.gifFamilyB =
          famIds.find((id) => id !== state.gifFamilyA && (id === "rhbi" || id === "bispf" || id === "bcap-pure")) ||
          famIds.find((id) => id !== state.gifFamilyA) ||
          state.gifFamilyA;
      }

      const repo = data.dataset_repo || rootEl.getAttribute("data-hf-repo") || DEFAULT_HF_REPO;
      const rev = data.hf_revision || "main";

      const renderCard = (slotLabel, famId) => {
        const rec = rowsForTask.find((r) => r.family_id === famId);
        if (!rec) {
          return `<div class="portal-gif-card"><div class="portal-gif-stage"><span class="portal-muted">No training GIF recorded for this pair.</span></div></div>`;
        }
        const url = hfGifUrl(repo, rev, rec.gif_relpath);
        const metricsEntries = Object.entries(rec.metrics_summary || {})
          .filter(([, v]) => typeof v === "number" || typeof v === "string")
          .slice(0, 4)
          .map(([k, v]) => `${k}=${typeof v === "number" ? v.toFixed(3) : v}`)
          .join(" · ");

        return `
          <div class="portal-gif-card">
            <div class="portal-gif-header">
              <span>${escapeHtml(slotLabel)}: ${escapeHtml(rec.family_label || rec.family_id)}</span>
              <span class="pg-badge ${statusBadgeClass(rec.status)}">${escapeHtml(rec.status)}</span>
            </div>
            <div class="portal-gif-stage">
              <img src="${escapeHtml(url)}" alt="${escapeHtml(rec.family_label)} training animation on ${escapeHtml(rec.task_id)}" loading="lazy"
                onerror="this.replaceWith(Object.assign(document.createElement('div'),{className:'portal-muted',textContent:'Streamed from HF Dataset (${escapeHtml(repo)}). Connect to the internet to load animation.'}))" />
            </div>
            <div class="portal-gif-meta">
              Candidate: <code>${escapeHtml(rec.candidate_id || "—")}</code>
              ${rec.wall_seconds != null ? ` · Wall: ${formatNumber(rec.wall_seconds, 1, "s")}` : ""}
              ${metricsEntries ? `<br/>${escapeHtml(metricsEntries)}` : ""}
            </div>
          </div>
        `;
      };

      return `
        <section class="portal-panel">
          <div class="portal-panel-head">
            <div>
              <h2 class="portal-panel-title">Live-Streamed Training GIF Comparison</h2>
              <p class="portal-panel-sub">Streamed directly from <code>${escapeHtml(repo)}</code> — zero binary GIFs committed in Git.</p>
            </div>
            <div class="portal-controls">
              <label class="portal-muted" for="pg-gif-task">Task:</label>
              <select class="portal-select" id="pg-gif-task">
                ${taskNames
                  .map(
                    (t) =>
                      `<option value="${escapeHtml(t)}" ${t === state.gifTask ? "selected" : ""}>${escapeHtml(t)} (${byTask.get(t).length})</option>`
                  )
                  .join("")}
              </select>
              <select class="portal-select" id="pg-gif-fam-a" aria-label="Left family">
                ${rowsForTask
                  .map(
                    (r) =>
                      `<option value="${escapeHtml(r.family_id)}" ${r.family_id === state.gifFamilyA ? "selected" : ""}>A: ${escapeHtml(r.family_label)} (${escapeHtml(r.status)})</option>`
                  )
                  .join("")}
              </select>
              <select class="portal-select" id="pg-gif-fam-b" aria-label="Right family">
                ${rowsForTask
                  .map(
                    (r) =>
                      `<option value="${escapeHtml(r.family_id)}" ${r.family_id === state.gifFamilyB ? "selected" : ""}>B: ${escapeHtml(r.family_label)} (${escapeHtml(r.status)})</option>`
                  )
                  .join("")}
              </select>
            </div>
          </div>
          <div class="portal-gif-grid">
            ${renderCard("Family A", state.gifFamilyA)}
            ${renderCard("Family B", state.gifFamilyB)}
          </div>
        </section>
      `;
    }

    function bindEvents() {
      rootEl.querySelectorAll("[data-filter]").forEach((btn) => {
        btn.addEventListener("click", () => {
          state.filterMode = btn.getAttribute("data-filter") || "all";
          render();
        });
      });

      const searchEl = rootEl.querySelector("#pg-lb-search");
      if (searchEl) {
        searchEl.addEventListener("input", (e) => {
          state.search = e.target.value || "";
          render();
          const next = rootEl.querySelector("#pg-lb-search");
          if (next) {
            next.focus();
            next.setSelectionRange(next.value.length, next.value.length);
          }
        });
      }

      rootEl.querySelectorAll("[data-sort]").forEach((th) => {
        th.addEventListener("click", () => {
          const key = th.getAttribute("data-sort");
          if (state.sortKey === key) {
            state.sortDir = state.sortDir === "desc" ? "asc" : "desc";
          } else {
            state.sortKey = key;
            state.sortDir = key === "label" || key === "full_fail" || key === "r256_ms" ? "asc" : "desc";
          }
          render();
        });
      });

      rootEl.querySelectorAll("[data-family-toggle]").forEach((tr) => {
        tr.addEventListener("click", () => {
          const fid = tr.getAttribute("data-family-toggle");
          state.expandedFamilyId = state.expandedFamilyId === fid ? null : fid;
          render();
        });
      });

      rootEl.querySelectorAll("[data-pareto-x]").forEach((btn) => {
        btn.addEventListener("click", () => {
          state.paretoX = btn.getAttribute("data-pareto-x") || "r256_joules_per_step";
          render();
        });
      });

      const idleBtn = rootEl.querySelector("[data-pareto-idle]");
      if (idleBtn) {
        idleBtn.addEventListener("click", () => {
          state.paretoIdleOnly = !state.paretoIdleOnly;
          render();
        });
      }

      rootEl.querySelectorAll("[data-energy-idx]").forEach((pt) => {
        pt.addEventListener("click", () => {
          state.selectedEnergyIdx = Number(pt.getAttribute("data-energy-idx"));
          render();
        });
      });

      rootEl.querySelectorAll("[data-law-fam]").forEach((btn) => {
        btn.addEventListener("click", () => {
          state.selectedLawCell = {
            famId: btn.getAttribute("data-law-fam"),
            lawId: btn.getAttribute("data-law-id"),
          };
          render();
        });
      });

      const gifTaskEl = rootEl.querySelector("#pg-gif-task");
      if (gifTaskEl) {
        gifTaskEl.addEventListener("change", (e) => {
          state.gifTask = e.target.value;
          render();
        });
      }
      const gifAEl = rootEl.querySelector("#pg-gif-fam-a");
      if (gifAEl) {
        gifAEl.addEventListener("change", (e) => {
          state.gifFamilyA = e.target.value;
          render();
        });
      }
      const gifBEl = rootEl.querySelector("#pg-gif-fam-b");
      if (gifBEl) {
        gifBEl.addEventListener("change", (e) => {
          state.gifFamilyB = e.target.value;
          render();
        });
      }
    }

    function renderFourLawsAdherencePanel(data) {
      const fl = data.four_laws_adherence_l4;
      if (!fl || !Array.isArray(fl.rows) || fl.rows.length === 0) return "";
      const rows = fl.rows
        .map((r) => {
          const fam = r.family_id ? `<code>${escapeHtml(r.family_id)}</code>` : "—";
          if (r.blocked) {
            const bgText =
              r.bytegan_measured > 0 ? `${r.bytegan_pass}/${r.bytegan_measured}` : "BLOCKED";
            return `
              <tr>
                <td><code>${escapeHtml(r.candidate_id)}</code></td>
                <td>${fam}</td>
                <td><span class="pg-badge pg-badge-warn">BLOCKED</span></td>
                <td><span class="pg-badge pg-badge-warn">BLOCKED</span></td>
                <td><span class="pg-badge pg-badge-warn">BLOCKED</span></td>
                <td><span class="pg-badge pg-badge-warn">BLOCKED</span></td>
                <td><code>0/0</code> <span class="portal-muted">(${escapeHtml(r.blocked_reason || "blocked")})</span></td>
                <td><code>${escapeHtml(bgText)}</code></td>
                <td class="num">—</td>
              </tr>
            `;
          }
          const lawCell = (lid) => {
            const b = (r.laws && r.laws[lid]) || {};
            const p = b.pass ?? 0;
            const t = b.total ?? 0;
            const cls = p > 0 ? "pg-badge-pass" : "pg-badge-muted";
            return `<span class="pg-badge ${cls}">${p}/${t}</span>`;
          };
          const bgText =
            r.bytegan_measured > 0 ? `${r.bytegan_pass}/${r.bytegan_measured}` : "—";
          const bgCls =
            r.bytegan_measured > 0 && r.bytegan_pass === r.bytegan_measured
              ? "pg-badge-pass"
              : "pg-badge-muted";
          const energyText =
            r.nine_probe_joules_total != null
              ? `${(r.nine_probe_joules_total / 1000).toFixed(2)} kJ`
              : "—";
          return `
            <tr>
              <td><strong><code>${escapeHtml(r.candidate_id)}</code></strong></td>
              <td>${fam}</td>
              <td>${lawCell("law1")}</td>
              <td>${lawCell("law2")}</td>
              <td>${lawCell("law3")}</td>
              <td>${lawCell("law4")}</td>
              <td><strong>${r.probes_pass}/${r.probes_measured}</strong></td>
              <td><span class="pg-badge ${bgCls}">${escapeHtml(bgText)}</span></td>
              <td class="num">${escapeHtml(energyText)}</td>
            </tr>
          `;
        })
        .join("");

      return `
        <section class="portal-panel">
          <div class="portal-panel-head">
            <div>
              <h2 class="portal-panel-title">Law Adherence Diagnostic Leaderboard (${escapeHtml(fl.hardware_gpu || "NVIDIA L4")})</h2>
              <p class="portal-panel-sub">Matched ${escapeHtml(fl.hardware_gpu || "NVIDIA L4")} cohort (<code>four-laws-adherence-v1</code>, protocol seed 0, unchanged recipes). Research diagnostic telemetry only — separate from ordinary selected-cohort qualification.</p>
            </div>
            <a class="portal-btn" href="${escapeHtml(siteRoot)}laws/">Full Per-Probe Margins →</a>
          </div>
          <div class="portal-table-wrap">
            <table class="portal-table">
              <thead>
                <tr>
                  <th>Candidate</th>
                  <th>Family</th>
                  <th>Law 1</th>
                  <th>Law 2</th>
                  <th>Law 3</th>
                  <th>Law 4</th>
                  <th>Probes PASS</th>
                  <th>ByteGAN (L4)</th>
                  <th class="num">9-Probe NVML</th>
                </tr>
              </thead>
              <tbody>${rows}</tbody>
            </table>
          </div>
        </section>
      `;
    }

    function render() {
      if (!state.data) return;
      rootEl.innerHTML = [
        renderKpis(state.data),
        renderLeaderboardPanel(state.data),
        renderParetoPanel(state.data),
        renderLawsMatrixPanel(state.data),
        renderFourLawsAdherencePanel(state.data),
        renderGifPanel(state.data),
      ].join("");
      bindEvents();
    }

    fetch(dataUrl)
      .then((resp) => {
        if (!resp.ok) throw new Error(`HTTP ${resp.status} loading ${dataUrl}`);
        return resp.json();
      })
      .then((payload) => {
        state.data = payload;
        render();
      })
      .catch((err) => {
        rootEl.innerHTML = `
          <div class="portal-loading-card">
            <p><strong>Unable to load leaderboard snapshot.</strong></p>
            <p class="portal-muted">${escapeHtml(err.message)} — view the <a href="${escapeHtml(siteRoot)}technique-inventory/">Canonical Markdown Leaderboard</a> or <a href="https://huggingface.co/spaces/HyperGAN/particlegan-leaderboard">Hugging Face Space</a>.</p>
          </div>
        `;
      });
  }

  function mount() {
    if (typeof document === "undefined") return;
    const rootEl = document.getElementById("portal-leaderboard-root");
    if (!rootEl || rootEl.dataset.pgMounted === "1") return;
    rootEl.dataset.pgMounted = "1";
    createPortalApp(rootEl);
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = {
      computeParetoFrontier,
      sortFamilies,
      hfGifUrl,
      formatNumber,
      formatJoules,
      cohortCategory,
      statusBadgeClass,
    };
  }

  if (typeof document !== "undefined") {
    if (typeof document$ !== "undefined" && document$.subscribe) {
      document$.subscribe(() => mount());
    }
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", mount, { once: true });
    } else {
      mount();
    }
  }
})(typeof globalThis !== "undefined" ? globalThis : window);
