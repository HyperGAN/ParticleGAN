/* Theme-aware Mermaid v11 renderer for ParticleGAN Research Portal.
 * Preserves raw diagram definitions so switching between Material light
 * ("default") and dark ("slate") palettes re-renders cleanly with high contrast.
 */
(() => {
  "use strict";

  function isDarkScheme() {
    const scheme =
      document.body?.getAttribute("data-md-color-scheme") ||
      document.documentElement?.getAttribute("data-md-color-scheme");
    if (scheme === "slate") return true;
    if (scheme === "default") return false;
    return Boolean(window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches);
  }

  function mermaidConfig() {
    const dark = isDarkScheme();
    return {
      startOnLoad: false,
      securityLevel: "loose",
      theme: dark ? "dark" : "neutral",
      themeVariables: dark
        ? {
            background: "#182235",
            primaryColor: "#24324d",
            primaryTextColor: "#f1f5f9",
            primaryBorderColor: "#818cf8",
            lineColor: "#94a3b8",
            secondaryColor: "#1b2a44",
            tertiaryColor: "#132033",
            fontFamily: "Inter, ui-sans-serif, system-ui, sans-serif",
          }
        : {
            background: "#f6f8fc",
            primaryColor: "#eef2ff",
            primaryTextColor: "#0f172a",
            primaryBorderColor: "#4338ca",
            lineColor: "#475569",
            secondaryColor: "#e0f2fe",
            tertiaryColor: "#ffffff",
            fontFamily: "Inter, ui-sans-serif, system-ui, sans-serif",
          },
      flowchart: {
        useMaxWidth: true,
        htmlLabels: true,
        curve: "basis",
      },
    };
  }

  function extractSource(block) {
    if (block.dataset.mermaidSource) return block.dataset.mermaidSource;
    const code = block.querySelector("code");
    const raw = (code ? code.textContent : block.textContent) || "";
    const trimmed = raw.trim();
    block.dataset.mermaidSource = trimmed;
    return trimmed;
  }

  async function renderAllMermaid() {
    if (typeof window.mermaid === "undefined") return;
    const blocks = Array.from(document.querySelectorAll("pre.mermaid, div.mermaid"));
    if (!blocks.length) return;

    window.mermaid.initialize(mermaidConfig());

    for (let idx = 0; idx < blocks.length; idx += 1) {
      const el = blocks[idx];
      const source = extractSource(el);
      if (!source) continue;
      const target = document.createElement("div");
      target.className = "mermaid";
      target.dataset.mermaidSource = source;
      try {
        const id = `pg-mermaid-${Date.now()}-${idx}`;
        const result = await window.mermaid.render(id, source);
        target.innerHTML = result.svg;
        if (typeof result.bindFunctions === "function") {
          result.bindFunctions(target);
        }
        el.replaceWith(target);
      } catch (err) {
        console.warn("[ParticleGAN Portal] Mermaid render warning:", err);
      }
    }
  }

  function boot() {
    renderAllMermaid();
    if (document.body && typeof MutationObserver !== "undefined") {
      let lastScheme = document.body.getAttribute("data-md-color-scheme");
      const obs = new MutationObserver(() => {
        const nextScheme = document.body.getAttribute("data-md-color-scheme");
        if (nextScheme !== lastScheme) {
          lastScheme = nextScheme;
          renderAllMermaid();
        }
      });
      obs.observe(document.body, {
        attributes: true,
        attributeFilter: ["data-md-color-scheme"],
      });
    }
  }

  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(() => {
      renderAllMermaid();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot, { once: true });
  } else {
    boot();
  }
})();
