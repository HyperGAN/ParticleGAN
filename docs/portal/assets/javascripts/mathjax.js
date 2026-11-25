/* MathJax 3 configuration for ParticleGAN Research Portal.
 * Supports $...$, $$...$$, \(...\), \[...\] and re-typesets cleanly on
 * MkDocs Material instant navigation events.
 */
window.MathJax = {
  tex: {
    inlineMath: [
      ["$", "$"],
      ["\\(", "\\)"],
    ],
    displayMath: [
      ["$$", "$$"],
      ["\\[", "\\]"],
    ],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
  startup: {
    typeset: true,
  },
};

if (typeof document$ !== "undefined" && document$.subscribe) {
  document$.subscribe(() => {
    if (window.MathJax && typeof window.MathJax.typesetPromise === "function") {
      window.MathJax.startup.output.clearCache();
      window.MathJax.typesetClear();
      window.MathJax.texReset();
      window.MathJax.typesetPromise();
    }
  });
}
