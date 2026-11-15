// One diagram style for every report page (pairs with reports/style/style.py and report.css).
// Diagrams are Mermaid flowcharts written inline as <script type="text/x-mermaid">; this module adds the shared
// classes, theme and font, then renders them. Use the classes, never inline colours:
//
//   data      source / dataset box (neutral panel)        step     processing step (white, ink border)
//   decision  rule / question (diamond in the diagram)   muted    something dropped or not used (dashed)
//   organ     liver / pancreas (green, as in CT figures)  tumour   tumour label (red, as in CT figures)
//   all ccReject ccRepair ccRepairDel ccAucRepair         one class per training arm, same colour as in charts
//   result    test-set outcome (ink fill)
//
//   <script type="module">import {renderDiagrams} from "../style/diagrams.js"; await renderDiagrams();</script>
//
// Mermaid is pinned; the page sets document.body.dataset.ready = "1" when every diagram is drawn
// (build scripts wait for it before printing to PDF).

import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.esm.min.mjs";

export const T = {
  ink: "#1f1f1e", ink2: "#5f5e58", muted: "#8a8985", grid: "#e6e5e0", surface: "#ffffff", panel: "#f6f5f2",
  organ: "#138a13", organTint: "#e3f7e3", tumour: "#c81414", tumourTint: "#fde4e4",
  arm: { all: "#2a78d6", ccReject: "#eb6834", ccRepair: "#1baf7a", ccRepairDel: "#eda100", ccAucRepair: "#e87ba4" },
};

const tint = (hex, a) => {  // mix hex with white
  const n = parseInt(hex.slice(1), 16), c = [n >> 16, (n >> 8) & 255, n & 255];
  return "#" + c.map(v => Math.round(255 - (255 - v) * a).toString(16).padStart(2, "0")).join("");
};

const CLASSES = [
  `classDef data fill:${T.panel},stroke:${T.grid},stroke-width:1px,color:${T.ink}`,
  `classDef step fill:${T.surface},stroke:${T.ink2},stroke-width:1px,color:${T.ink}`,
  `classDef decision fill:${T.surface},stroke:${T.ink},stroke-width:1.2px,color:${T.ink}`,
  `classDef muted fill:${T.surface},stroke:${T.muted},stroke-width:1px,stroke-dasharray:4 3,color:${T.ink2}`,
  `classDef organ fill:${T.organTint},stroke:${T.organ},stroke-width:1.2px,color:${T.ink}`,
  `classDef tumour fill:${T.tumourTint},stroke:${T.tumour},stroke-width:1.2px,color:${T.ink}`,
  `classDef result fill:${T.ink},stroke:${T.ink},stroke-width:1px,color:#ffffff`,
  ...Object.entries(T.arm).map(([k, c]) =>
    `classDef ${k} fill:${tint(c, 0.16)},stroke:${c},stroke-width:1.6px,color:${T.ink}`),
].join("\n");

export async function renderDiagrams(selector = 'script[type="text/x-mermaid"]') {
  // Diagram source sits in <script type="text/x-mermaid"> (never parsed as HTML, so <br/> and --> survive);
  // each one is replaced by a <div class="diagram"> holding the rendered SVG.
  mermaid.initialize({
    startOnLoad: false, theme: "base", securityLevel: "strict",
    flowchart: { curve: "basis", padding: 12, nodeSpacing: 28, rankSpacing: 34, htmlLabels: true, useMaxWidth: true },
    themeVariables: {
      fontFamily: "Inter, DejaVu Sans, sans-serif", fontSize: "12.5px",
      primaryColor: T.surface, primaryBorderColor: T.ink2, primaryTextColor: T.ink,
      lineColor: T.ink2, textColor: T.ink, edgeLabelBackground: T.surface,
      clusterBkg: T.panel, clusterBorder: T.grid, titleColor: T.ink2,
    },
  });
  await document.fonts.ready;
  let i = 0;
  for (const el of [...document.querySelectorAll(selector)]) {
    const { svg } = await mermaid.render(`diagram-${i++}`, el.textContent.trim() + "\n" + CLASSES);
    const div = document.createElement("div");
    div.className = "diagram";
    div.innerHTML = svg;
    el.replaceWith(div);
  }
  document.body.dataset.ready = "1";
}
