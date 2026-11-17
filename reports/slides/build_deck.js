// Supervisor slide deck: liver filter arms, test results (from report 4 + checkpoint_best), as .pptx.
//
//   python reports/slides/make_assets.py <assets>          # charts + CT crops (PNG/JPG) + sizes.json
//   NODE_PATH=<dir with pptxgenjs> node reports/slides/build_deck.js <assets> <out.pptx> [<apply_theme.js>]
//
// Style: reports/style (ink / panel colours, fixed arm colours, green liver / red tumour). Text is Arial so the deck
// renders the same on any machine. Every number here is printed by make_assets.py or is in notes/RESULTS.md.
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const [ASSETS, OUT, THEME_JS] = process.argv.slice(2);
const SIZES = JSON.parse(fs.readFileSync(path.join(ASSETS, "sizes.json")));

const THEME = {
  name: "BT2026 report style", headFontFace: "Arial", bodyFontFace: "Arial",
  colors: {
    dk1: "1F1F1E", lt1: "FFFFFF", dk2: "5F5E58", lt2: "F6F5F2",
    accent1: "2A78D6", accent2: "EB6834", accent3: "1BAF7A", accent4: "EDA100", accent5: "E87BA4", accent6: "C81414",
    hlink: "2A78D6", folHlink: "6250D6",
  },
};
const HEX = { muted: "8A8985", grid: "E6E5E0", organ: "138A13", tumourTint: "FDE4E4", organTint: "E3F7E3",
  darkPanel: "2C2C2A", darkMuted: "B5B4AE" };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Liver filter arms: test results";
pres.subject = "BT2026 supervisor meeting";
const C = pres.SchemeColor;
const ARM = { all: C.accent1, ccReject: C.accent2, ccRepair: C.accent3, ccRepairDel: C.accent4,
  ccAucRepair: C.accent5 };

const W = 13.333, M = 0.6, CW = W - 2 * M; // content width 12.13
const TOP = 1.6, BOTTOM = 6.85;
const FOOT = "BT2026 · synthetic CT for liver tumour segmentation · liver filter arms";

// ------------------------------------------------------------------ layouts
pres.defineSlideMaster({
  title: "Title", background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "kicker", type: "body", x: 0.8, y: 1.9, w: 11.7, h: 0.4, fontSize: 14, bold: true,
      color: C.accent1, charSpacing: 2, margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.35, w: 11.7, h: 1.9, fontSize: 40, bold: true,
      color: C.background1, valign: "top", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "sub", type: "body", x: 0.8, y: 4.45, w: 11.0, h: 1.2, fontSize: 18,
      color: C.background2, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "Content", background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "kicker", type: "body", x: M, y: 0.38, w: CW, h: 0.3, fontSize: 12, bold: true,
      color: C.accent1, charSpacing: 2, margin: 0, valign: "middle" }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.7, w: CW, h: 0.75, fontSize: 30, bold: true,
      color: C.text1, valign: "top", align: "left", margin: 0 }, text: "" } },
    { text: { text: FOOT, options: { x: M, y: 7.0, w: 9, h: 0.3, fontSize: 10, color: C.text2, margin: 0 } } },
  ],
  slideNumber: { x: W - M - 0.6, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: C.text2, align: "right", margin: 0 },
});
pres.defineSlideMaster({
  title: "Dark", background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "kicker", type: "body", x: M, y: 0.38, w: CW, h: 0.3, fontSize: 12, bold: true,
      color: C.accent1, charSpacing: 2, margin: 0, valign: "middle" }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.7, w: CW, h: 0.75, fontSize: 30, bold: true,
      color: C.background1, valign: "top", align: "left", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: W - M - 0.6, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: C.background2, align: "right", margin: 0 },
});

// ------------------------------------------------------------------ helpers
let section = "";
function slide(master, kicker, title, notes) {
  const s = pres.addSlide({ masterName: master, sectionTitle: section });
  if (kicker) s.addText(kicker, { placeholder: "kicker" });
  s.addText(title, { placeholder: "title" });
  if (notes) s.addNotes(notes);
  return s;
}

// Fit an asset into a box, keeping its aspect ratio. align: horizontal "l" | "c" | "r", vertical "t" | "m".
function img(s, name, x, y, w, h, align = "c", valign = "t") {
  const [pw, ph] = SIZES[name];
  const k = Math.min(w / pw, h / ph);
  const iw = pw * k, ih = ph * k;
  const ix = align === "l" ? x : align === "r" ? x + w - iw : x + (w - iw) / 2;
  const iy = valign === "t" ? y : y + (h - ih) / 2;
  s.addImage({ path: path.join(ASSETS, name), x: ix, y: iy, w: iw, h: ih, altText: name, objectName: name });
  return { x: ix, y: iy, w: iw, h: ih };
}

function text(s, t, o) {
  return s.addText(t, Object.assign({ isTextBox: true, margin: 0, fontSize: 14, color: C.text1, valign: "top",
    paraSpaceAfter: 4 }, o));
}

function panel(s, x, y, w, h, fill = C.background2, name = "panel") {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: fill, width: 0 },
    rectRadius: 0.08, objectName: name });
}

// a "what this means" box: tinted panel, small label, body runs
function takeaway(s, x, y, w, h, label, runs, o = {}) {
  panel(s, x, y, w, h, o.fill || C.background2, "takeaway");
  text(s, label, { x: x + 0.25, y: y + 0.18, w: w - 0.5, h: 0.3, fontSize: 11, bold: true, charSpacing: 1.5,
    color: o.labelColor || C.accent3 });
  text(s, runs, { x: x + 0.25, y: y + 0.5, w: w - 0.5, h: h - 0.65, fontSize: o.fontSize || 14,
    color: o.color || C.text1 });
}

const b = (t, o = {}) => ({ text: t, options: Object.assign({ bold: true }, o) });
const r = (t, o = {}) => ({ text: t, options: o });
const br = (t, o = {}) => ({ text: t, options: Object.assign({ breakLine: true }, o) });
const bullet = (runs, last = false) => runs.map((x, i) => i === runs.length - 1 && !last
  ? { text: x.text, options: Object.assign({}, x.options, { breakLine: true }) } : x);

function caption(s, t, x, y, w, o = {}) {
  text(s, t, Object.assign({ x, y, w, h: 0.32, fontSize: 11, color: C.text2 }, o));
}

// CT strip: images side by side at equal height, each with a label above
function ctRow(s, names, labels, x, y, h, gap = 0.08) {
  let cx = x;
  names.forEach((n, i) => {
    const [pw, ph] = SIZES[n];
    const w = (pw / ph) * h;
    if (labels && labels[i]) caption(s, labels[i], cx, y - 0.3, w + 0.4, { fontSize: 10.5 });
    s.addImage({ path: path.join(ASSETS, n), x: cx, y, w, h, altText: labels ? labels[i] : n, objectName: n });
    cx += w + gap;
  });
  return cx - gap; // right edge
}

function tile(s, x, y, w, h, big, label, color) {
  panel(s, x, y, w, h, C.background2, "stat tile");
  text(s, big, { x: x + 0.25, y: y + 0.22, w: w - 0.5, h: 0.8, fontSize: 30, bold: true, color: color || C.text1 });
  text(s, label, { x: x + 0.25, y: y + 1.05, w: w - 0.5, h: h - 1.2, fontSize: 13, color: C.text2 });
}

// table helpers: header row + body rows, light rules
function table(s, head, rows, x, y, w, colW, o = {}) {
  const fs_ = o.fontSize || 13;
  const hd = head.map((t, i) => ({ text: t, options: { bold: true, color: C.text2, fontSize: fs_ - 1,
    align: i === 0 || o.left ? "left" : "right", border: [{ type: "none" }, { type: "none" }, { pt: 1, color: HEX.muted },
      { type: "none" }] } }));
  const body = rows.map((row, ri) => row.map((cell, i) => {
    const c = typeof cell === "object" && cell !== null && cell.text !== undefined ? cell : { text: String(cell) };
    return { text: c.text, options: Object.assign({ fontSize: fs_, color: C.text1, align: i === 0 || o.left ? "left" : "right",
      border: [{ type: "none" }, { type: "none" }, { pt: 0.75, color: HEX.grid }, { type: "none" }] }, c.options || {}) };
  }));
  s.addTable([hd, ...body], { x, y, w, colW, rowH: o.rowH || 0.36, margin: [0.03, 0.08, 0.03, 0.08], valign: "middle",
    fontFace: "Arial", objectName: o.name || "table" });
}

// a coloured run label: arm name in its colour
const armTxt = (name, armKey, o = {}) => ({ text: name, options: Object.assign({ bold: true, color: ARM[armKey] }, o) });

// box + arrow diagram pieces
function box(s, x, y, w, h, t, o = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.06, fill: { color: o.fill || C.background1 },
    line: { color: o.line || HEX.muted, width: 1, dashType: o.dash ? "dash" : "solid" }, objectName: o.name || "box" });
  text(s, t, { x: x + 0.1, y, w: w - 0.2, h, fontSize: o.fontSize || 13, align: "center", valign: "middle",
    color: o.color || C.text1, bold: !!o.bold });
}
function arrow(s, x1, y1, x2, y2, o = {}) {
  s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.001,
    h: Math.abs(y2 - y1) || 0.001, flipV: (y2 < y1) !== (x2 < x1) && y1 !== y2,
    line: { color: o.color || C.text2, width: 1.25, endArrowType: "triangle", dashType: o.dash ? "dash" : "solid" },
    objectName: "arrow" });
}

// ================================================================== main deck
section = "Main";
pres.addSection({ title: section });

// 1 ---------------------------------------------------------------- title
{
  const s = pres.addSlide({ masterName: "Title", sectionTitle: section });
  s.addText("BT2026 · SUPERVISOR MEETING", { placeholder: "kicker" });
  s.addText("Filtering MAISI synthetic CT for liver tumour segmentation: test results", { placeholder: "title" });
  s.addText([br("7 nnU-Net runs trained only on MAISI synthetic CT, tested on 111 real LiTS cases"),
    r("Sebestyén · November 2026")], { placeholder: "sub" });
  // motif: the two overlay colours of every CT figure
  s.addShape(pres.shapes.OVAL, { x: 0.8, y: 6.2, w: 0.32, h: 0.32, fill: { color: "00D900" }, line: { width: 0 },
    objectName: "liver dot" });
  s.addShape(pres.shapes.OVAL, { x: 1.22, y: 6.2, w: 0.32, h: 0.32, fill: { color: "FF1A1A" }, line: { width: 0 },
    objectName: "tumour dot" });
  text(s, "liver · tumour", { x: 1.7, y: 6.2, w: 3, h: 0.32, fontSize: 12, color: HEX.darkMuted, valign: "middle" });
  s.addNotes("Progress update on the liver filter study. Three things today: why we filter the way we do, what the " +
    "test set says, and how that led us to post-processing and what that showed. At the end, three questions I'd " +
    "like your view on.");
}

// 2 ---------------------------------------------------------------- summary
{
  const s = slide("Content", "SUMMARY", "Filtering helps, mostly in a way a test-time rule can copy",
    "Four numbers carry the story. Filtering is clearly better than using MAISI output as generated: +0.10 tumour " +
    "Dice for the best arm. Almost all of that comes from removing false positives: 67 invented tumour pieces per " +
    "case drop to 9. But the same rule applied to the predictions of the unfiltered model (ccPost) lifts it from " +
    "0.30 to 0.38, the level of the filtered arms. And two identical training runs of the best arm differ by 0.021, " +
    "about as much as the arms differ from each other.");
  const tw = (CW - 3 * 0.3) / 4;
  [["+0.10", "tumour Dice, best filter (ccAucRepair) vs training on all 999 volumes: 0.303 → 0.403", C.accent5],
    ["67 → 9", "false-positive tumour pieces per test case, all → ccAucRepair", C.accent6],
    ["0.30 → 0.38", "the unfiltered model with the cc rule applied at test time (ccPost) = filtered arms (0.37–0.38)",
      C.accent1],
    ["0.021", "tumour Dice between two identical ccAucRepair runs (two identical all runs: 0.004)", C.text2],
  ].forEach(([big, lab, col], i) => tile(s, M + i * (tw + 0.3), TOP + 0.05, tw, 2.0, big, lab, col));
  const y = TOP + 2.45;
  text(s, [
    b("1  "), r("MAISI makes two kinds of label error: "), b("tumours in the wrong place", { color: C.accent6 }),
    r(" and "), b("invisible tumours", { color: C.accent6 }), br(" inside the liver."),
    b("2  "), r("Filtering them out of training removes most false positives: every arm beats the baseline by "),
    br("+0.07 to +0.10."),
    b("3  "), r("The matching rules applied to the predictions (ccPost, ccAucPost) do most of the same job. What "),
    br("training adds on top is within run-to-run noise."),
    b("4  "), r("So the filters are an understanding of "), r("what goes wrong", { italic: true }),
    r("; the test-time rules are the cheap fix."),
  ], { x: M, y, w: CW, h: BOTTOM - y, fontSize: 18, paraSpaceAfter: 10, lineSpacingMultiple: 1.05 });
}

// 3 ---------------------------------------------------------------- why filter
{
  const s = slide("Content", "WHY THESE FILTERS", "MAISI draws tumour labels where no tumour can be seen",
    "MAISI first draws a full-body label map, then the CT from it. Two errors are common in our pool. Top: wrong " +
    "place, 534 of 999 volumes have a tumour label at least 30 mm from the liver, often big blobs in the pelvis; here " +
    "95 ml in the pelvis over normal tissue. Bottom: invisible, a label inside the liver with nothing visible under " +
    "it; its separability is 0.54, and every real validation lesion scores at least 0.547. LiTS only labels liver " +
    "tumours, so both teach the model something wrong. The two rules on the right follow directly; they were fixed " +
    "on 12 and 14 November, before any filtered model was trained.");
  const h = 1.72;
  let y = TOP + 0.35;
  text(s, [b("Wrong place  ", { color: C.accent6 }), r("534 of 999 volumes have a tumour label ≥ 30 mm from the liver",
    { color: C.text2 })], { x: M, y: y - 0.35, w: 7.4, h: 0.3, fontSize: 13 });
  y += 0.3;
  ctRow(s, ["ct_off_0.jpg", "ct_off_1.jpg", "ct_off_3.jpg"], ["image", "image + label", "coronal"], M, y, h);
  y += h + 0.75;
  text(s, [b("Invisible  ", { color: C.accent6 }), r("label on tissue that looks like its surroundings (sep 0.54)",
    { color: C.text2 })], { x: M, y: y - 0.65, w: 7.4, h: 0.3, fontSize: 13 });
  ctRow(s, ["ct_inv_0.jpg", "ct_inv_1.jpg", "ct_inv_3.jpg"], ["image", "image + label", "coronal"], M, y, h);
  caption(s, "liver_0048 (95 ml, 131 mm from the liver) · liver_0735 (49 ml). W400/L40, green liver, red tumour.",
    M, y + h + 0.08, 6.8, { fontSize: 10.5 });

  const x = 7.55, w = W - M - x;
  panel(s, x, TOP, w, 2.45, C.background2, "cc rule");
  text(s, [b("cc rule", { fontSize: 18 }), r("   all arms", { color: C.text2, fontSize: 13 })],
    { x: x + 0.25, y: TOP + 0.2, w: w - 0.5, h: 0.4 });
  text(s, [br("A tumour must be part of the main liver structure (26-connected)."),
    r("Outside it → "), b("reject", { color: ARM.ccReject }), r(" the volume (343 left) or "),
    b("repair", { color: ARM.ccRepair }), r(": relabel as background (999 kept).")],
  { x: x + 0.25, y: TOP + 0.7, w: w - 0.5, h: 1.6, fontSize: 15 });
  const y2 = TOP + 2.7;
  panel(s, x, y2, w, 2.45, C.background2, "AUC rule");
  text(s, [b("AUC rule", { fontSize: 18 }), r("   ccAucRepair only", { color: C.text2, fontSize: 13 })],
    { x: x + 0.25, y: y2 + 0.2, w: w - 0.5, h: 0.4 });
  text(s, [br("A tumour must be at least as visible as the least visible real validation lesion: separability ≥ 0.5475."),
    r("Below → relabel to its surroundings (162 components in 144 volumes).")],
  { x: x + 0.25, y: y2 + 0.7, w: w - 0.5, h: 1.6, fontSize: 15 });
}

// 4 ---------------------------------------------------------------- every filter beats all
{
  const s = slide("Content", "RESULT 1", "Every filter beats training on all 999 volumes",
    "Main analysis, fixed in advance: per-case tumour Dice on 111 LiTS test cases, checkpoint_final, each arm " +
    "against the mean of the two identical 'all' runs, paired Wilcoxon with Holm over the four arms. All four arms " +
    "are clear wins, +0.067 to +0.100, and the repeat ccAucRepair run confirms it at +0.079. The grey rows at the " +
    "bottom are what retraining alone does: 0.004 for the 'all' pair, but 0.021 for the ccAucRepair pair. The main " +
    "claim, filter versus no filter, is far above both. Which filter is best is a different question; that comes " +
    "back on the noise slide. Liver Dice barely moves.");
  img(s, "gain.png", M, TOP + 0.05, 7.3, 4.6, "l");
  const x = 8.25, w = W - M - x;
  table(s, ["run", "vols", "Dice", "Holm p"], [
    [armTxt("all · all (run 2)", "all"), "999", "0.301 · 0.305", "–"],
    [armTxt("ccReject", "ccReject"), "343", "0.382", "0.001"],
    [armTxt("ccRepair", "ccRepair"), "999", "0.375", "2e−9"],
    [armTxt("ccRepairDel", "ccRepairDel"), "842", "0.370", "2e−7"],
    [armTxt("ccAucRepair", "ccAucRepair"), "999", { text: "0.403", options: { bold: true } }, "2e−10"],
    [armTxt("ccAucRepair (run 2)", "ccAucRepair"), "999", "0.381", "6e−11 *"],
  ].map(row => row.map(c => (typeof c === "string" ? c : { text: c.text, options: c.options }))),
  x, TOP + 0.1, w, [1.85, 0.5, 1.45, 0.78], { fontSize: 12.5, rowH: 0.42 });
  caption(s, "Mean tumour Dice per case, 111 LiTS test cases, checkpoint_final. Baseline 0.303 = mean of the two " +
    "all runs. * repeat run, unadjusted (outside the Holm family).", x, TOP + 3.15, w, { h: 0.8, fontSize: 11 });
  takeaway(s, x, TOP + 4.05, w, 1.2, "LIVER DICE", [r("barely moves: −0.003 to −0.022 (lowest in the arms with " +
    "fewer volumes)")], { fontSize: 13, labelColor: C.text2 });
}

// 5 ---------------------------------------------------------------- fewer false positives
{
  const s = slide("Content", "WHY IT WORKS", "The gain is fewer invented tumours, not more tumour found",
    "What changes is how much tumour the model invents. The unfiltered runs predict 46 to 67 tumour pieces per case " +
    "that touch no real lesion; the filtered runs 9 to 12. Precision rises from 0.55 to 0.70–0.85. Lesion recall " +
    "stays where it was for the repair arms; only ccReject loses some. The diagram is the likely mechanism, a " +
    "hypothesis: nnU-Net fits the labels it gets. 1,701 of 3,873 synthetic tumour components lie outside the liver, " +
    "so the model learns that tumour can appear anywhere; invisible labels teach it that normal liver texture can be " +
    "tumour. Both produce false positives, and false positives hurt most when the real tumour is small.");
  const r0 = img(s, "detection.png", M, TOP, CW, 3.4, "l");
  const y = TOP + r0.h + 0.35, bh = 0.62;
  const xs = [M, M + 3.15, M + 6.75], bw = [2.85, 3.3, 4.6];
  box(s, xs[0], y, bw[0], bh, "wrong-place tumour labels", { fill: C.background2, line: C.accent6, name: "cause 1" });
  box(s, xs[0], y + bh + 0.22, bw[0], bh, "invisible tumour labels", { fill: C.background2, line: C.accent6,
    name: "cause 2" });
  box(s, xs[1], y, bw[1], bh, "learns: tumour can be outside the liver", { fontSize: 12.5 });
  box(s, xs[1], y + bh + 0.22, bw[1], bh, "learns: normal liver can be tumour", { fontSize: 12.5 });
  box(s, xs[2], y + 0.42, bw[2], bh, "many false-positive pieces → low precision → low Dice, worst for small tumours",
    { fill: C.text1, line: C.text1, color: C.background1, fontSize: 12.5, name: "effect" });
  arrow(s, xs[0] + bw[0], y + bh / 2, xs[1], y + bh / 2);
  arrow(s, xs[0] + bw[0], y + bh * 1.5 + 0.22, xs[1], y + bh * 1.5 + 0.22);
  arrow(s, xs[1] + bw[1], y + bh / 2, xs[2], y + 0.42 + bh / 2 - 0.1);
  arrow(s, xs[1] + bw[1], y + bh * 1.5 + 0.22, xs[2], y + 0.42 + bh / 2 + 0.1);
  caption(s, "likely mechanism (hypothesis)", xs[2], y + 0.42 + bh + 0.08, bw[2], { fontSize: 10.5 });
}

// 6 ---------------------------------------------------------------- where FPs sit
{
  const s = slide("Content", "WHAT LED TO POST-PROCESSING", "The unfiltered model paints tumour all over the body",
    "For every predicted tumour piece we checked whether it lies inside the predicted main liver structure. In the " +
    "unfiltered runs 59 of 67 and 39 of 46 false-positive pieces per case lie outside the liver: most are specks, but " +
    "they hold half of the false-positive volume. They sit on round, low-contrast structures anywhere: a mass in the " +
    "chest, stool in the colon of a tumour-free case. These are the same places where MAISI drew wrong-place labels. " +
    "The filtered arms have only 2 to 4 such pieces left. That raised the obvious reviewer question: why not just " +
    "post-process the predictions instead of filtering the training data?");
  img(s, "fp_where.png", M, TOP, 5.2, 4.1, "l");
  const x = 6.25, h = 1.95;
  text(s, [b("lits_115 · all  "), r("32 ml “tumour” on a mass in the right chest", { color: C.text2 })],
    { x, y: TOP - 0.02, w: W - M - x, h: 0.3, fontSize: 13 });
  ctRow(s, ["ct_A1_0.jpg", "ct_A1_1.jpg", "ct_A1_2.jpg"], null, x, TOP + 0.33, h);
  const y2 = TOP + 0.33 + h + 0.25;
  text(s, [b("lits_091 · all  "), r("16 ml “tumour” on stool, a case without tumour", { color: C.text2 })],
    { x, y: y2, w: W - M - x, h: 0.3, fontSize: 13 });
  ctRow(s, ["ct_A5_0.jpg", "ct_A5_1.jpg", "ct_A5_2.jpg"], null, x, y2 + 0.35, h);
  panel(s, M, TOP + 4.45, 5.2, 0.8, C.background2, "question");
  text(s, [b("→ "), r("why not remove them at test time?")], { x: M + 0.25, y: TOP + 4.45, w: 4.8, h: 0.8,
    fontSize: 16, valign: "middle" });
}

// 7 ---------------------------------------------------------------- ccPost
{
  const s = slide("Content", "POST-PROCESSING 1 · ccPost", "ccPost gives the unfiltered model the same gain",
    "ccPost is the cc rule applied to the predictions: keep only the predicted liver-plus-tumour piece that holds the " +
    "largest liver piece. No training, no data. It lifts the two unfiltered runs from 0.30 to 0.38, the level of the " +
    "cc filter arms, and barely changes the filtered runs, which have little left to remove. With both sides " +
    "post-processed, ccReject, ccRepair and ccRepairDel no longer beat the baseline. Only ccAucRepair stays ahead: " +
    "+0.031 in run 1, Holm 0.003, but only +0.014 in run 2, Holm 0.09. So for wrong-place labels, filtering and " +
    "post-processing are interchangeable. ccPost costs 2 to 10 real lesions per run and fails when the liver itself is " +
    "missed, see backup.");
  img(s, "ccpost.png", M, TOP, 7.2, 4.7, "l");
  const x = 8.2, w = W - M - x;
  text(s, "both sides after ccPost, arm − baseline", { x, y: TOP + 0.05, w, h: 0.3, fontSize: 13, bold: true,
    color: C.text2 });
  table(s, ["arm", "gain", "Holm p"], [
    [armTxt("ccReject", "ccReject"), "+0.005", "0.89"],
    [armTxt("ccRepair", "ccRepair"), "−0.001", "0.86"],
    [armTxt("ccRepairDel", "ccRepairDel"), "−0.004", "0.62"],
    [armTxt("ccAucRepair", "ccAucRepair"), { text: "+0.031", options: { bold: true } }, { text: "0.003",
      options: { bold: true } }],
    [armTxt("ccAucRepair (run 2)", "ccAucRepair"), "+0.014", "0.09"],
  ], x, TOP + 0.45, w, [2.25, 1.0, 1.28], { fontSize: 13, rowH: 0.4 });
  takeaway(s, x, TOP + 3.05, w, 2.2, "WHAT THIS MEANS", [
    br("Wrong-place labels: filtering the training data and post-processing the output are interchangeable."),
    r("Only the visibility (AUC) repair keeps a lead; it is in the liver, where ccPost can't reach.")],
  { fontSize: 14 });
}

// 8 ---------------------------------------------------------------- ccAucPost + sweep
{
  const s = slide("Content", "POST-PROCESSING 2 · ccAucPost", "The visibility rule at test time closes the remaining gap",
    "If invisible training labels make the model paint tumour on normal-looking liver, those predicted pieces should " +
    "themselves be invisible in the CT. ccAucPost is ccPost plus the AUC rule on predicted pieces inside the liver, " +
    "with the training cut 0.5475; cut and main comparison were written down before the script ran. It works: real " +
    "hits have separability around 0.9, false positives around 0.6. The baseline with ccAucPost reaches 0.403, " +
    "exactly ccAucRepair as saved. The AUC-trained arm with ccPost against the baseline with ccAucPost: +0.009 in " +
    "run 1, −0.008 in run 2, zero on average. On the right, stricter cuts: 0.60 adds another 0.03 on test and the " +
    "20 validation cases agree, for 1 to 9 real lesions lost per run; 0.70 only loses lesions. Exploratory: the sweep " +
    "was chosen after the results.");
  img(s, "aucpost.png", M, TOP, 6.3, 4.15, "l");
  img(s, "sweep.png", 7.4, TOP - 0.05, W - M - 7.4, 4.25, "c");
  const y = TOP + 4.35;
  text(s, [
    b("Baseline + ccAucPost 0.403", { color: ARM.all }), r(" = ccAucRepair as saved. AUC training on top: "),
    b("+0.009"), r(" (run 1, p 0.42), "), b("−0.008"), br(" (run 2, p 0.053)."),
    r("Stricter cut 0.60: +0.03 more for 1–9 real lesions lost per run; val agrees, but chosen after the results.",
      { color: C.text2 }),
  ], { x: M, y, w: CW, h: BOTTOM - y + 0.05, fontSize: 15 });
}

// 9 ---------------------------------------------------------------- small tumours
{
  const s = slide("Content", "WHERE IT HELPS", "The biggest gain is on small tumours",
    "Test cases with tumour split into thirds by reference tumour volume. With a small tumour, a few false-positive " +
    "pieces outweigh the real lesion: the baseline's voxel precision there is only 0.02 and its Dice 0.09. That is " +
    "where filtering helps most, to 0.19–0.27, and also where the two ccAucRepair runs differ most. With large tumours " +
    "false positives matter less and the gain shrinks to +0.01 to +0.03. ccReject falls below the baseline there, " +
    "0.53 vs 0.60: rejecting a volume removes all of it, including correctly placed large tumours, so the model " +
    "becomes conservative. That is also part of why repair is the better default than reject.");
  img(s, "burden.png", M, TOP, CW, 4.0, "l");
  const y = TOP + 4.25, w = (CW - 0.3) / 2;
  takeaway(s, M, y, w, 1.05, "SMALL TUMOURS", [r("baseline 0.09 → 0.19–0.27: a few false positives outweigh the " +
    "real lesion (baseline precision 0.02)")], { fontSize: 14 });
  takeaway(s, M + w + 0.3, y, w, 1.05, "LARGE TUMOURS", [r("+0.01 to +0.03; "), armTxt("ccReject", "ccReject"),
    r(" falls below the baseline (0.53 vs 0.60)")], { fontSize: 14 });
}

// 10 --------------------------------------------------------------- noise
{
  const s = slide("Content", "HOW SURE ARE WE", "Retraining moves Dice as much as changing the filter",
    "We trained two arms twice with identical data and settings. The two 'all' runs differ by 0.004, which set our " +
    "0.01 noise floor. The two ccAucRepair runs differ by 0.021, with p 7e-5. That is the size of every arm-vs-arm " +
    "difference we see, 0.005 to 0.028. Even ccAucRepair vs ccRepair, the one with a CI clear of zero in run 1, " +
    "shrinks to +0.007 with run 2. So the main claim, filter versus no filter, is safe; any claim about which filter " +
    "is best needs at least two runs per arm, and the noise floor should come from all repeat pairs.");
  img(s, "noise.png", M, TOP + 0.1, 7.6, 4.2, "l");
  const x = 8.6, w = W - M - x;
  tile(s, x, TOP + 0.1, (w - 0.25) / 2, 1.75, "0.004", "two identical all runs");
  tile(s, x + (w - 0.25) / 2 + 0.25, TOP + 0.1, (w - 0.25) / 2, 1.75, "0.021", "two identical ccAucRepair runs",
    C.accent5);
  takeaway(s, x, TOP + 2.15, w, 2.35, "WHAT THIS MEANS", [
    br("Filter vs no filter (+0.07 to +0.10) is far above both."),
    br("Arm-vs-arm differences (0.005–0.028) are not: they need ≥ 2 runs per arm.", { breakLine: true }),
    r("The 0.01 noise floor from the all pair was too small.")], { fontSize: 14 });
}

// 11 --------------------------------------------------------------- checkpoint_best
{
  const s = slide("Content", "SECONDARY CHECK · checkpoint_best", "At checkpoint_best only the AUC arms keep their lead",
    "Secondary analysis, results from today: the same evaluation on checkpoint_best, the epoch with the best smoothed " +
    "validation pseudo Dice. Faded circles are checkpoint_final, the main analysis; diamonds are checkpoint_best. " +
    "First, ccReject is not overfitting late: its best checkpoint is much worse, 0.333 vs 0.382. The cc arms gain " +
    "from late training, the baseline does not. As saved, only the two AUC arms beat the baseline clearly at both " +
    "checkpoints, +0.063 and +0.066. With both sides post-processed, the cc arms fall below the baseline at " +
    "checkpoint_best and the AUC arms are no longer significant. Nothing beats a post-processed baseline at best.");
  img(s, "best.png", M, TOP, CW, 3.95, "l");
  const y = TOP + 4.2, w = (CW - 0.6) / 3;
  takeaway(s, M, y, w, 1.1, "NO LATE OVERFITTING", [armTxt("ccReject", "ccReject"), r(" is worse at best: 0.333 " +
    "vs 0.382 at final")], { fontSize: 13.5 });
  takeaway(s, M + w + 0.3, y, w, 1.1, "AS SAVED", [r("only the AUC arms beat the baseline at both checkpoints " +
    "(+0.063 / +0.066)")], { fontSize: 13.5 });
  takeaway(s, M + 2 * (w + 0.3), y, w, 1.1, "AFTER ccPost", [r("cc arms below baseline (Holm ≤ 0.015), AUC arms " +
    "n.s. (+0.014 / +0.009)")], { fontSize: 13.5 });
}

// 12 --------------------------------------------------------------- limits
{
  const s = slide("Content", "LIMITS", "What did not get better, and what to keep in mind",
    "What filtering does not fix: every model predicts tumour in all 10 tumour-free test cases, and neither " +
    "post-processing step helps there, because those false positives are visible fluid structures inside the " +
    "predicted liver. Lesion recall stays near 0.41: filtering removes wrong signal but adds no new signal. Liver " +
    "Dice dips slightly. And synthetic-only training is still far from real data: 0.377 vs 0.653 on the 40 thesis " +
    "test cases, which is expected and not what this study compares. Caveats on the right: one run per arm, the " +
    "post-processing analyses were added after the results, the metric favours removing false positives, MAISI saw " +
    "LiTS, and ccReject mixes 'fewer but cleaner' with 'fewer'.");
  const w = (CW - 0.4) / 2, h = 4.95;
  const items1 = [
    ["Tumour-free cases", "all 7 models predict tumour in all 10; ccPost and ccAucPost don't help (visible fluid " +
      "inside the predicted liver)"],
    ["Lesion recall ~0.41", "0.75 for lesions ≥ 1 ml; filtering removes wrong signal but adds none"],
    ["Liver Dice", "dips slightly in every arm (−0.003 to −0.022)"],
    ["Absolute level", "0.377 vs 0.653 for the real-data thesis model (40 thesis test cases)"],
  ];
  const items2 = [
    ["One run per arm", "two for ccAucRepair, and those differ by 0.021"],
    ["Added after the results", "ccPost, ccAucPost and the cut sweep are exploratory (cut 0.5475 fixed before)"],
    ["Metric", "per-case mean Dice rewards removing false positives in small-tumour cases"],
    ["Leakage / size", "MAISI was trained on data incl. LiTS (all arms alike); ccReject has a third of the volumes"],
  ];
  [[items1, "NOT IMPROVED", C.accent6, M], [items2, "CAVEATS", C.accent4, M + w + 0.4]].forEach(([items, lab, col, x]) => {
    panel(s, x, TOP, w, h, C.background2, lab);
    s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: TOP + 0.3, w: 0.22, h: 0.22, fill: { color: col }, line: { width: 0 },
      objectName: "marker" });
    text(s, lab, { x: x + 0.65, y: TOP + 0.26, w: w - 1, h: 0.3, fontSize: 12, bold: true, charSpacing: 1.5,
      color: C.text2 });
    const runs = [];
    items.forEach(([head, body], i) => {
      runs.push(br(head, { bold: true, fontSize: 15 }));
      runs.push(i === items.length - 1 ? r(body, { color: C.text2 }) : br(body, { color: C.text2 }));
      if (i < items.length - 1) runs.push(br(" ", { fontSize: 6 }));
    });
    text(s, runs, { x: x + 0.3, y: TOP + 0.75, w: w - 0.6, h: h - 0.95, fontSize: 14, paraSpaceAfter: 2 });
  });
}

// 13 --------------------------------------------------------------- discussion
{
  const s = slide("Dark", "FOR DISCUSSION", "Three things I'd like your view on",
    "One: post-processing lifts the raw MAISI set to the quality of the filtered sets. Is that the finding, or does " +
    "the paper need a gain on the training side? I think 'the filters explain what goes wrong, the test-time rules " +
    "are the cheap fix' is a clean story, but it is more modest than 'filter your synthetic data'. Two: the main " +
    "analysis was fixed as saved predictions before the results; I'd keep that and report ccPost and ccAucPost as " +
    "clearly marked secondary analyses, but the reader needs them to read the main result. Linked to that: MIDL full " +
    "paper around 3 December, or the short paper in April? Three: the next experiment I have ready is the size split, " +
    "two new folds of the ccAucRepair data split at the median tumour burden, about 7 GPU hours each.");
  const y = TOP + 0.1, h = 4.95, w = (CW - 0.6) / 3;
  const cards = [
    ["1", "Is “post-processing lifts the raw set to filtered quality” the finding?",
      [br("all + ccPost 0.381 ≈ cc arms 0.370–0.382"), br("all + ccAucPost 0.403 = ccAucRepair 0.403"),
        r("Enough for a paper, or does it need a training-side gain?")]],
    ["2", "How to report the post-processing, and which MIDL track?",
      [br("Main analysis (saved predictions) was fixed before the results."),
        br("ccPost / ccAucPost as clearly marked secondary analyses?"),
        r("Full paper (~3 Dec) or short paper (~15 Apr)?")]],
    ["3", "Next experiment: size-split arms?",
      [br("ccAucRepair split at the median burden (2.29 ml): sizeLarge 590 vols vs sizeSmall 589"),
        br("~7 GPU h per arm; ids ready"),
        r("Caveat: larger also means ~85× more tumour")]],
  ];
  cards.forEach(([n, head, body], i) => {
    const x = M + i * (w + 0.3);
    panel(s, x, y, w, h, HEX.darkPanel, "question card");
    s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: y + 0.3, w: 0.55, h: 0.55, fill: { color: C.accent1 },
      line: { width: 0 }, objectName: "number" });
    text(s, n, { x: x + 0.3, y: y + 0.3, w: 0.55, h: 0.55, fontSize: 18, bold: true, align: "center",
      valign: "middle", color: C.background1 });
    text(s, head, { x: x + 0.3, y: y + 1.05, w: w - 0.6, h: 1.3, fontSize: 18, bold: true, color: C.background1 });
    text(s, body, { x: x + 0.3, y: y + 2.45, w: w - 0.6, h: h - 2.6, fontSize: 14, color: HEX.darkMuted,
      paraSpaceAfter: 8 });
  });
}

// ================================================================== backup
section = "Backup";
pres.addSection({ title: section });
{
  const s = pres.addSlide({ masterName: "Dark", sectionTitle: section });
  s.addText("BACKUP", { placeholder: "kicker" });
  s.addText("Details for questions", { placeholder: "title" });
  text(s, "rules and arms · study design · full results table · where ccPost fails · repair vs reject · ccAucPost " +
    "readout · metrics", { x: M, y: 1.7, w: CW, h: 0.6, fontSize: 16, color: HEX.darkMuted });
}

// B1 rules + arms
{
  const s = slide("Content", "BACKUP · RULES AND ARMS", "Two rules per tumour component, five training sets",
    "Each tumour component of each synthetic volume goes through the same two questions. ccReject keeps only the " +
    "343 volumes the cc rule leaves unchanged; ccRepair keeps all 999 with the labels repaired, 157 of them " +
    "tumour-free afterwards; ccRepairDel drops those 157; ccAucRepair is ccRepair plus the AUC repair.");
  const bw = 2.3, x2 = M + 3.0, w2 = 2.9;
  box(s, M, TOP + 0.1, bw, 0.65, "each tumour component", { fill: HEX.tumourTint, line: C.accent6, name: "start" });
  box(s, M, TOP + 1.25, bw, 0.75, "in the main liver?", { bold: true, name: "q1" });
  box(s, x2, TOP + 1.18, w2, 0.9, "wrong place\nreject: drop the volume\nrepair: label → background",
    { dash: true, fontSize: 12, color: C.text2, name: "wrong place" });
  box(s, M, TOP + 2.55, bw, 0.85, "visible?\nsep ≥ 0.5475", { bold: true, name: "q2" });
  box(s, x2, TOP + 2.52, w2, 0.9, "invisible label\nccAucRepair only:\nlabel → surroundings",
    { dash: true, fontSize: 12, color: C.text2, name: "invisible" });
  box(s, M, TOP + 3.95, bw, 0.65, "keep as tumour", { fill: HEX.tumourTint, line: C.accent6, bold: true, name: "keep" });
  arrow(s, M + bw / 2, TOP + 0.75, M + bw / 2, TOP + 1.25);
  arrow(s, M + bw, TOP + 1.625, x2, TOP + 1.625);
  caption(s, "no", M + bw + 0.1, TOP + 1.3, 0.5);
  arrow(s, M + bw / 2, TOP + 2.0, M + bw / 2, TOP + 2.55);
  caption(s, "yes", M + bw / 2 + 0.1, TOP + 2.1, 0.5);
  arrow(s, M + bw, TOP + 2.975, x2, TOP + 2.975);
  caption(s, "no", M + bw + 0.1, TOP + 2.65, 0.5);
  arrow(s, M + bw / 2, TOP + 3.4, M + bw / 2, TOP + 3.95);
  caption(s, "yes", M + bw / 2 + 0.1, TOP + 3.5, 0.5);
  img(s, "composition.png", 6.9, TOP + 0.6, W - M - 6.9, 4.2, "r");
}

// B2 study design
{
  const s = slide("Content", "BACKUP · STUDY DESIGN", "Train on synthetic CT, monitor and test on real LiTS",
    "Training uses synthetic data only. Real LiTS cases are used only for validation monitoring, 20 cases, and the " +
    "test, 111 cases including the 40 thesis test cases. Main comparison: per-case tumour Dice of each arm vs the " +
    "baseline, paired Wilcoxon with Holm over the four arms; a clear win also needs a mean gain above 0.01.");
  const y0 = TOP + 0.4;
  box(s, M, y0 + 1.3, 2.3, 1.0, "MAISI pool\n999 synthetic liver CTs", { fill: C.background2, name: "pool" });
  box(s, M + 2.9, y0 + 1.4, 1.5, 0.8, "cc / AUC rules", { bold: true, name: "rules" });
  const arms = [["all · 999 (×2)", "all"], ["ccReject · 343", "ccReject"], ["ccRepair · 999", "ccRepair"],
    ["ccRepairDel · 842", "ccRepairDel"], ["ccAucRepair · 999 (×2)", "ccAucRepair"]];
  arms.forEach(([t, k], i) => {
    const yy = y0 + i * 0.75;
    box(s, M + 5.0, yy, 2.6, 0.55, t, { line: ARM[k], bold: true, fontSize: 12.5, name: k });
    arrow(s, M + 7.6, yy + 0.275, M + 8.3, y0 + 1.8);
    if (k === "all") arrow(s, M + 2.3, y0 + 1.6, M + 5.0, yy + 0.275);
    else arrow(s, M + 4.4, y0 + 1.8, M + 5.0, yy + 0.275);
  });
  arrow(s, M + 2.3, y0 + 1.8, M + 2.9, y0 + 1.8);
  box(s, M + 8.3, y0 + 1.35, 1.9, 0.9, "nnU-Net 3d_fullres\n1000 epochs per run", { fontSize: 12, name: "nnunet" });
  box(s, M + 10.6, y0 + 1.35, 1.5, 0.9, "111 LiTS test\ncheckpoint_final", { fill: C.text1, line: C.text1,
    color: C.background1, fontSize: 12, name: "test" });
  arrow(s, M + 10.2, y0 + 1.8, M + 10.6, y0 + 1.8);
  box(s, M + 8.3, y0 + 3.3, 1.9, 0.7, "20 real LiTS val", { fill: C.background2, fontSize: 12, name: "val" });
  arrow(s, M + 9.25, y0 + 3.3, M + 9.25, y0 + 2.25, { dash: true });
  caption(s, "monitoring only", M + 9.35, y0 + 2.6, 2);
}

// B3 full results table
{
  const s = slide("Content", "BACKUP · RESULTS TABLE", "Main analysis, checkpoint_final, 111 test cases",
    "Tumour Dice = mean of per-case Dice, the primary metric. Global = all tumour voxels pooled, which weights large " +
    "tumours. On the 40 thesis test cases alone the repair arms stay significant; ccReject does not.");
  table(s, ["run", "train vols", "tumour Dice", "gain [95 % CI]", "Holm p", "global Dice", "liver Dice"], [
    [armTxt("all · all (run 2)", "all"), "999", "0.301 · 0.305", "baseline 0.303", "–", "0.582 · 0.590", "0.870 · 0.876"],
    [armTxt("ccReject", "ccReject"), "343", "0.382", "+0.079 [+0.038, +0.120]", "0.0011", "0.594", "0.857"],
    [armTxt("ccRepair", "ccRepair"), "999", "0.375", "+0.072 [+0.051, +0.094]", "2.4e−9", "0.651", "0.866"],
    [armTxt("ccRepairDel", "ccRepairDel"), "842", "0.370", "+0.067 [+0.043, +0.092]", "2.2e−7", "0.651", "0.851"],
    [armTxt("ccAucRepair", "ccAucRepair"), "999", "0.403", "+0.100 [+0.072, +0.129]", "2.4e−10", "0.675", "0.870"],
    [armTxt("ccAucRepair (run 2)", "ccAucRepair"), "999", "0.381", "+0.079 [+0.057, +0.101]", "6.3e−11 *", "0.663",
      "0.873"],
  ], M, TOP + 0.1, CW, [2.6, 1.2, 1.55, 2.7, 1.3, 1.4, 1.38], { fontSize: 14, rowH: 0.5 });
  caption(s, "Gain = arm − baseline (mean of the two all runs), paired bootstrap 95 % CI (10,000 resamples); Wilcoxon " +
    "signed-rank, Holm over the 4 arms. * repeat run, unadjusted. Detection (all → filtered): FP pieces/case 46–67 → " +
    "9–12, precision 0.55–0.57 → 0.70–0.85, lesion recall 0.42–0.44 → 0.35–0.42. 40 thesis test cases: repair arms " +
    "+0.055 to +0.099 (significant); ccReject +0.057 [−0.005, +0.123], p 0.20.", M, TOP + 3.8, CW,
  { h: 1.0, fontSize: 12 });
}

// B4 ccPost failures
{
  const s = slide("Content", "BACKUP · WHERE ccPost FAILS", "ccPost trusts that the largest liver piece is the liver",
    "When a model breaks down on a case, the cc rule at test time can remove real anatomy. Top: all run 2 calls most " +
    "of the liver tumour, so the largest predicted liver piece is the spleen; ccPost keeps the spleen and deletes the " +
    "real liver with its tumour. Bottom: a correctly found 25 ml tumour in the left lobe is not connected to the " +
    "predicted liver because the liver around it is missing; ccPost deletes it. Every run still improves its liver " +
    "Dice under ccPost, but each loses 2 to 10 real lesions.");
  const h = 1.85;
  text(s, [b("lits_043 · all (run 2)  "), r("liver called tumour → largest “liver” piece is the spleen",
    { color: C.text2 })], { x: M, y: TOP - 0.05, w: CW, h: 0.3, fontSize: 13 });
  ctRow(s, ["ct_A3_0.jpg", "ct_A3_1.jpg", "ct_A3_2.jpg", "ct_A3_3.jpg"],
    ["CT", "reference", "as saved", "after ccPost (lesions 1 → 0)"], M, TOP + 0.6, h);
  const y2 = TOP + 0.6 + h + 0.25;
  text(s, [b("lits_130 · all (run 2)  "), r("a found 25 ml tumour beyond a gap in the predicted liver",
    { color: C.text2 })], { x: M, y: y2, w: CW, h: 0.3, fontSize: 13 });
  ctRow(s, ["ct_B1_0.jpg", "ct_B1_1.jpg", "ct_B1_2.jpg", "ct_B1_3.jpg"],
    ["reference, full slice", "reference, zoom", "as saved (Dice 0.14)", "after ccPost (Dice 0.02)"], M, y2 + 0.65,
    h);
  takeaway(s, 9.75, TOP + 0.6, W - M - 9.75, 3.2, "COST OF ccPost", [
    br("2–10 real lesions lost per run"), br(" ", { fontSize: 6 }),
    br("liver Dice still improves in every run (+0.015 to +0.027)"), br(" ", { fontSize: 6 }),
    r("what it can't reach: fluid or low-contrast structures touching the liver")], { fontSize: 14 });
}

// B5 repair vs reject
{
  const s = slide("Content", "BACKUP · REPAIR VS REJECT", "Reject and repair reach similar means, but repair is safer",
    "ccReject's mean +0.079 comes with the widest spread: 49 cases better, 31 worse, median gain 0. The repair arms " +
    "move almost every case up a little. ccReject's validation loss bottoms out at epoch 756 and then rises, and " +
    "it has the largest train/val gap; but checkpoint_best showed that its final checkpoint is still the better one. " +
    "Repair is the better default: same or higher Dice, fewer losers, better on large tumours, and it keeps N = 999.");
  img(s, "per_case.png", M, TOP + 0.1, 7.9, 3.7, "l");
  img(s, "curves.png", 8.85, TOP, W - M - 8.85, 3.9, "r");
  takeaway(s, M, TOP + 4.05, CW, 1.2, "WHAT THIS MEANS", [r("Same criterion, different action: repair moves almost " +
    "every case up a little (66 better, 15 worse), reject is all or nothing (49 better, 31 worse). Under ccPost the " +
    "mean difference disappears (+0.005 vs −0.001); the case for repair rests on spread, large tumours and N.")],
  { fontSize: 14 });
}

// B6 ccAucPost readout
{
  const s = slide("Content", "BACKUP · ccAucPost READOUT", "Training vs post-processing for the visibility rule",
    "The pre-declared readout is row two: the baseline with ccAucPost against the AUC-trained arm with ccPost. " +
    "+0.009 in run 1, −0.008 in run 2, zero on average. With ccAucPost on both sides ccAucRepair stays ahead in both " +
    "runs, but only run 1 is significant.");
  table(s, ["comparison (tumour Dice per case, 111 test cases)", "x", "y", "y − x [95 % CI]", "p"], [
    ["baseline: ccPost → ccAucPost (what the test-time AUC rule adds)", "0.381", "0.403", "+0.022 [+0.011, +0.035]",
      "7.8e−10"],
    [{ text: "baseline + ccAucPost → ccAucRepair + ccPost (pre-declared readout)", options: { bold: true } }, "0.403",
      "0.412", { text: "+0.009 [−0.008, +0.026]", options: { bold: true } }, "0.42"],
    ["the same readout, ccAucRepair (run 2)", "0.403", "0.395", "−0.008 [−0.021, +0.004]", "0.053"],
    ["the same readout, mean of the two ccAucRepair runs", "0.403", "0.403", "+0.000 [−0.013, +0.014]", "0.53"],
    ["ccAucRepair: ccPost → ccAucPost", "0.412", "0.426", "+0.014 [+0.006, +0.024]", "7.3e−9"],
    ["baseline → ccAucRepair, both with ccAucPost (Holm 0.043)", "0.403", "0.426", "+0.023 [+0.005, +0.043]",
      "0.0087"],
    ["baseline → ccAucRepair (run 2), both with ccAucPost (Holm 0.71)", "0.403", "0.417", "+0.014 [+0.002, +0.028]",
      "0.18"],
  ], M, TOP + 0.1, CW, [6.15, 0.95, 0.95, 2.8, 1.28], { fontSize: 13, rowH: 0.46 });
  caption(s, "ccAucPost = ccPost, then every predicted tumour piece inside the predicted liver with separability " +
    "< 0.5475 is relabelled to its surroundings. Cut and readout fixed in plans/PLAN_followups.md before the script " +
    "ran. Tumour-free cases made clean: 0/10 in every run and variant.", M, TOP + 3.95, CW, { h: 0.9, fontSize: 12 });
}

// B7 metrics
{
  const s = slide("Content", "BACKUP · DEFINITIONS", "Metrics and rules used on these slides",
    "Definitions as in report 4, section 12.");
  table(s, ["term", "definition"], [
    [b("tumour Dice per case"), "Dice of the tumour label per test case, averaged over cases (primary metric; small " +
      "and large tumours count equally)"],
    [b("baseline"), "per-case mean tumour Dice of the two identical all runs"],
    [b("separability (sep)"), "AUC of core HU vs a 2–10 mm shell around a tumour component; 0.5 = invisible, " +
      "1 = fully separated"],
    [b("AUC cut 0.5475"), "lowest sep of any real validation lesion (20 LiTS, lits_049 excluded as unreliable)"],
    [b("ccPost / ccAucPost"), "the cc rule / cc + AUC rule applied to the predictions instead of the training labels"],
    [b("false-positive piece"), "predicted 26-connected tumour piece that overlaps no reference lesion"],
    [b("lesion recall"), "reference lesions overlapped by any predicted tumour voxel / all reference lesions"],
    [b("statistics"), "paired bootstrap 95 % CI (10,000), Wilcoxon signed-rank, Holm over the 4 arms; clear win = " +
      "Holm p < 0.05 and mean gain > 0.01"],
  ].map(row => row.map(c => (typeof c === "string" ? c : { text: c.text, options: c.options }))),
  M, TOP + 0.05, CW, [2.9, 9.23], { fontSize: 13, rowH: 0.5, left: true });
}

// ------------------------------------------------------------------ write
(async () => {
  await pres.writeFile({ fileName: OUT });
  if (THEME_JS) {
    const { applyTheme } = require(THEME_JS);
    await applyTheme(OUT, THEME);
  }
  console.log("wrote", OUT);
})();
