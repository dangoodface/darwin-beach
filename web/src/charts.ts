// The curve (how much of the map is lit, generation by generation) and the
// family tree (every bot that ever held a style, at the exporter's x, y).
import type { CurvePoint, Tree, TreeNode } from "./data.ts";

export type ChartColours = { line: string; area: string; grid: string; text: string; node: string; lineage: string; lineageFaint: string; champ: string };

export function drawCurve(canvas: HTMLCanvasElement, points: CurvePoint[], total: number, c: ChartColours) {
  const ctx = canvas.getContext("2d")!;
  const { width: w, height: h } = canvas;
  const pad = { l: 6, r: 6, t: 10, b: 18 };
  ctx.clearRect(0, 0, w, h);
  if (points.length < 2) return;
  // Runs from before the lit flag only have "filled" and the map score; then the curve shows the map score.
  const hasLit = points.some((p) => p.lit !== undefined);
  const value = (p: CurvePoint) => (hasLit ? p.lit ?? 0 : p.qd_score);
  const top = hasLit ? total : Math.max(...points.map(value));
  const g0 = points[0].gen;
  const g1 = points.at(-1)!.gen || 1;
  const X = (g: number) => pad.l + ((g - g0) / (g1 - g0 || 1)) * (w - pad.l - pad.r);
  const Y = (v: number) => h - pad.b - (v / (top || 1)) * (h - pad.t - pad.b);

  ctx.strokeStyle = c.grid;
  ctx.lineWidth = 1;
  for (const f of [0.25, 0.5, 0.75, 1]) {
    ctx.beginPath();
    ctx.moveTo(pad.l, Y(top * f));
    ctx.lineTo(w - pad.r, Y(top * f));
    ctx.stroke();
  }
  ctx.beginPath();
  ctx.moveTo(X(points[0].gen), Y(0));
  for (const p of points) ctx.lineTo(X(p.gen), Y(value(p)));
  ctx.lineTo(X(points.at(-1)!.gen), Y(0));
  ctx.closePath();
  ctx.fillStyle = c.area;
  ctx.fill();
  ctx.beginPath();
  points.forEach((p, n) => (n ? ctx.lineTo(X(p.gen), Y(value(p))) : ctx.moveTo(X(p.gen), Y(value(p)))));
  ctx.strokeStyle = c.line;
  ctx.lineWidth = 2;
  ctx.stroke();
  const last = points.at(-1)!;
  ctx.fillStyle = c.line;
  ctx.beginPath();
  ctx.arc(X(last.gen), Y(value(last)), 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = c.text;
  ctx.font = "11px 'JetBrains Mono', monospace";
  ctx.textBaseline = "bottom";
  ctx.textAlign = "left";
  ctx.fillText(`gen ${g0}`, pad.l, h - 2);
  ctx.textAlign = "right";
  ctx.fillText(`gen ${g1.toLocaleString("en")}`, w - pad.r, h - 2);
  return hasLit ? "lit" : "score";
}

export function drawTree(canvas: HTMLCanvasElement, tree: Tree, c: ChartColours) {
  const ctx = canvas.getContext("2d")!;
  const { width: w, height: h } = canvas;
  const pad = 8;
  const X = (x: number) => pad + x * (w - 2 * pad);
  const Y = (y: number) => pad + y * (h - 2 * pad);
  const byId = new Map(tree.nodes.map((n) => [n.id, n]));
  const lineage = new Set(tree.champion_lineage);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = c.node;
  for (const n of tree.nodes) ctx.fillRect(X(n.x) - 1, Y(n.y) - 1, 2, 2);
  // The champion's whole ancestry is a web (crossover gives two parents), so it is drawn faint,
  // and one line through it, always following the stronger parent, is drawn bright.
  ctx.strokeStyle = c.lineageFaint;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (const [p, ch] of tree.edges) {
    if (!lineage.has(p) || !lineage.has(ch)) continue;
    const a = byId.get(p);
    const b = byId.get(ch);
    if (!a || !b) continue;
    ctx.moveTo(X(a.x), Y(a.y));
    ctx.lineTo(X(b.x), Y(b.y));
  }
  ctx.stroke();
  const line = mainLine(tree);
  ctx.strokeStyle = c.lineage;
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  line.forEach((n, k) => (k ? ctx.lineTo(X(n.x), Y(n.y)) : ctx.moveTo(X(n.x), Y(n.y))));
  ctx.stroke();
  const champ = tree.nodes.find((n) => n.champion);
  if (champ) {
    ctx.fillStyle = c.champ;
    ctx.beginPath();
    ctx.arc(X(champ.x), Y(champ.y), 5, 0, Math.PI * 2);
    ctx.fill();
  }
}

// From the champion back to a generation-zero slime, always through the stronger parent.
export function mainLine(tree: Tree): TreeNode[] {
  const byId = new Map(tree.nodes.map((n) => [n.id, n]));
  let n = tree.nodes.find((x) => x.champion);
  const out: TreeNode[] = [];
  const seen = new Set<string>();
  while (n && !seen.has(n.id)) {
    seen.add(n.id);
    out.push(n);
    const parents = n.parents.map((id) => byId.get(id)).filter((x): x is TreeNode => !!x);
    n = parents.sort((a, b) => b.fitness - a.fitness)[0];
  }
  return out.reverse();
}
