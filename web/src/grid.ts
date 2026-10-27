// The map: one square per style. x is court position (back wall to net),
// y is airtime (grounded at the bottom). Unlit cells are dim; lit ones glow
// brighter the stronger their best bot is.
import { isLit, type Cell, type Latest } from "./data.ts";

export type GridColours = { empty: string; dim: string; lit: [number, number, number]; frame: string; select: string; champ: string };

// 0 at the floor, 1 at the strongest lit cell; used for the glow.
export function glow(fitness: number, floor: number, best: number): number {
  if (best <= floor) return 1;
  return Math.max(0, Math.min(1, (fitness - floor) / (best - floor)));
}

export class Grid {
  private cells = new Map<string, Cell>();
  private n: [number, number] = [20, 20];
  private floor = 0;
  private best = 1;
  private champion = "";
  selected: Cell | null = null;
  hover: Cell | null = null;

  private canvas: HTMLCanvasElement;
  private colours: GridColours;

  constructor(canvas: HTMLCanvasElement, colours: GridColours) {
    this.canvas = canvas;
    this.colours = colours;
  }

  setData(latest: Latest) {
    this.n = latest.grid.resolution;
    this.floor = latest.grid.lit_floor ?? 0;
    this.champion = latest.stats.champion_id;
    this.cells = new Map(latest.cells.map((c) => [`${c.i},${c.j}`, c]));
    this.best = Math.max(this.floor, ...latest.cells.map((c) => c.fitness));
    if (this.selected) this.selected = this.cells.get(`${this.selected.i},${this.selected.j}`) ?? null;
  }

  cellAt(clientX: number, clientY: number): Cell | null {
    const r = this.canvas.getBoundingClientRect();
    const i = Math.floor(((clientX - r.left) / r.width) * this.n[0]);
    const j = this.n[1] - 1 - Math.floor(((clientY - r.top) / r.height) * this.n[1]);
    return this.cells.get(`${i},${j}`) ?? null;
  }

  championCell(): Cell | null {
    for (const c of this.cells.values()) if (c.genome_id === this.champion) return c;
    return null;
  }

  isChampion(c: Cell): boolean {
    return c.genome_id === this.champion;
  }

  draw() {
    const ctx = this.canvas.getContext("2d")!;
    const { width: w, height: h } = this.canvas;
    const [nx, ny] = this.n;
    const cw = w / nx;
    const ch = h / ny;
    const gap = Math.max(1, Math.round(cw * 0.08));
    const { lit } = this.colours;
    ctx.clearRect(0, 0, w, h);
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < ny; j++) {
        const c = this.cells.get(`${i},${j}`);
        const x = i * cw;
        const y = (ny - 1 - j) * ch;
        if (!c) ctx.fillStyle = this.colours.empty;
        else if (!isLit(c, this.floor)) ctx.fillStyle = this.colours.dim;
        else {
          const g = 0.28 + 0.72 * glow(c.fitness, this.floor, this.best);
          ctx.fillStyle = `rgba(${lit[0]},${lit[1]},${lit[2]},${g.toFixed(3)})`;
        }
        ctx.fillRect(x + gap / 2, y + gap / 2, cw - gap, ch - gap);
      }
    }
    const ring = (c: Cell | null, colour: string, width: number) => {
      if (!c) return;
      ctx.strokeStyle = colour;
      ctx.lineWidth = width;
      ctx.strokeRect(c.i * cw + width / 2, (ny - 1 - c.j) * ch + width / 2, cw - width, ch - width);
    };
    ring(this.championCell(), this.colours.champ, Math.max(2, cw * 0.12));
    ring(this.hover, this.colours.frame, Math.max(1, cw * 0.08));
    ring(this.selected, this.colours.select, Math.max(2, cw * 0.14));
  }
}
