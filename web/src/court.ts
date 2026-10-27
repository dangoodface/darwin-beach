// Draws the kernel's state. World units follow slimevolleygym: the court runs
// from x = -24 to 24, the ground's top is at y = 1.5, the fence stands at x = 0
// up to y = 3.5 with a round cap, slimes are half circles of radius 1.5.
import type { Snapshot } from "./kernel.ts";

export const VIEW = { x0: -24, x1: 24, y0: 0, y1: 18 };
const GROUND = 1.5;
const SLIME_R = 1.5;
const BALL_R = 0.5;

export type CourtColours = { sky: string; sand: string; line: string; left: string; right: string; ball: string; eye: string; pupil: string; text: string };

// World to canvas pixels for a canvas of width w and height h.
export function toScreen(x: number, y: number, w: number, h: number): [number, number] {
  const sx = ((x - VIEW.x0) / (VIEW.x1 - VIEW.x0)) * w;
  const sy = h - ((y - VIEW.y0) / (VIEW.y1 - VIEW.y0)) * h;
  return [sx, sy];
}

export function drawCourt(ctx: CanvasRenderingContext2D, s: Snapshot, c: CourtColours, labels?: { left: string; right: string; score?: string }) {
  const { width: w, height: h } = ctx.canvas;
  const unit = w / (VIEW.x1 - VIEW.x0);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = c.sky;
  ctx.fillRect(0, 0, w, h);

  // sand and the fence
  const [, groundY] = toScreen(0, GROUND, w, h);
  ctx.fillStyle = c.sand;
  ctx.fillRect(0, groundY, w, h - groundY);
  const [fx, fy] = toScreen(0, 3.5, w, h);
  ctx.fillStyle = c.line;
  ctx.fillRect(fx - unit * 0.5, fy, unit, groundY - fy);
  ctx.beginPath();
  ctx.arc(fx, fy, unit * 0.5, 0, Math.PI * 2);
  ctx.fill();

  const [bx, by] = toScreen(s.ball[0], s.ball[1], w, h);
  slime(ctx, s.a, c.left, bx, by, w, h, unit, c);
  slime(ctx, s.b, c.right, bx, by, w, h, unit, c);

  // the ball; when it flies above the view, a marker at the top edge shows where
  ctx.fillStyle = c.ball;
  if (by < 0) {
    ctx.beginPath();
    ctx.moveTo(bx, 2);
    ctx.lineTo(bx - unit * 0.5, unit * 0.8);
    ctx.lineTo(bx + unit * 0.5, unit * 0.8);
    ctx.fill();
  } else {
    ctx.beginPath();
    ctx.arc(bx, by, unit * BALL_R, 0, Math.PI * 2);
    ctx.fill();
  }

  if (labels) {
    ctx.fillStyle = c.text;
    ctx.font = `${Math.max(10, Math.round(w / 48))}px "Press Start 2P", monospace`;
    ctx.textBaseline = "top";
    ctx.textAlign = "left";
    ctx.fillText(labels.left, unit * 0.8, unit * 0.8);
    ctx.textAlign = "right";
    ctx.fillText(labels.right, w - unit * 0.8, unit * 0.8);
    if (labels.score) {
      ctx.textAlign = "center";
      ctx.fillText(labels.score, w / 2, unit * 0.8);
    }
  }
}

function slime(ctx: CanvasRenderingContext2D, p: number[], colour: string, bx: number, by: number, w: number, h: number, unit: number, c: CourtColours) {
  const [x, y] = toScreen(p[0], p[1], w, h);
  ctx.fillStyle = colour;
  ctx.beginPath();
  ctx.arc(x, y, unit * SLIME_R, Math.PI, 0);
  ctx.closePath();
  ctx.fill();
  // the eye sits on the side facing the net and follows the ball, as in the original game
  const facing = p[0] < 0 ? 1 : -1;
  const ex = x + facing * unit * 0.6;
  const ey = y - unit * 0.85;
  ctx.fillStyle = c.eye;
  ctx.beginPath();
  ctx.arc(ex, ey, unit * 0.33, 0, Math.PI * 2);
  ctx.fill();
  const dx = bx - ex;
  const dy = by - ey;
  const d = Math.hypot(dx, dy) || 1;
  ctx.fillStyle = c.pupil;
  ctx.beginPath();
  ctx.arc(ex + (dx / d) * unit * 0.14, ey + (dy / d) * unit * 0.14, unit * 0.15, 0, Math.PI * 2);
  ctx.fill();
}
