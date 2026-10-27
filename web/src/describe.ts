// Plain words for a style and a result, for people who never read the docs.
import type { Cell } from "./data.ts";

export function positionWords(p: number): string {
  if (p < 0.2) return "camps at the back wall";
  if (p < 0.4) return "plays deep in the back court";
  if (p < 0.6) return "plays mid court";
  if (p < 0.8) return "plays close to the net";
  return "hugs the net";
}

export function airWords(a: number): string {
  if (a < 0.1) return "almost never jumps";
  if (a < 0.3) return "jumps now and then";
  if (a < 0.55) return "hops a lot";
  if (a < 0.8) return "spends most of its time in the air";
  return "barely touches the sand";
}

export function styleSentence(cell: Pick<Cell, "bd">): string {
  const [p, a] = cell.bd;
  const s = `${positionWords(p)} and ${airWords(a)}`;
  return s[0].toUpperCase() + s.slice(1) + ".";
}

// Fitness is the average point margin per match against the four rivals.
export function strengthWords(fitness: number): string {
  if (fitness >= 2) return "Strong: wins most of its matches by a couple of points";
  if (fitness >= 1) return "Good: usually comes out ahead";
  if (fitness >= 0) return "Holds its own";
  if (fitness >= -1) return "Still learning: loses a little more than it wins";
  return "Still learning: loses most matches";
}

export function margin(f: number): string {
  return `${f >= 0 ? "+" : ""}${f.toFixed(2)} points a match`;
}

// The coach's column in the per-rival margins. Graded on the run's fixed test serves,
// which flatter the bots picked on them, so the page never calls this "beats the coach".
export function coachLine(cell: { margins?: number[] }, probes: string[] | undefined, coach: string): string | null {
  const k = probes?.indexOf(coach) ?? -1;
  const m = k >= 0 ? cell.margins?.[k] : undefined;
  if (m === undefined) return null;
  const verdict = m > 0 ? "ahead" : m < 0 ? "behind" : "level";
  return `Against the coach: ${verdict}, ${margin(m)} on the serves it was graded on.`;
}
