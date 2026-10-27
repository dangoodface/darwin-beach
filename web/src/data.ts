// The files the trainer writes and the page reads (docs/DATA.md). Every URL is
// relative so the page works from the dev server at / and from
// play.thegoodface.ai/darwin-beach/ alike.

export type Replay = {
  seed: number | string;
  opponent_id: string;
  side: "A" | "B";
  final_hash: string;
  ticks: number;
  points: [number, number];
};

export type Cell = {
  i: number;
  j: number;
  genome_id: string;
  fitness: number;
  rating: number | null;
  lit?: boolean;
  bd: [number, number];
  gen: number;
  era: number;
  margins?: number[];
  fresh?: { won: number; margin: number };
  replay: Replay;
};

export type Latest = {
  run_id: string;
  kernel_sha: string;
  kernel_version: number;
  generated_at: string;
  generation: number;
  era: number;
  probes?: string[];
  grid: {
    descriptors: string[];
    bounds: [number, number][];
    resolution: [number, number];
    lit_floor?: number;
  };
  cells: Cell[];
  stats: {
    filled: number;
    lit?: number;
    qd_score: number;
    champion_id: string;
    champion_rating: number | null;
    matches_total: number;
    matches_per_s: number;
    elapsed_s: number;
    fresh?: { seeds: number; matches: number; checked: number; beats_coach: number };
  };
};

export type CurvePoint = {
  gen: number;
  t: number;
  filled: number;
  lit?: number;
  qd_score: number;
  best_fitness: number;
  matches_total: number;
};

export type TreeNode = {
  id: string;
  parents: string[];
  op: string;
  gen: number;
  cell: [number, number] | null;
  fitness: number;
  alive: boolean;
  champion: boolean;
  x: number;
  y: number;
};

export type Tree = { nodes: TreeNode[]; edges: [string, string][]; champion_lineage: string[] };

export type RunIndexEntry = { run_id: string; title?: string; started_at?: string; latest_generation?: number };

export const SCRIPTED = "scripted-v1";

// Seeds are u64: JSON carries them as numbers up to 2^53 and as decimal strings above.
export function parseSeed(seed: number | string): bigint {
  if (typeof seed === "number") {
    if (!Number.isSafeInteger(seed) || seed < 0) throw new Error(`bad seed ${seed}`);
    return BigInt(seed);
  }
  if (!/^\d+$/.test(seed)) throw new Error(`bad seed ${seed}`);
  const v = BigInt(seed);
  if (v >= 1n << 64n) throw new Error(`seed out of range ${seed}`);
  return v;
}

// final_hash is "0x" plus 16 hex digits.
export function parseHash(hex: string): bigint {
  if (!/^0x[0-9a-f]{16}$/i.test(hex)) throw new Error(`bad hash ${hex}`);
  return BigInt(hex);
}

// The first run predates the lit flag; there a cell is lit when its fitness reaches the floor.
export function isLit(cell: Cell, floor = 0): boolean {
  return cell.lit ?? cell.fitness >= floor;
}

export function litCount(latest: Latest): number {
  const floor = latest.grid.lit_floor ?? 0;
  return latest.stats.lit ?? latest.cells.filter((c) => isLit(c, floor)).length;
}

export async function getJSON<T>(url: string): Promise<T> {
  const res = await fetch(url, { cache: "no-cache" });
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return (await res.json()) as T;
}

export async function getBytes(url: string): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.arrayBuffer();
}

// The newest run in data/index.json, by start time, falling back to the id (which starts with a timestamp).
export function newestRun(index: RunIndexEntry[]): string {
  if (!index.length) throw new Error("no runs yet");
  return [...index].sort((a, b) => (a.started_at ?? a.run_id).localeCompare(b.started_at ?? b.run_id)).at(-1)!.run_id;
}

export function runUrl(base: string, runId: string, file: string): string {
  return `${base}${runId}/${file}`;
}
