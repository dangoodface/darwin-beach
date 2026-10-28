// The browser's side of the kernel (docs/ABI.md). The same .wasm bytes the
// trainer ran, so a replay here lands on the trainer's hash or something is wrong.

// A genome is 515 f32 for kernel versions 1 and 2 (12-32-3) and 1,027 from version 3 (12-64-3).
export const GENOME_LEN = 515;
export const GENOME_LEN_V3 = 1027;
const N_IN = 12;
const N_OUT = 3;
export const SCRIPTED_SLOT = 0xffffffff;

// Action bits, per side: forward means towards the net.
export const FORWARD = 1;
export const BACKWARD = 2;
export const JUMP = 4;

type Exports = {
  memory: WebAssembly.Memory;
  kernel_version(): number;
  genome_len(): number;
  table_capacity(): number;
  genome_table_ptr(): number;
  state_ptr(): number;
  reset(seed: bigint): void;
  set_actions(a: number, b: number): void;
  step(): number;
  is_done(maxTicks: number): number;
  policy_action(side: number, slot: number): number;
  set_holds?(holdA: number, holdB: number): number;
  held_action?(side: number, slot: number): number;
  state_hash(): bigint;
};

export type Snapshot = {
  ball: [number, number, number, number];
  a: [number, number, number, number];
  b: [number, number, number, number];
  livesA: number;
  livesB: number;
  tick: number;
  delay: number;
  lastPoint: number;
};

// A 515-parameter genome in the 1,027 layout (docs/ABI.md, "Padding a version 2 genome"):
// each block keeps its values first and gains zeros for hidden units 32 to 63, so it plays the same match.
export function padGenome(v2: Float32Array): Float32Array<ArrayBuffer> {
  const old = 32;
  const hid = 64;
  const out = new Float32Array(GENOME_LEN_V3);
  out.set(v2.subarray(0, old * N_IN), 0);
  out.set(v2.subarray(old * N_IN, old * N_IN + old), hid * N_IN);
  const w2 = old * N_IN + old;
  for (let r = 0; r < N_OUT; r++) out.set(v2.subarray(w2 + r * old, w2 + (r + 1) * old), hid * N_IN + hid + r * hid);
  out.set(v2.subarray(w2 + N_OUT * old), hid * N_IN + hid + N_OUT * hid);
  return out;
}

export class Kernel {
  readonly k: Exports;
  readonly genomeLen: number;
  private slots = new Map<string, number>();

  private constructor(k: Exports) {
    this.k = k;
    this.genomeLen = k.genome_len();
    if (this.genomeLen !== GENOME_LEN && this.genomeLen !== GENOME_LEN_V3) throw new Error(`kernel genome_len ${this.genomeLen}`);
  }

  static async load(bytes: BufferSource): Promise<Kernel> {
    const { instance } = await WebAssembly.instantiate(bytes, {});
    return new Kernel(instance.exports as unknown as Exports);
  }

  get version(): number {
    return this.k.kernel_version();
  }

  // Copies a genome's little-endian f32 into the next free table slot and returns the slot.
  // A 515-parameter genome on a version 3 kernel is padded first. The same genome id always gets the same slot.
  loadGenome(id: string, bytes: ArrayBuffer): number {
    const have = this.slots.get(id);
    if (have !== undefined) return have;
    let f: Float32Array = new Float32Array(bytes.slice(0));
    if (f.length === GENOME_LEN && this.genomeLen === GENOME_LEN_V3) f = padGenome(f);
    if (f.length !== this.genomeLen) throw new Error(`genome ${id} is ${bytes.byteLength} bytes`);
    const slot = this.slots.size;
    if (slot >= this.k.table_capacity()) throw new Error("genome table full");
    new Float32Array(this.k.memory.buffer, this.k.genome_table_ptr() + slot * this.genomeLen * 4, this.genomeLen).set(f);
    this.slots.set(id, slot);
    return slot;
  }

  // Holds (version 3): a side with hold h picks a new action every h ticks and repeats it in between.
  reset(seed: bigint, holdA = 1, holdB = 1): void {
    this.k.reset(seed);
    if (this.k.set_holds) {
      if (this.k.set_holds(holdA, holdB) !== 1) throw new Error(`bad holds ${holdA}, ${holdB}`);
    } else if (holdA !== 1 || holdB !== 1) throw new Error(`kernel version ${this.version} has no holds`);
  }

  // The action a genome (or the scripted bot) takes now for a side, under its hold: 0 is left (A), 1 is right (B).
  policy(side: 0 | 1, slot: number): number {
    const act = this.k.held_action ? this.k.held_action(side, slot) : this.k.policy_action(side, slot);
    if (act < 0) throw new Error(`policy_action refused slot ${slot}`);
    return act;
  }

  // One tick. Returns +1 when the left slime wins the point, -1 when the right one does, 0 otherwise.
  step(a: number, b: number): number {
    this.k.set_actions(a, b);
    return this.k.step();
  }

  done(maxTicks = 3000): boolean {
    return this.k.is_done(maxTicks) === 1;
  }

  hash(): bigint {
    return BigInt.asUintN(64, this.k.state_hash());
  }

  snapshot(): Snapshot {
    const s = new Float64Array(this.k.memory.buffer, this.k.state_ptr(), 17);
    return {
      ball: [s[0], s[1], s[2], s[3]],
      a: [s[4], s[5], s[6], s[7]],
      b: [s[8], s[9], s[10], s[11]],
      livesA: s[12],
      livesB: s[13],
      tick: s[14],
      delay: s[15],
      lastPoint: s[16],
    };
  }
}

// A whole match the way the trainer's Kernel.replay plays it, without drawing.
// Used by the tests and by the "verify all" check.
export function replayHeadless(k: Kernel, seed: bigint, left: number, right: number, maxTicks = 3000, holdA = 1, holdB = 1) {
  k.reset(seed, holdA, holdB);
  let pointsLeft = 0;
  let pointsRight = 0;
  while (!k.done(maxTicks)) {
    const won = k.step(k.policy(0, left), k.policy(1, right));
    if (won > 0) pointsLeft++;
    else if (won < 0) pointsRight++;
  }
  return { ticks: k.snapshot().tick, hash: k.hash(), pointsLeft, pointsRight };
}
