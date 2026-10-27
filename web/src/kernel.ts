// The browser's side of the kernel (docs/ABI.md). The same .wasm bytes the
// trainer ran, so a replay here lands on the trainer's hash or something is wrong.

export const GENOME_LEN = 515;
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

export class Kernel {
  readonly k: Exports;
  private slots = new Map<string, number>();

  private constructor(k: Exports) {
    this.k = k;
    if (k.genome_len() !== GENOME_LEN) throw new Error(`kernel genome_len ${k.genome_len()}`);
  }

  static async load(bytes: BufferSource): Promise<Kernel> {
    const { instance } = await WebAssembly.instantiate(bytes, {});
    return new Kernel(instance.exports as unknown as Exports);
  }

  get version(): number {
    return this.k.kernel_version();
  }

  // Copies a genome's 515 little-endian f32 into the next free table slot and returns the slot.
  // The same genome id always gets the same slot.
  loadGenome(id: string, bytes: ArrayBuffer): number {
    const have = this.slots.get(id);
    if (have !== undefined) return have;
    if (bytes.byteLength !== GENOME_LEN * 4) throw new Error(`genome ${id} is ${bytes.byteLength} bytes`);
    const slot = this.slots.size;
    if (slot >= this.k.table_capacity()) throw new Error("genome table full");
    new Uint8Array(this.k.memory.buffer).set(new Uint8Array(bytes), this.k.genome_table_ptr() + slot * GENOME_LEN * 4);
    this.slots.set(id, slot);
    return slot;
  }

  reset(seed: bigint): void {
    this.k.reset(seed);
  }

  // The action a genome (or the scripted bot) takes now for a side: 0 is left (A), 1 is right (B).
  policy(side: 0 | 1, slot: number): number {
    const act = this.k.policy_action(side, slot);
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
export function replayHeadless(k: Kernel, seed: bigint, left: number, right: number, maxTicks = 3000) {
  k.reset(seed);
  let pointsLeft = 0;
  let pointsRight = 0;
  while (!k.done(maxTicks)) {
    const won = k.step(k.policy(0, left), k.policy(1, right));
    if (won > 0) pointsLeft++;
    else if (won < 0) pointsRight++;
  }
  return { ticks: k.snapshot().tick, hash: k.hash(), pointsLeft, pointsRight };
}
