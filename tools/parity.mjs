#!/usr/bin/env node
// Parity harness, Node side: plays matches on the kernel in V8 and writes
// every match's per-tick hashes, so tests/test_parity.py can compare them
// with wasmtime's tick by tick.
//
//   node tools/parity.mjs <kernel.wasm> < input.bin > output.bin
//   node tools/parity.mjs <kernel.wasm> --text < input.bin
//
// Input (little-endian): u32 genome count, u32 job count, u32 max_ticks,
// then genome_len f32 per genome, then 16-byte jobs (u64 seed, u32 a, u32 b).
// Output per job: u32 ticks, u32 flags, u64 final hash, the whole result block
// (24 + 16 * desc_count bytes: descriptors included), then ticks u64 tick hashes.
// --text prints "job tick hash" lines instead, for reading by eye.

import { readFileSync, writeSync } from "node:fs";

const [wasmPath, mode] = process.argv.slice(2);
if (!wasmPath) {
  console.error("usage: node tools/parity.mjs <kernel.wasm> [--text] < input.bin");
  process.exit(2);
}

const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {});
const k = instance.exports;
const genomeLen = k.genome_len();
const blockBytes = 24 + 16 * (k.desc_count ? k.desc_count() : 8); // version 1 has eight, and no export

const input = readFileSync(0);
const head = new DataView(input.buffer, input.byteOffset, input.byteLength);
const nGenomes = head.getUint32(0, true);
const nJobs = head.getUint32(4, true);
const maxTicks = head.getUint32(8, true);
const genomeBytes = nGenomes * genomeLen * 4;
const jobsAt = 12 + genomeBytes;
if (input.byteLength !== jobsAt + nJobs * 16) {
  console.error(`input is ${input.byteLength} bytes, expected ${jobsAt + nJobs * 16}`);
  process.exit(2);
}

// One bulk copy of the whole table, as the Python side does.
new Uint8Array(k.memory.buffer).set(input.subarray(12, jobsAt), k.genome_table_ptr());

const chunks = [];
for (let j = 0; j < nJobs; j++) {
  const at = jobsAt + j * 16;
  const seed = head.getBigUint64(at, true); // wasm wraps a BigInt to i64 itself
  const a = head.getUint32(at + 8, true); // 0xFFFFFFFF wraps to the i32 -1, the same bits
  const b = head.getUint32(at + 12, true);
  const ticks = k.play_match(seed, a, b, maxTicks, 1) >>> 0;
  const mem = new DataView(k.memory.buffer);
  const results = k.results_ptr();
  const flags = mem.getUint32(results + 20, true);
  const finalHash = mem.getBigUint64(results, true);
  const hashes = new Uint8Array(k.memory.buffer, k.tick_hash_ptr(), ticks * 8);
  if (mode === "--text") {
    const view = new DataView(hashes.buffer, hashes.byteOffset, hashes.byteLength);
    const lines = [];
    for (let t = 0; t < ticks; t++) {
      lines.push(`${j} ${t + 1} ${view.getBigUint64(t * 8, true).toString(16).padStart(16, "0")}`);
    }
    lines.push(`${j} final ${finalHash.toString(16).padStart(16, "0")} flags ${flags}`);
    writeSync(1, lines.join("\n") + "\n");
    continue;
  }
  const header = Buffer.alloc(16);
  header.writeUInt32LE(ticks, 0);
  header.writeUInt32LE(flags, 4);
  header.writeBigUInt64LE(finalHash, 8);
  const block = new Uint8Array(k.memory.buffer, results, blockBytes);
  chunks.push(header, Buffer.from(block), Buffer.from(hashes)); // copies before the next match reuses the buffers
}

if (mode !== "--text") {
  const out = Buffer.concat(chunks);
  let written = 0;
  while (written < out.length) written += writeSync(1, out, written);
}
