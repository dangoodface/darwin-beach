//! Slime volleyball kernel, ABI version 1 (docs/ABI.md).
//!
//! Physics ported from `slimevolley.py` in hardmaru's slimevolleygym (David
//! Ha, 2020, Apache-2.0, https://github.com/hardmaru/slimevolleygym); see
//! `physics.rs` for the deviations. The same wasm bytes run in wasmtime for
//! training and in the browser for replays, which is where parity comes from:
//! inside, only + - * / sqrt, comparisons and integer ops on f64.
//!
//! All buffers are static (no allocator). Callers write genomes and jobs into
//! the exported memory in bulk, call `play_match` or `play_batch`, and read
//! result blocks back in bulk.

pub mod descriptors;
pub mod hash;
pub mod physics;
pub mod policy;
pub mod rng;
pub mod scripted;

use core::cell::UnsafeCell;

use descriptors::N_DESC;
use physics::{FLAG_NAN, Game};
use policy::{GENOME_LEN, net_action, upcast};

pub const KERNEL_VERSION: u32 = 1;
pub const MAX_TICKS_CAP: u32 = 3000;
pub const TABLE_CAPACITY: usize = 1024;
pub const BATCH_CAPACITY: usize = 256;
/// The table slot that means "the scripted anchor bot" in jobs and calls.
pub const SCRIPTED_SLOT: u32 = 0xFFFF_FFFF;
pub const STATE_LEN: usize = 17;

pub const FLAG_MAX_TICKS: u32 = 2;
pub const FLAG_BAD_JOB: u32 = 4;

/// One job in the jobs buffer: 16 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Job {
    pub seed: u64,
    pub a: u32,
    pub b: u32,
}

/// One result block: 152 bytes, laid out as docs/ABI.md says.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MatchResult {
    pub final_hash: u64,
    pub ticks: u32,
    pub points_a: u32,
    pub points_b: u32,
    pub flags: u32,
    pub desc_a: [f64; N_DESC],
    pub desc_b: [f64; N_DESC],
}

const _: () = assert!(core::mem::size_of::<Job>() == 16);
const _: () = assert!(core::mem::size_of::<MatchResult>() == 152);

impl MatchResult {
    pub const BAD_JOB: MatchResult = MatchResult {
        final_hash: 0,
        ticks: 0,
        points_a: 0,
        points_b: 0,
        flags: FLAG_BAD_JOB,
        desc_a: [0.0; N_DESC],
        desc_b: [0.0; N_DESC],
    };
}

/// Single-threaded wasm, so plain shared cells are enough. Native tests that
/// touch these go through `tests::exports_lock`.
struct Shared<T>(UnsafeCell<T>);

unsafe impl<T> Sync for Shared<T> {}

impl<T> Shared<T> {
    const fn new(value: T) -> Shared<T> {
        Shared(UnsafeCell::new(value))
    }

    fn ptr(&self) -> *mut T {
        self.0.get()
    }
}

static GENOMES: Shared<[f32; TABLE_CAPACITY * GENOME_LEN]> =
    Shared::new([0.0; TABLE_CAPACITY * GENOME_LEN]);
static JOBS: Shared<[Job; BATCH_CAPACITY]> = Shared::new(
    [Job {
        seed: 0,
        a: 0,
        b: 0,
    }; BATCH_CAPACITY],
);
static RESULTS: Shared<[MatchResult; BATCH_CAPACITY]> =
    Shared::new([MatchResult::BAD_JOB; BATCH_CAPACITY]);
static TICK_HASH: Shared<[u64; MAX_TICKS_CAP as usize]> = Shared::new([0; MAX_TICKS_CAP as usize]);
static STATE: Shared<[f64; STATE_LEN]> = Shared::new([0.0; STATE_LEN]);
/// The interactive game, separate from matches so `play_match` never
/// disturbs a replay in progress.
static GAME: Shared<Game> = Shared::new(Game::new(0));

/// Who controls a side: a genome from the table, or the anchor bot. Built
/// once per side per match, on the stack (boxing would need an allocator).
#[allow(clippy::large_enum_variant)]
enum Controller {
    Net([f64; GENOME_LEN]),
    Scripted,
}

impl Controller {
    fn from_slot(table: &[f32], slot: u32) -> Option<Controller> {
        if slot == SCRIPTED_SLOT {
            return Some(Controller::Scripted);
        }
        let start = (slot as usize).checked_mul(GENOME_LEN)?;
        let params = table.get(start..start + GENOME_LEN)?;
        Some(Controller::Net(upcast(params)))
    }

    fn action(&self, obs: &[f64; 12]) -> u32 {
        match self {
            Controller::Net(w) => net_action(w, obs),
            Controller::Scripted => scripted::scripted_action(obs),
        }
    }
}

/// Plays one match from a fresh game and writes its result block. Returns
/// the ticks played, 0 for a bad job.
pub fn run_match(
    table: &[f32],
    seed: u64,
    a: u32,
    b: u32,
    max_ticks: u32,
    mut tick_hashes: Option<&mut [u64]>,
    out: &mut MatchResult,
) -> u32 {
    let (Some(side_a), Some(side_b)) = (
        Controller::from_slot(table, a),
        Controller::from_slot(table, b),
    ) else {
        *out = MatchResult::BAD_JOB;
        return 0;
    };
    if max_ticks == 0 || max_ticks > MAX_TICKS_CAP {
        *out = MatchResult::BAD_JOB;
        return 0;
    }
    let mut game = Game::new(seed);
    while !game.is_done(max_ticks) {
        let action_a = side_a.action(&game.a.obs);
        let action_b = side_b.action(&game.b.obs);
        game.set_actions(action_a, action_b);
        game.step();
        if let Some(hashes) = tick_hashes.as_deref_mut()
            && game.flags & FLAG_NAN == 0
        {
            hashes[(game.tick - 1) as usize] = game.hash;
        }
    }
    let mut flags = game.flags;
    if game.tick >= max_ticks && game.a.life > 0 && game.b.life > 0 {
        flags |= FLAG_MAX_TICKS;
    }
    *out = MatchResult {
        final_hash: game.hash,
        ticks: game.tick,
        points_a: game.desc[0].points,
        points_b: game.desc[1].points,
        flags,
        desc_a: game.desc[0].finish(game.tick),
        desc_b: game.desc[1].finish(game.tick),
    };
    game.tick
}

fn genome_table() -> &'static [f32] {
    unsafe { &*GENOMES.ptr() }
}

fn write_state(game: &Game) {
    let state = unsafe { &mut *STATE.ptr() };
    state[..12].copy_from_slice(&game.state_values());
    state[12] = game.a.life as f64;
    state[13] = game.b.life as f64;
    state[14] = game.tick as f64;
    state[15] = game.delay as f64;
    state[16] = game.last_point as f64;
}

// ---- exports ---------------------------------------------------------------

#[cfg_attr(target_arch = "wasm32", unsafe(no_mangle))]
pub extern "C" fn kernel_version() -> u32 {
    KERNEL_VERSION
}

#[cfg_attr(target_arch = "wasm32", unsafe(no_mangle))]
pub extern "C" fn genome_len() -> u32 {
    GENOME_LEN as u32
}

#[cfg_attr(target_arch = "wasm32", unsafe(no_mangle))]
pub extern "C" fn max_ticks_cap() -> u32 {
    MAX_TICKS_CAP
}

#[cfg_attr(target_arch = "wasm32", unsafe(no_mangle))]
pub extern "C" fn table_capacity() -> u32 {
    TABLE_CAPACITY as u32
}

#[cfg_attr(target_arch = "wasm32", unsafe(no_mangle))]
pub extern "C" fn batch_capacity() -> u32 {
    BATCH_CAPACITY as u32
}

#[cfg_attr(target_arch = "wasm32", unsafe(no_mangle))]
pub extern "C" fn scripted_slot() -> u32 {
    SCRIPTED_SLOT
}

#[cfg_attr(target_arch = "wasm32", unsafe(no_mangle))]
pub extern "C" fn genome_table_ptr() -> u32 {
    GENOMES.ptr() as usize as u32
}

#[cfg_attr(target_arch = "wasm32", unsafe(no_mangle))]
pub extern "C" fn jobs_ptr() -> u32 {
    JOBS.ptr() as usize as u32
}

#[cfg_attr(target_arch = "wasm32", unsafe(no_mangle))]
pub extern "C" fn results_ptr() -> u32 {
    RESULTS.ptr() as usize as u32
}

#[cfg_attr(target_arch = "wasm32", unsafe(no_mangle))]
pub extern "C" fn tick_hash_ptr() -> u32 {
    TICK_HASH.ptr() as usize as u32
}

#[cfg_attr(target_arch = "wasm32", unsafe(no_mangle))]
pub extern "C" fn state_ptr() -> u32 {
    STATE.ptr() as usize as u32
}

/// Plays table slot `a` on the left and `b` on the right; fills result
/// block 0, and the tick hashes when `record_hashes` is nonzero.
#[cfg_attr(target_arch = "wasm32", unsafe(no_mangle))]
pub extern "C" fn play_match(seed: u64, a: u32, b: u32, max_ticks: u32, record_hashes: u32) -> u32 {
    let results = unsafe { &mut *RESULTS.ptr() };
    let tick_hashes = unsafe { &mut *TICK_HASH.ptr() };
    let hashes = if record_hashes != 0 {
        Some(&mut tick_hashes[..])
    } else {
        None
    };
    run_match(
        genome_table(),
        seed,
        a,
        b,
        max_ticks,
        hashes,
        &mut results[0],
    )
}

/// Runs the first `n` jobs in order; stops at the first bad job, whose block
/// carries flag bit 2, and returns how many completed before it.
#[cfg_attr(target_arch = "wasm32", unsafe(no_mangle))]
pub extern "C" fn play_batch(n: u32, max_ticks: u32) -> u32 {
    if n as usize > BATCH_CAPACITY {
        return 0;
    }
    let jobs = unsafe { &*JOBS.ptr() };
    let results = unsafe { &mut *RESULTS.ptr() };
    for i in 0..n as usize {
        let job = jobs[i];
        run_match(
            genome_table(),
            job.seed,
            job.a,
            job.b,
            max_ticks,
            None,
            &mut results[i],
        );
        if results[i].flags & FLAG_BAD_JOB != 0 {
            return i as u32;
        }
    }
    n
}

#[cfg_attr(target_arch = "wasm32", unsafe(no_mangle))]
pub extern "C" fn reset(seed: u64) {
    let game = unsafe { &mut *GAME.ptr() };
    *game = Game::new(seed);
    write_state(game);
}

#[cfg_attr(target_arch = "wasm32", unsafe(no_mangle))]
pub extern "C" fn set_actions(a: u32, b: u32) {
    unsafe { &mut *GAME.ptr() }.set_actions(a, b);
}

#[cfg_attr(target_arch = "wasm32", unsafe(no_mangle))]
pub extern "C" fn step() -> i32 {
    let game = unsafe { &mut *GAME.ptr() };
    let point = game.step();
    write_state(game);
    point
}

#[cfg_attr(target_arch = "wasm32", unsafe(no_mangle))]
pub extern "C" fn is_done(max_ticks: u32) -> u32 {
    unsafe { &*GAME.ptr() }.is_done(max_ticks) as u32
}

/// The action genome `slot` (or the anchor, for `scripted_slot()`) would take
/// now for `side` (0 = A, 1 = B). 0xFFFFFFFF for a bad side or slot.
#[cfg_attr(target_arch = "wasm32", unsafe(no_mangle))]
pub extern "C" fn policy_action(side: u32, slot: u32) -> u32 {
    let game = unsafe { &*GAME.ptr() };
    let obs = match side {
        0 => &game.a.obs,
        1 => &game.b.obs,
        _ => return u32::MAX,
    };
    match Controller::from_slot(genome_table(), slot) {
        Some(controller) => controller.action(obs),
        None => u32::MAX,
    }
}

#[cfg_attr(target_arch = "wasm32", unsafe(no_mangle))]
pub extern "C" fn scripted_action(side: u32) -> u32 {
    policy_action(side, SCRIPTED_SLOT)
}

#[cfg_attr(target_arch = "wasm32", unsafe(no_mangle))]
pub extern "C" fn state_hash() -> u64 {
    unsafe { &*GAME.ptr() }.hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, MutexGuard};

    /// The exports share static buffers; tests that call them take turns.
    fn exports_lock() -> MutexGuard<'static, ()> {
        static LOCK: Mutex<()> = Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn load(slot: usize, params: &[f32]) {
        let table = unsafe { &mut *GENOMES.ptr() };
        table[slot * GENOME_LEN..(slot + 1) * GENOME_LEN].copy_from_slice(params);
    }

    fn result(i: usize) -> MatchResult {
        let results = unsafe { &*RESULTS.ptr() };
        results[i]
    }

    fn tick_hash(i: usize) -> u64 {
        let hashes = unsafe { &*TICK_HASH.ptr() };
        hashes[i]
    }

    #[test]
    fn zero_genomes_play_a_full_match() {
        let _guard = exports_lock();
        load(0, &[0.0; GENOME_LEN]);
        load(1, &[0.0; GENOME_LEN]);
        let ticks = play_match(42, 0, 1, MAX_TICKS_CAP, 1);
        let r = result(0);
        assert_eq!(r.ticks, ticks);
        assert!(ticks > 0 && ticks <= MAX_TICKS_CAP);
        let ended_by_lives = r.points_a == 5 || r.points_b == 5;
        assert!(ended_by_lives || r.flags & FLAG_MAX_TICKS != 0);
        assert_eq!(r.flags & (FLAG_NAN | FLAG_BAD_JOB), 0);
        assert_eq!(tick_hash(ticks as usize - 1), r.final_hash);
        assert_eq!(r.desc_a[7], r.points_a as f64);
    }

    #[test]
    fn same_seed_same_match_and_batch_agrees() {
        let _guard = exports_lock();
        load(0, &[0.0; GENOME_LEN]);
        let mut busy = [0.0f32; GENOME_LEN];
        busy[512] = 1.0; // always forward
        busy[514] = 1.0; // always jump
        load(1, &busy);
        play_match(7, 0, 1, 3000, 0);
        let first = result(0);
        play_match(7, 0, 1, 3000, 0);
        assert_eq!(result(0), first);

        let jobs = unsafe { &mut *JOBS.ptr() };
        jobs[0] = Job {
            seed: 7,
            a: 0,
            b: 1,
        };
        jobs[1] = Job {
            seed: 8,
            a: 1,
            b: SCRIPTED_SLOT,
        };
        jobs[2] = Job {
            seed: 9,
            a: SCRIPTED_SLOT,
            b: SCRIPTED_SLOT,
        };
        assert_eq!(play_batch(3, 3000), 3);
        assert_eq!(result(0), first);
        play_match(8, 1, SCRIPTED_SLOT, 3000, 0);
        let scripted = result(0);
        play_batch(2, 3000);
        assert_eq!(result(1), scripted);
    }

    #[test]
    fn bad_jobs_are_flagged() {
        let _guard = exports_lock();
        assert_eq!(play_match(1, TABLE_CAPACITY as u32, 0, 3000, 0), 0);
        assert_eq!(result(0), MatchResult::BAD_JOB);
        assert_eq!(play_match(1, 0, 0, 0, 0), 0);
        assert_eq!(result(0).flags, FLAG_BAD_JOB);
        assert_eq!(play_match(1, 0, 0, MAX_TICKS_CAP + 1, 0), 0);
        assert_eq!(result(0).flags, FLAG_BAD_JOB);

        let jobs = unsafe { &mut *JOBS.ptr() };
        jobs[0] = Job {
            seed: 1,
            a: 0,
            b: 0,
        };
        jobs[1] = Job {
            seed: 1,
            a: 0,
            b: 5000,
        };
        jobs[2] = Job {
            seed: 1,
            a: 0,
            b: 0,
        };
        assert_eq!(play_batch(3, 100), 1);
        assert_eq!(result(1).flags, FLAG_BAD_JOB);
        assert_eq!(play_batch(BATCH_CAPACITY as u32 + 1, 100), 0);
        assert_eq!(policy_action(2, 0), u32::MAX);
        assert_eq!(policy_action(0, 5000), u32::MAX);
    }

    #[test]
    fn short_match_ends_on_max_ticks() {
        let _guard = exports_lock();
        load(0, &[0.0; GENOME_LEN]);
        assert_eq!(play_match(3, 0, 0, 10, 0), 10);
        assert_eq!(result(0).flags, FLAG_MAX_TICKS);
    }

    #[test]
    fn interactive_game_replays_a_match() {
        let _guard = exports_lock();
        let mut busy = [0.0f32; GENOME_LEN];
        busy[512] = 0.5;
        busy[514] = 1.0;
        load(3, &busy);
        play_match(99, 3, SCRIPTED_SLOT, 3000, 0);
        let expected = result(0);

        reset(99);
        while is_done(3000) == 0 {
            set_actions(policy_action(0, 3), scripted_action(1));
            step();
        }
        assert_eq!(state_hash(), expected.final_hash);
        let state = unsafe { &*STATE.ptr() };
        assert_eq!(state[14], expected.ticks as f64);
        assert_eq!(state[12], (5 - expected.points_b) as f64);
        assert_eq!(state[13], (5 - expected.points_a) as f64);
    }
}
