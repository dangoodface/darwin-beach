//! Native harness for the oracle exploration (plan B4): one rung against a table of genomes
//! and the scripted bot on the given seeds, both sides, with the official board's Elo fit.
//!
//!     cargo +1.95.0 run --release --example oracle_eval -- <dir> <rung> [seeds] [threads] [max_ticks]
//!
//! `<dir>` holds members.f32 (little-endian f32 rows of GENOME_LEN), members.txt (one
//! "rating<TAB>label" line per row) and seeds.txt (one u64 a line). Exploration only:
//! the official ratings come from the wasm build through Python.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use slime_kernel::policy::GENOME_LEN;
use slime_kernel::{MatchResult, ORACLE_SLOT_BASE, SCRIPTED_SLOT, run_match};

fn expected(r: f64, opp: f64) -> f64 {
    1.0 / (1.0 + 10f64.powf((opp - r) / 400.0))
}

/// `arena.ladder.elo_mle`: one virtual draw against each opponent, bisection.
fn elo_mle(recs: &[(f64, f64, u32)]) -> f64 {
    let target: f64 = recs.iter().map(|&(_, s, _)| s + 0.5).sum();
    let mut lo = recs.iter().map(|r| r.0).fold(f64::INFINITY, f64::min) - 4000.0;
    let mut hi = recs.iter().map(|r| r.0).fold(f64::NEG_INFINITY, f64::max) + 4000.0;
    while hi - lo > 1e-9 {
        let mid = 0.5 * (lo + hi);
        let total: f64 = recs
            .iter()
            .map(|&(r, _, n)| (n as f64 + 1.0) * expected(mid, r))
            .sum();
        if total < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = std::path::Path::new(&args[1]);
    let rung: u32 = args[2].parse().unwrap();
    let n_seeds: usize = args.get(3).map_or(64, |s| s.parse().unwrap());
    let threads: usize = args.get(4).map_or(4, |s| s.parse().unwrap());
    let max_ticks: u32 = args.get(5).map_or(6000, |s| s.parse().unwrap());

    let bytes = std::fs::read(dir.join("members.f32")).unwrap();
    let table: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let mut members: Vec<(f64, String, u32)> = std::fs::read_to_string(dir.join("members.txt"))
        .unwrap()
        .lines()
        .enumerate()
        .map(|(i, l)| {
            let (r, label) = l.split_once('\t').unwrap();
            (r.parse().unwrap(), label.to_string(), i as u32)
        })
        .collect();
    assert_eq!(table.len(), members.len() * GENOME_LEN);
    members.insert(0, (1000.0, "the coach".to_string(), SCRIPTED_SLOT));
    let seeds: Vec<u64> = std::fs::read_to_string(dir.join("seeds.txt"))
        .unwrap()
        .lines()
        .take(n_seeds)
        .map(|l| l.trim().parse().unwrap())
        .collect();

    let oracle = ORACLE_SLOT_BASE + rung;
    let jobs: Vec<(usize, u64, bool)> = (0..members.len())
        .flat_map(|m| {
            seeds
                .iter()
                .flat_map(move |&s| [(m, s, false), (m, s, true)])
        })
        .collect();
    let next = AtomicUsize::new(0);
    let start = Instant::now();
    let mut results: Vec<(usize, i64, u32)> = Vec::with_capacity(jobs.len()); // (member, margin, ticks)
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut mine = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(&(m, seed, swap)) = jobs.get(i) else {
                            break;
                        };
                        let slot = members[m].2;
                        let (a, b) = if swap { (slot, oracle) } else { (oracle, slot) };
                        let mut out = MatchResult::BAD_JOB;
                        run_match(&table, seed, a, b, max_ticks, 1, 1, None, &mut out);
                        let (me, them) = if swap {
                            (out.points_b, out.points_a)
                        } else {
                            (out.points_a, out.points_b)
                        };
                        mine.push((m, me as i64 - them as i64, out.ticks));
                    }
                    mine
                })
            })
            .collect();
        for h in handles {
            results.extend(h.join().unwrap());
        }
    });
    let secs = start.elapsed().as_secs_f64();
    let ticks: u64 = results.iter().map(|r| r.2 as u64).sum();

    let mut recs = Vec::new();
    for (m, (rating, label, _)) in members.iter().enumerate() {
        let mine: Vec<_> = results.iter().filter(|r| r.0 == m).collect();
        let won = mine.iter().filter(|r| r.1 > 0).count();
        let lost = mine.iter().filter(|r| r.1 < 0).count();
        let tied = mine.len() - won - lost;
        let margin = mine.iter().map(|r| r.1 as f64).sum::<f64>() / mine.len() as f64;
        let score = won as f64 + 0.5 * tied as f64;
        recs.push((*rating, score, mine.len() as u32));
        println!(
            "{label:20} {rating:6.0}  won {won:3} lost {lost:3} tied {tied:3}  score {:.2}  margin {margin:+.2}",
            score / mine.len() as f64
        );
    }
    println!(
        "rung {rung}: Elo {:.0} over {} matches; {:.1} s, {:.2} M ticks, {:.0} ticks a match",
        elo_mle(&recs),
        results.len(),
        secs,
        ticks as f64 / 1e6,
        ticks as f64 / results.len() as f64
    );
}
