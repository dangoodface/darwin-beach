//! How the oracle (side A) wins and loses points against a genome from the harness table
//! (side B): every point classified by whose touch came last, over the first seeds.
//!
//!     cargo +1.95.0 run --release --example oracle_trace -- <dir> <rung> <member row> <seeds> [v]
//!
//! With `v`, every touch and point of every match is printed too.

use slime_kernel::oracle::Oracle;
use slime_kernel::physics::Game;
use slime_kernel::policy::{GENOME_LEN, net_action, upcast};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = std::path::Path::new(&args[1]);
    let rung: usize = args[2].parse().unwrap();
    let row: usize = args[3].parse().unwrap();
    let n_seeds: usize = args[4].parse().unwrap();
    let verbose = args.get(5).is_some_and(|v| v == "v");
    let bytes = std::fs::read(dir.join("members.f32")).unwrap();
    let table: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let net = upcast(&table[row * GENOME_LEN..(row + 1) * GENOME_LEN]);
    let seeds: Vec<u64> = std::fs::read_to_string(dir.join("seeds.txt"))
        .unwrap()
        .lines()
        .take(n_seeds)
        .map(|l| l.trim().parse().unwrap())
        .collect();
    // [won, lost] x [last touch: none, ours, theirs]
    let mut tally = [[0u32; 3]; 2];
    let mut lost_after_their_touch = Vec::new(); // (ticks from their touch to the point, landing x, our x)
    for &seed in &seeds {
        let mut game = Game::new(seed);
        let mut oracle = Oracle::new(rung, seed, 0);
        let (mut last, mut last_tick) = (0usize, 0u32);
        while !game.is_done(6000) {
            let a = oracle.action(&game.a.obs);
            let b = net_action(&net, &game.b.obs);
            game.set_actions(a, b);
            let hits = (game.desc[0].hits, game.desc[1].hits);
            let (vx, vy, ball) = (game.ball.vx, game.ball.vy, (game.ball.x, game.ball.y));
            let point = game.step();
            for (side, touched) in [
                (1, game.desc[0].hits > hits.0),
                (2, game.desc[1].hits > hits.1),
            ] {
                if touched {
                    (last, last_tick) = (side, game.tick);
                    if verbose {
                        println!(
                            "{:5} touch {} ball ({:6.2},{:5.2}) v ({:6.2},{:6.2}) -> ({:6.2},{:6.2})  A x {:6.2} y {:4.2} vx {:6.2}  B x {:6.2} y {:4.2}  plans {} chose {:?}",
                            game.tick,
                            if side == 1 { "A" } else { "B" },
                            ball.0,
                            ball.1,
                            vx,
                            vy,
                            game.ball.vx,
                            game.ball.vy,
                            -game.a.x,
                            game.a.y,
                            -game.a.vx,
                            game.b.x,
                            game.b.y,
                            oracle.replans(),
                            oracle.choice()
                        );
                    }
                }
            }
            if point != 0 {
                let ours = point > 0; // `step` returns +1 when A wins the point
                tally[if ours { 0 } else { 1 }][last] += 1;
                if !ours && last == 2 {
                    lost_after_their_touch.push((game.tick - last_tick, -ball.0, -game.a.x));
                }
                if verbose {
                    println!(
                        "{:5} POINT {} (last touch {}) landed at {:6.2} in A's frame, A at {:6.2}, B at {:6.2}",
                        game.tick,
                        if ours { "oracle" } else { "member" },
                        ["-", "A", "B"][last],
                        -ball.0,
                        -game.a.x,
                        game.b.x
                    );
                }
                last = 0;
            }
        }
        if verbose {
            println!(
                "seed {seed}: {} to {} in {} ticks",
                game.desc[0].points, game.desc[1].points, game.tick
            );
        }
    }
    println!(
        "won: last touch none {} ours {} theirs {}",
        tally[0][0], tally[0][1], tally[0][2]
    );
    println!(
        "lost: last touch none {} ours {} theirs {}",
        tally[1][0], tally[1][1], tally[1][2]
    );
    lost_after_their_touch.sort_by_key(|l| l.0);
    for (ticks, x, me) in lost_after_their_touch {
        println!(
            "  lost {ticks:3} ticks after their touch: landed at {x:6.2}, we were at {me:6.2}"
        );
    }
}
