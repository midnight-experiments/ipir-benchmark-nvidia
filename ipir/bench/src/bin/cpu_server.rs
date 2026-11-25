//! Task 3 on the CPU backend: resident tables that answer calls of 32 requests on the CPU, on command, while
//! `block_worker` compiles chain blocks in another process (`bench/budget.ts --answer cpu` drives both).
//!
//!   cpu_server <rows>x<cols>,<rows>x<cols>,...
//!
//! Builds one table per shape with `NativeServer::build_with_concurrency(setup, cells, 1)` (xorshift cells), and 32
//! synthetic requests for each. Prints `{"ready":...}` once, then reads commands on stdin, one per line:
//!
//! - `call <id> <table>`: answers the table's 32 requests, one after another (`NativeServer::respond`), as a task on
//!   Rayon's global pool (all logical cores, normal priority); calls run concurrently when they overlap. Prints
//!   `{"done":<id>,"ms":<time from the command to the answer>}`;
//! - `cancel`: calls whose work has not started yet are dropped; each prints `{"done":<id>,"cancelled":true}`;
//! - `quit`.
use ipir_bench::*;
use ipir_sp::native::NativeServer;
use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: cpu_server <rows>x<cols>,...");
        std::process::exit(2);
    }
    let shapes: Vec<(usize, usize)> = args[0]
        .split(',')
        .map(|s| {
            let (r, c) = s.split_once('x').expect("ROWSxCOLS");
            (r.parse().unwrap(), c.parse().unwrap())
        })
        .collect();
    let mut builds = Vec::new();
    let tables: Vec<(NativeServer, Vec<u8>, usize)> = shapes
        .iter()
        .enumerate()
        .map(|(i, &(rows, cols))| {
            let mut rng = Xorshift::new(0xb10c ^ (i as u32).wrapping_mul(7919) ^ rows as u32 ^ cols as u32);
            let cells = xorshift_cells(rows, cols, &mut rng);
            let setup = setup_cols(rows, cols);
            let id = setup.id();
            let t = Instant::now();
            let server = NativeServer::build_with_concurrency(setup, cells, 1).unwrap();
            builds.push(format!("{:.0}", ms(t.elapsed())));
            let each = request_bytes(rows);
            let requests = synthetic_requests(each, 32, &id, &mut rng);
            (server, requests, each)
        })
        .collect();
    let tables = Arc::new(tables);
    let out = Arc::new(Mutex::new(std::io::stdout()));
    let say = |out: &Mutex<std::io::Stdout>, s: String| {
        let mut o = out.lock().unwrap();
        writeln!(o, "{s}").unwrap();
        o.flush().unwrap();
    };
    say(
        &out,
        format!(
            "{{\"ready\":true,\"shapes\":{:?},\"build_ms\":[{}],\"logical_cpus\":{},\"global_threads\":{}}}",
            shapes.iter().map(|(r, c)| format!("{r}x{c}")).collect::<Vec<_>>(),
            builds.join(","),
            logical_cpus(),
            rayon::current_num_threads()
        ),
    );
    let generation = Arc::new(AtomicU64::new(0));
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        let words: Vec<&str> = line.split_whitespace().collect();
        match words.as_slice() {
            ["quit"] | [] => break,
            ["cancel"] => {
                generation.fetch_add(1, Ordering::SeqCst);
            }
            ["call", id, table] => {
                let id: u64 = id.parse().unwrap();
                let table: usize = table.parse().unwrap();
                let (tables, out, generation) = (tables.clone(), out.clone(), generation.clone());
                let gen = generation.load(Ordering::SeqCst);
                let t = Instant::now();
                rayon::spawn(move || {
                    if generation.load(Ordering::SeqCst) != gen {
                        say(&out, format!("{{\"done\":{id},\"cancelled\":true}}"));
                        return;
                    }
                    let (server, requests, each) = &tables[table];
                    for r in requests.chunks_exact(*each) {
                        server.respond(r).unwrap();
                    }
                    say(&out, format!("{{\"done\":{id},\"ms\":{:.3}}}", ms(t.elapsed())));
                });
            }
            other => {
                eprintln!("cpu_server: unknown command {other:?}");
                std::process::exit(2);
            }
        }
    }
}
