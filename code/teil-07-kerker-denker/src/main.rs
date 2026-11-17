use std::time::Instant;

const BREITE: usize = 5;
const HOEHE: usize = 5;

const KARTE: [&str; HOEHE] = [
    "S..#.",
    ".#F..",
    ".#.#.",
    "...F.",
    "##..Z",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aktion {
    Hoch = 0,
    Runter = 1,
    Links = 2,
    Rechts = 3,
}

const ALLE_AKTIONEN: [Aktion; 4] = [Aktion::Hoch, Aktion::Runter, Aktion::Links, Aktion::Rechts];

#[derive(Clone)]
struct Lcg(u64);

impl Lcg {
    fn neu(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 32) as u32
    }

    fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }
}

pub fn feld_zeichen(x: usize, y: usize) -> char {
    KARTE[y].chars().nth(x).unwrap_or('#')
}

pub fn schritt(x: usize, y: usize, aktion: Aktion) -> (usize, usize, f32, bool) {
    let (nx, ny) = match aktion {
        Aktion::Hoch if y > 0 => (x, y - 1),
        Aktion::Runter if y + 1 < HOEHE => (x, y + 1),
        Aktion::Links if x > 0 => (x - 1, y),
        Aktion::Rechts if x + 1 < BREITE => (x + 1, y),
        _ => return (x, y, -2.0, false),
    };
    match feld_zeichen(nx, ny) {
        '#' => (x, y, -2.0, false),
        'F' => (nx, ny, -50.0, true),
        'Z' => (nx, ny, 100.0, true),
        _ => (nx, ny, -1.0, false),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct QTabelle(pub [[f32; 4]; BREITE * HOEHE]);

impl QTabelle {
    pub fn trainiere(episoden: usize, seed: u64) -> Self {
        let mut q = [[0.0_f32; 4]; BREITE * HOEHE];
        let mut rng = Lcg::neu(seed);
        let alpha = 0.2;
        let gamma = 0.9;

        for ep in 0..episoden {
            let eps = (0.9 - (ep as f32 / episoden as f32) * 0.85).max(0.05);
            let (mut x, mut y) = (0, 0);
            for _ in 0..80 {
                let s = y * BREITE + x;
                let a = if rng.next_f32() < eps {
                    ALLE_AKTIONEN[(rng.next_u32() as usize) % 4]
                } else {
                    beste_aktion(&q[s])
                };
                let (nx, ny, r, ende) = schritt(x, y, a);
                let ns = ny * BREITE + nx;
                let max_next = if ende {
                    0.0
                } else {
                    q[ns].iter().copied().fold(f32::NEG_INFINITY, f32::max)
                };
                let ai = a as usize;
                q[s][ai] += alpha * (r + gamma * max_next - q[s][ai]);
                x = nx;
                y = ny;
                if ende {
                    break;
                }
            }
        }
        Self(q)
    }

    pub fn policy_karte(&self) -> Vec<String> {
        let mut zeilen = Vec::with_capacity(HOEHE);
        for y in 0..HOEHE {
            let mut zeile = String::new();
            for x in 0..BREITE {
                let ch = match feld_zeichen(x, y) {
                    '#' | 'F' | 'Z' => feld_zeichen(x, y),
                    _ => match beste_aktion(&self.0[y * BREITE + x]) {
                        Aktion::Hoch => '^',
                        Aktion::Runter => 'v',
                        Aktion::Links => '<',
                        Aktion::Rechts => '>',
                    },
                };
                zeile.push(ch);
            }
            zeilen.push(zeile);
        }
        zeilen
    }
}

fn beste_aktion(werte: &[f32; 4]) -> Aktion {
    let mut best = 0;
    for i in 1..4 {
        if werte[i] > werte[best] {
            best = i;
        }
    }
    ALLE_AKTIONEN[best]
}

#[derive(Debug, Clone)]
pub struct BenchmarkErgebnis {
    pub erfolgsquote: f32,
    pub durchschnitt_schritte: f32,
}

pub fn bewerte_zufalls_agent(runden: usize, seed: u64) -> BenchmarkErgebnis {
    let mut rng = Lcg::neu(seed);
    let mut siege = 0usize;
    let mut schritte_gesamt = 0usize;

    for _ in 0..runden {
        let (mut x, mut y) = (0, 0);
        for schritt_nr in 1..=60 {
            let a = ALLE_AKTIONEN[(rng.next_u32() as usize) % 4];
            let (nx, ny, _, ende) = schritt(x, y, a);
            x = nx;
            y = ny;
            if ende {
                if feld_zeichen(x, y) == 'Z' {
                    siege += 1;
                    schritte_gesamt += schritt_nr;
                }
                break;
            }
        }
    }

    BenchmarkErgebnis {
        erfolgsquote: siege as f32 / runden as f32,
        durchschnitt_schritte: if siege > 0 {
            schritte_gesamt as f32 / siege as f32
        } else {
            60.0
        },
    }
}

pub fn bewerte_q_agent(tabelle: &QTabelle, runden: usize) -> BenchmarkErgebnis {
    let mut siege = 0usize;
    let mut schritte_gesamt = 0usize;

    for _ in 0..runden {
        let (mut x, mut y) = (0, 0);
        for schritt_nr in 1..=30 {
            let a = beste_aktion(&tabelle.0[y * BREITE + x]);
            let (nx, ny, _, ende) = schritt(x, y, a);
            x = nx;
            y = ny;
            if ende {
                if feld_zeichen(x, y) == 'Z' {
                    siege += 1;
                    schritte_gesamt += schritt_nr;
                }
                break;
            }
        }
    }

    BenchmarkErgebnis {
        erfolgsquote: siege as f32 / runden as f32,
        durchschnitt_schritte: if siege > 0 {
            schritte_gesamt as f32 / siege as f32
        } else {
            30.0
        },
    }
}

fn main() {
    let start = Instant::now();
    let tabelle = QTabelle::trainiere(600, 42);
    let dauer = start.elapsed();

    let zufall = bewerte_zufalls_agent(200, 99);
    let q_res = bewerte_q_agent(&tabelle, 200);

    println!("Training (600 Episoden) abgeschlossen in {:?}", dauer);
    println!("Zufalls-Agent:   {:.1} % Siege (Ø {:.1} Schritte)", zufall.erfolgsquote * 100.0, zufall.durchschnitt_schritte);
    println!("Q-Learning-Agent:{:.1} % Siege (Ø {:.1} Schritte)", q_res.erfolgsquote * 100.0, q_res.durchschnitt_schritte);
    println!("\nGelernte Policy-Karte:");
    for z in tabelle.policy_karte() {
        println!("  {}", z);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gleicher_seed_liefert_identische_q_tabelle() {
        let a = QTabelle::trainiere(400, 42);
        let b = QTabelle::trainiere(400, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn q_agent_schlaegt_zufalls_agent() {
        let tabelle = QTabelle::trainiere(600, 42);
        let zufall = bewerte_zufalls_agent(200, 99);
        let q_res = bewerte_q_agent(&tabelle, 200);
        assert!(q_res.erfolgsquote > zufall.erfolgsquote);
        assert!((q_res.erfolgsquote - 1.0).abs() < 1e-6);
    }

    #[test]
    fn policy_karte_hat_richtige_groesse() {
        let tabelle = QTabelle::trainiere(400, 42);
        let karte = tabelle.policy_karte();
        assert_eq!(karte.len(), HOEHE);
        assert!(karte.iter().all(|z| z.len() == BREITE));
    }
}
