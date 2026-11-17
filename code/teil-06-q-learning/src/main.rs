const BREITE: usize = 5;
const HOEHE: usize = 5;

/// S = Start (0,0), . = Frei, # = Mauer, F = Falle, Z = Ziel (4,4)
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

pub fn zustand_id(x: usize, y: usize) -> usize {
    y * BREITE + x
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

pub fn bellman_update(q_aktuell: f32, belohnung: f32, max_naechstes_q: f32, alpha: f32, gamma: f32) -> f32 {
    q_aktuell + alpha * (belohnung + gamma * max_naechstes_q - q_aktuell)
}

pub struct QAgent {
    pub q_tabelle: [[f32; 4]; BREITE * HOEHE],
}

impl QAgent {
    pub fn neu() -> Self {
        Self {
            q_tabelle: [[0.0; 4]; BREITE * HOEHE],
        }
    }

    pub fn beste_aktion(&self, x: usize, y: usize) -> Aktion {
        let s = zustand_id(x, y);
        let mut best_idx = 0;
        for a in 1..4 {
            if self.q_tabelle[s][a] > self.q_tabelle[s][best_idx] {
                best_idx = a;
            }
        }
        ALLE_AKTIONEN[best_idx]
    }

    pub fn trainiere(&mut self, episoden: usize, seed: u64) {
        let mut rng = Lcg::neu(seed);
        let alpha = 0.2;
        let gamma = 0.9;

        for ep in 0..episoden {
            let epsilon = (0.9 - (ep as f32 / episoden as f32) * 0.85).max(0.05);
            let (mut x, mut y) = (0, 0);

            for _ in 0..80 {
                let aktion = if rng.next_f32() < epsilon {
                    ALLE_AKTIONEN[(rng.next_u32() as usize) % 4]
                } else {
                    self.beste_aktion(x, y)
                };

                let (nx, ny, belohnung, ende) = schritt(x, y, aktion);
                let s = zustand_id(x, y);
                let ns = zustand_id(nx, ny);
                let max_next = if ende {
                    0.0
                } else {
                    self.q_tabelle[ns].iter().copied().fold(f32::NEG_INFINITY, f32::max)
                };

                let a_idx = aktion as usize;
                self.q_tabelle[s][a_idx] =
                    bellman_update(self.q_tabelle[s][a_idx], belohnung, max_next, alpha, gamma);

                x = nx;
                y = ny;
                if ende {
                    break;
                }
            }
        }
    }

    pub fn finde_pfad(&self) -> Vec<(usize, usize)> {
        let mut pfad = vec![(0, 0)];
        let (mut x, mut y) = (0, 0);
        for _ in 0..30 {
            if feld_zeichen(x, y) == 'Z' || feld_zeichen(x, y) == 'F' {
                break;
            }
            let aktion = self.beste_aktion(x, y);
            let (nx, ny, _, ende) = schritt(x, y, aktion);
            pfad.push((nx, ny));
            x = nx;
            y = ny;
            if ende {
                break;
            }
        }
        pfad
    }
}

fn main() {
    let mut agent = QAgent::neu();
    agent.trainiere(600, 42);
    let pfad = agent.finde_pfad();
    println!("Gelernter Pfad ({} Schritte): {:?}", pfad.len().saturating_sub(1), pfad);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bellman_update_steigt_bei_belohnung() {
        let neu = bellman_update(0.0, 100.0, 0.0, 0.2, 0.9);
        assert!((neu - 20.0).abs() < 1e-6);
    }

    #[test]
    fn q_agent_findet_sicheren_weg_zum_ziel() {
        let mut agent = QAgent::neu();
        agent.trainiere(600, 42);
        let pfad = agent.finde_pfad();
        assert_eq!(pfad.last(), Some(&(4, 4)));
        assert!(pfad.len() <= 11);
    }

    #[test]
    fn falle_wird_vermieden() {
        let mut agent = QAgent::neu();
        agent.trainiere(600, 42);
        let pfad = agent.finde_pfad();
        for &(x, y) in &pfad {
            assert_ne!(feld_zeichen(x, y), 'F');
        }
    }
}
