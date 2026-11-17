#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Spieler {
    X, // Maximierer (+10)
    O, // Minimierer (-10)
}

impl Spieler {
    pub fn gegner(self) -> Self {
        match self {
            Spieler::X => Spieler::O,
            Spieler::O => Spieler::X,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Brett {
    pub felder: [Option<Spieler>; 9],
}

impl Brett {
    pub fn leer() -> Self {
        Self { felder: [None; 9] }
    }

    pub fn gewinner(&self) -> Option<Spieler> {
        const LINIEN: [[usize; 3]; 8] = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8], // Zeilen
            [0, 3, 6], [1, 4, 7], [2, 5, 8], // Spalten
            [0, 4, 8], [2, 4, 6],            // Diagonalen
        ];
        for [a, b, c] in LINIEN {
            if let (Some(s1), Some(s2), Some(s3)) = (self.felder[a], self.felder[b], self.felder[c]) {
                if s1 == s2 && s2 == s3 {
                    return Some(s1);
                }
            }
        }
        None
    }

    pub fn ist_voll(&self) -> bool {
        self.felder.iter().all(|f| f.is_some())
    }

    pub fn freie_zuege(&self) -> Vec<usize> {
        self.felder
            .iter()
            .enumerate()
            .filter_map(|(i, f)| if f.is_none() { Some(i) } else { None })
            .collect()
    }
}

/// Minimax ohne Schnitt (zum Vergleich der besuchten Spielbaum-Knoten).
pub fn minimax_klassisch(brett: &mut Brett, tiefe: i32, ist_max: bool, knoten: &mut usize) -> i32 {
    *knoten += 1;
    if let Some(g) = brett.gewinner() {
        return match g {
            Spieler::X => 10 - tiefe,
            Spieler::O => tiefe - 10,
        };
    }
    if brett.ist_voll() {
        return 0;
    }

    if ist_max {
        let mut best = i32::MIN;
        for zug in brett.freie_zuege() {
            brett.felder[zug] = Some(Spieler::X);
            best = best.max(minimax_klassisch(brett, tiefe + 1, false, knoten));
            brett.felder[zug] = None;
        }
        best
    } else {
        let mut best = i32::MAX;
        for zug in brett.freie_zuege() {
            brett.felder[zug] = Some(Spieler::O);
            best = best.min(minimax_klassisch(brett, tiefe + 1, true, knoten));
            brett.felder[zug] = None;
        }
        best
    }
}

/// Minimax mit Alpha-Beta-Pruning: schneidet aussichtslose Äste sofort ab.
pub fn minimax_alpha_beta(
    brett: &mut Brett,
    tiefe: i32,
    mut alpha: i32,
    mut beta: i32,
    ist_max: bool,
    knoten: &mut usize,
) -> i32 {
    *knoten += 1;
    if let Some(g) = brett.gewinner() {
        return match g {
            Spieler::X => 10 - tiefe,
            Spieler::O => tiefe - 10,
        };
    }
    if brett.ist_voll() {
        return 0;
    }

    if ist_max {
        let mut best = i32::MIN;
        for zug in brett.freie_zuege() {
            brett.felder[zug] = Some(Spieler::X);
            best = best.max(minimax_alpha_beta(brett, tiefe + 1, alpha, beta, false, knoten));
            brett.felder[zug] = None;
            alpha = alpha.max(best);
            if beta <= alpha {
                break; // Beta-Cutoff!
            }
        }
        best
    } else {
        let mut best = i32::MAX;
        for zug in brett.freie_zuege() {
            brett.felder[zug] = Some(Spieler::O);
            best = best.min(minimax_alpha_beta(brett, tiefe + 1, alpha, beta, true, knoten));
            brett.felder[zug] = None;
            beta = beta.min(best);
            if beta <= alpha {
                break; // Alpha-Cutoff!
            }
        }
        best
    }
}

pub fn bester_zug(brett: &mut Brett, spieler: Spieler) -> Option<usize> {
    let ist_max = spieler == Spieler::X;
    let mut best_zug = None;
    let mut best_wert = if ist_max { i32::MIN } else { i32::MAX };
    let mut knoten = 0;

    for zug in brett.freie_zuege() {
        brett.felder[zug] = Some(spieler);
        let wert = minimax_alpha_beta(brett, 0, i32::MIN, i32::MAX, !ist_max, &mut knoten);
        brett.felder[zug] = None;
        if (ist_max && wert > best_wert) || (!ist_max && wert < best_wert) {
            best_wert = wert;
            best_zug = Some(zug);
        }
    }
    best_zug
}

fn main() {
    let mut brett = Brett::leer();
    let mut k_klassisch = 0;
    let mut k_ab = 0;
    let _ = minimax_klassisch(&mut brett, 0, true, &mut k_klassisch);
    let _ = minimax_alpha_beta(&mut brett, 0, i32::MIN, i32::MAX, true, &mut k_ab);

    println!("Leeres Tic-Tac-Toe-Brett:");
    println!("  Minimax klassisch:   {:6} Knoten geprüft", k_klassisch);
    println!("  Alpha-Beta-Pruning:  {:6} Knoten geprüft", k_ab);

    let zug = bester_zug(&mut brett, Spieler::X);
    println!("Empfohlener Eröffnungszug für X: Feld {:?}", zug);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimax_gewinnt_sofort_wenn_moeglich() {
        let mut brett = Brett {
            felder: [
                Some(Spieler::X), Some(Spieler::X), None,
                Some(Spieler::O), Some(Spieler::O), None,
                None, None, None,
            ],
        };
        assert_eq!(bester_zug(&mut brett, Spieler::X), Some(2));
    }

    #[test]
    fn minimax_blockt_gegner_sieg() {
        let mut brett = Brett {
            felder: [
                Some(Spieler::O), Some(Spieler::O), None,
                Some(Spieler::X), None, None,
                None, None, Some(Spieler::X),
            ],
        };
        assert_eq!(bester_zug(&mut brett, Spieler::X), Some(2));
    }

    #[test]
    fn alpha_beta_pruning_spart_knoten() {
        let mut brett = Brett::leer();
        let mut k_klassisch = 0;
        let mut k_ab = 0;
        let v1 = minimax_klassisch(&mut brett, 0, true, &mut k_klassisch);
        let v2 = minimax_alpha_beta(&mut brett, 0, i32::MIN, i32::MAX, true, &mut k_ab);
        assert_eq!(v1, v2);
        assert!(k_ab < k_klassisch / 5);
    }
}

