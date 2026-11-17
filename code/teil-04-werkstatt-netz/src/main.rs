pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn sigmoid_ableitung_aus_y(y: f32) -> f32 {
    y * (1.0 - y)
}

/// Wandelt Rohwerte (Logits) in eine Wahrscheinlichkeitsverteilung (Summe = 1.0) um.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_wert = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_werte: Vec<f32> = logits.iter().map(|&x| (x - max_wert).exp()).collect();
    let summe: f32 = exp_werte.iter().sum();
    exp_werte.iter().map(|&e| e / summe).collect()
}

pub fn argmax(werte: &[f32]) -> usize {
    werte
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

struct Lcg(u64);

impl Lcg {
    fn neu(seed: u64) -> Self {
        Self(seed)
    }

    fn naechstes_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        let norm = ((self.0 >> 32) as f32) / (u32::MAX as f32);
        norm * 2.0 - 1.0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ZiffernNetz {
    pub n_ein: usize,
    pub n_versteckt: usize,
    pub n_aus: usize,
    pub w_ev: Vec<f32>,
    pub b_v: Vec<f32>,
    pub w_va: Vec<f32>,
    pub b_a: Vec<f32>,
}

impl ZiffernNetz {
    pub fn neu(n_ein: usize, n_versteckt: usize, n_aus: usize, seed: u64) -> Self {
        let mut rng = Lcg::neu(seed);
        Self {
            n_ein,
            n_versteckt,
            n_aus,
            w_ev: (0..n_versteckt * n_ein).map(|_| rng.naechstes_f32() * 0.5).collect(),
            b_v: vec![0.0; n_versteckt],
            w_va: (0..n_aus * n_versteckt).map(|_| rng.naechstes_f32() * 0.5).collect(),
            b_a: vec![0.0; n_aus],
        }
    }

    pub fn vorwaerts(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut versteckt = vec![0.0; self.n_versteckt];
        for h in 0..self.n_versteckt {
            let mut s = self.b_v[h];
            for i in 0..self.n_ein {
                s += self.w_ev[h * self.n_ein + i] * x[i];
            }
            versteckt[h] = sigmoid(s);
        }

        let mut ausgabe = vec![0.0; self.n_aus];
        for o in 0..self.n_aus {
            let mut s = self.b_a[o];
            for h in 0..self.n_versteckt {
                s += self.w_va[o * self.n_versteckt + h] * versteckt[h];
            }
            ausgabe[o] = sigmoid(s);
        }
        (versteckt, ausgabe)
    }

    pub fn klassifiziere(&self, x: &[f32]) -> (usize, Vec<f32>) {
        let (_, aus) = self.vorwaerts(x);
        let wahrsch = softmax(&aus);
        (argmax(&wahrsch), wahrsch)
    }

    pub fn trainiere_schritt(&mut self, x: &[f32], ziel_klasse: usize, lernrate: f32) -> f32 {
        let (versteckt, ausgabe) = self.vorwaerts(x);
        let mut ziel = vec![0.0; self.n_aus];
        ziel[ziel_klasse] = 1.0;

        let mut mse = 0.0;
        let mut delta_a = vec![0.0; self.n_aus];
        for o in 0..self.n_aus {
            let fehler = ziel[o] - ausgabe[o];
            mse += fehler * fehler;
            delta_a[o] = fehler * sigmoid_ableitung_aus_y(ausgabe[o]);
        }

        let mut delta_v = vec![0.0; self.n_versteckt];
        for h in 0..self.n_versteckt {
            let mut anteil = 0.0;
            for o in 0..self.n_aus {
                anteil += delta_a[o] * self.w_va[o * self.n_versteckt + h];
            }
            delta_v[h] = anteil * sigmoid_ableitung_aus_y(versteckt[h]);
        }

        for o in 0..self.n_aus {
            for h in 0..self.n_versteckt {
                self.w_va[o * self.n_versteckt + h] += lernrate * delta_a[o] * versteckt[h];
            }
            self.b_a[o] += lernrate * delta_a[o];
        }

        for h in 0..self.n_versteckt {
            for i in 0..self.n_ein {
                self.w_ev[h * self.n_ein + i] += lernrate * delta_v[h] * x[i];
            }
            self.b_v[h] += lernrate * delta_v[h];
        }

        mse / (self.n_aus as f32)
    }

    pub fn genauigkeit(&self, daten: &[([f32; 25], usize)]) -> f32 {
        let korrekt = daten
            .iter()
            .filter(|(bild, ziel)| self.klassifiziere(bild).0 == *ziel)
            .count();
        korrekt as f32 / daten.len() as f32
    }

    /// Serialisiert Gewichte in ein einfaches, lesbares Textformat.
    pub fn als_text(&self) -> String {
        let join_f32 = |v: &[f32]| -> String {
            v.iter().map(|x| format!("{x:.6}")).collect::<Vec<_>>().join(" ")
        };
        format!(
            "{} {} {}\n{}\n{}\n{}\n{}",
            self.n_ein,
            self.n_versteckt,
            self.n_aus,
            join_f32(&self.w_ev),
            join_f32(&self.b_v),
            join_f32(&self.w_va),
            join_f32(&self.b_a),
        )
    }

    pub fn aus_text(text: &str) -> Result<Self, String> {
        let zeilen: Vec<&str> = text.lines().collect();
        if zeilen.len() < 5 {
            return Err("Zu wenige Zeilen im Modelltext".into());
        }
        let dims: Vec<usize> = zeilen[0]
            .split_whitespace()
            .map(|s| s.parse::<usize>().map_err(|e| e.to_string()))
            .collect::<Result<_, _>>()?;
        if dims.len() != 3 {
            return Err("Erste Zeile braucht genau 3 Dimensionen".into());
        }
        let parse_f32s = |z: &str| -> Result<Vec<f32>, String> {
            z.split_whitespace()
                .map(|s| s.parse::<f32>().map_err(|e| e.to_string()))
                .collect()
        };
        Ok(Self {
            n_ein: dims[0],
            n_versteckt: dims[1],
            n_aus: dims[2],
            w_ev: parse_f32s(zeilen[1])?,
            b_v: parse_f32s(zeilen[2])?,
            w_va: parse_f32s(zeilen[3])?,
            b_a: parse_f32s(zeilen[4])?,
        })
    }
}

/// Wandelt ein 5x5-ASCII-Muster ('#' = 1.0, '.' = 0.0) in 25 Eingabewerte um.
pub fn raster(muster: [&str; 5]) -> [f32; 25] {
    let mut werte = [0.0; 25];
    for (z, reihe) in muster.iter().enumerate() {
        for (s, ch) in reihe.chars().take(5).enumerate() {
            werte[z * 5 + s] = if ch == '#' { 1.0 } else { 0.0 };
        }
    }
    werte
}

pub fn trainings_ziffern() -> Vec<([f32; 25], usize)> {
    vec![
        (raster([".###.", "#...#", "#...#", "#...#", ".###."]), 0),
        (raster(["..#..", ".##..", "..#..", "..#..", ".###."]), 1),
        (raster([".###.", "....#", ".###.", "#....", "#####"]), 2),
        (raster(["####.", "....#", ".###.", "....#", "####."]), 3),
        (raster(["#...#", "#...#", "#####", "....#", "....#"]), 4),
    ]
}

pub fn verrauschte_test_ziffern() -> Vec<([f32; 25], usize)> {
    vec![
        (raster([".###.", "#...#", "#..##", "#...#", ".###."]), 0),
        (raster(["..#..", ".##..", "..#..", "..#..", "..#.."]), 1),
        (raster([".###.", "....#", "..##.", "#....", "#####"]), 2),
        (raster(["####.", "....#", "..##.", "....#", "####."]), 3),
        (raster(["#...#", "#..##", "#####", "....#", "....#"]), 4),
    ]
}

fn main() {
    let train = trainings_ziffern();
    let test = verrauschte_test_ziffern();
    let mut netz = ZiffernNetz::neu(25, 16, 5, 1337);

    for epoche in 0..800 {
        let mut verlust = 0.0;
        for (bild, klasse) in &train {
            verlust += netz.trainiere_schritt(bild, *klasse, 0.35);
        }
        if epoche % 200 == 0 {
            println!("Epoche {:3}: Verlust = {:.4}", epoche, verlust / train.len() as f32);
        }
    }

    println!(
        "\nGenauigkeit auf verrauschten Test-Ziffern: {:.0} %",
        netz.genauigkeit(&test) * 100.0
    );
    for (bild, ziel) in &test {
        let (vorhersage, wahrsch) = netz.klassifiziere(bild);
        println!(
            "  Ziel {} -> Erkannt als {} (Konfidenz: {:.1} %)",
            ziel,
            vorhersage,
            wahrsch[vorhersage] * 100.0
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_summiert_auf_eins() {
        let p = softmax(&[2.0, 1.0, 0.1]);
        let summe: f32 = p.iter().sum();
        assert!((summe - 1.0).abs() < 1e-6);
        assert!(p[0] > p[1] && p[1] > p[2]);
    }

    #[test]
    fn werkstatt_netz_erkennt_verrauschte_ziffern() {
        let train = trainings_ziffern();
        let test = verrauschte_test_ziffern();
        let mut netz = ZiffernNetz::neu(25, 16, 5, 1337);
        for _ in 0..800 {
            for (bild, klasse) in &train {
                netz.trainiere_schritt(bild, *klasse, 0.35);
            }
        }
        assert!((netz.genauigkeit(&test) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn modell_speichern_und_laden() {
        let netz = ZiffernNetz::neu(25, 8, 5, 7);
        let text = netz.als_text();
        let geladen = ZiffernNetz::aus_text(&text).expect("Laden muss klappen");
        assert_eq!(netz.n_ein, geladen.n_ein);
        assert_eq!(netz.n_versteckt, geladen.n_versteckt);
        assert_eq!(netz.n_aus, geladen.n_aus);
        assert_eq!(netz.w_ev.len(), geladen.w_ev.len());
    }
}

