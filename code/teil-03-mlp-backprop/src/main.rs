pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Ableitung von Sigmoid, berechnet direkt aus dem bereits bekannten Ausgabewert y = sigmoid(x).
pub fn sigmoid_ableitung_aus_y(y: f32) -> f32 {
    y * (1.0 - y)
}

/// Kompakter deterministischer Zufallsgenerator (Linear Congruential Generator).
struct Lcg(u64);

impl Lcg {
    fn neu(seed: u64) -> Self {
        Self(seed)
    }

    /// Liefert eine Zufallszahl im Bereich [-1.0, 1.0].
    fn naechstes_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        let norm = ((self.0 >> 32) as f32) / (u32::MAX as f32);
        norm * 2.0 - 1.0
    }
}

/// Ein zweischichtiges Multi-Layer Perceptron (Eingabe -> Versteckt -> Ausgabe).
pub struct Mlp {
    n_ein: usize,
    n_versteckt: usize,
    n_aus: usize,
    w_ev: Vec<f32>, // Gewichte Eingabe -> Versteckt (n_versteckt x n_ein)
    b_v: Vec<f32>,  // Bias Versteckt
    w_va: Vec<f32>, // Gewichte Versteckt -> Ausgabe (n_aus x n_versteckt)
    b_a: Vec<f32>,  // Bias Ausgabe
}

impl Mlp {
    pub fn neu(n_ein: usize, n_versteckt: usize, n_aus: usize, seed: u64) -> Self {
        let mut rng = Lcg::neu(seed);
        let w_ev = (0..n_versteckt * n_ein).map(|_| rng.naechstes_f32()).collect();
        let b_v = vec![0.0; n_versteckt];
        let w_va = (0..n_aus * n_versteckt).map(|_| rng.naechstes_f32()).collect();
        let b_a = vec![0.0; n_aus];
        Self {
            n_ein,
            n_versteckt,
            n_aus,
            w_ev,
            b_v,
            w_va,
            b_a,
        }
    }

    /// Berechnet versteckte Aktivierungen und finale Ausgabe (Forward Pass).
    pub fn vorwaerts(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        assert_eq!(x.len(), self.n_ein);

        let mut versteckt = vec![0.0; self.n_versteckt];
        for h in 0..self.n_versteckt {
            let mut summe = self.b_v[h];
            for i in 0..self.n_ein {
                summe += self.w_ev[h * self.n_ein + i] * x[i];
            }
            versteckt[h] = sigmoid(summe);
        }

        let mut ausgabe = vec![0.0; self.n_aus];
        for o in 0..self.n_aus {
            let mut summe = self.b_a[o];
            for h in 0..self.n_versteckt {
                summe += self.w_va[o * self.n_versteckt + h] * versteckt[h];
            }
            ausgabe[o] = sigmoid(summe);
        }

        (versteckt, ausgabe)
    }

    /// Führt einen kompletten Trainings-Schritt (Forward + Backprop + Update) aus und liefert den MSE-Fehler.
    pub fn trainiere_schritt(&mut self, x: &[f32], ziel: &[f32], lernrate: f32) -> f32 {
        let (versteckt, ausgabe) = self.vorwaerts(x);

        let mut mse = 0.0;
        let mut delta_a = vec![0.0; self.n_aus];
        for o in 0..self.n_aus {
            let fehler = ziel[o] - ausgabe[o];
            mse += fehler * fehler;
            delta_a[o] = fehler * sigmoid_ableitung_aus_y(ausgabe[o]);
        }

        let mut delta_v = vec![0.0; self.n_versteckt];
        for h in 0..self.n_versteckt {
            let mut fehler_anteil = 0.0;
            for o in 0..self.n_aus {
                fehler_anteil += delta_a[o] * self.w_va[o * self.n_versteckt + h];
            }
            delta_v[h] = fehler_anteil * sigmoid_ableitung_aus_y(versteckt[h]);
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
}

fn main() {
    let xor_daten: [([f32; 2], [f32; 1]); 4] = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ];

    let mut mlp = Mlp::neu(2, 4, 1, 42);
    for epoche in 0..6000 {
        let mut gesamt_mse = 0.0;
        for (x, y) in &xor_daten {
            gesamt_mse += mlp.trainiere_schritt(x, y, 0.5);
        }
        if epoche % 1500 == 0 {
            println!("Epoche {:4}: MSE = {:.5}", epoche, gesamt_mse / 4.0);
        }
    }

    println!("\nXOR-Vorhersagen nach dem Training:");
    for (x, y) in &xor_daten {
        let (_, aus) = mlp.vorwaerts(x);
        println!("  {:?} -> {:.4} (Ziel: {})", x, aus[0], y[0]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_ableitung_korrekt() {
        let y = sigmoid(0.0); // 0.5
        assert!((sigmoid_ableitung_aus_y(y) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn mlp_lernt_xor() {
        let xor_daten: [([f32; 2], [f32; 1]); 4] = [
            ([0.0, 0.0], [0.0]),
            ([0.0, 1.0], [1.0]),
            ([1.0, 0.0], [1.0]),
            ([1.0, 1.0], [0.0]),
        ];
        let mut mlp = Mlp::neu(2, 4, 1, 42);
        for _ in 0..6000 {
            for (x, y) in &xor_daten {
                mlp.trainiere_schritt(x, y, 0.5);
            }
        }
        assert!(mlp.vorwaerts(&[0.0, 0.0]).1[0] < 0.15);
        assert!(mlp.vorwaerts(&[0.0, 1.0]).1[0] > 0.85);
        assert!(mlp.vorwaerts(&[1.0, 0.0]).1[0] > 0.85);
        assert!(mlp.vorwaerts(&[1.0, 1.0]).1[0] < 0.15);
    }
}
