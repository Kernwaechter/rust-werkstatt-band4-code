/// Sprungfunktion (Heaviside Step): 1.0 ab 0.0, sonst 0.0.
pub fn step(x: f32) -> f32 {
    if x >= 0.0 { 1.0 } else { 0.0 }
}

/// Glatte S-Kurve zwischen 0.0 und 1.0 (ideal für Wahrscheinlichkeiten).
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Rectified Linear Unit: lässt positive Signale durch, kappt negative auf 0.
pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Ein einzelnes künstliches Neuron (Perzeptron).
#[derive(Debug, Clone)]
pub struct Perzeptron {
    pub gewichte: Vec<f32>,
    pub bias: f32,
    pub lernrate: f32,
}

impl Perzeptron {
    pub fn neu(anzahl_eingaben: usize, lernrate: f32) -> Self {
        Self {
            gewichte: vec![0.0; anzahl_eingaben],
            bias: 0.0,
            lernrate,
        }
    }

    pub fn rohsumme(&self, eingabe: &[f32]) -> f32 {
        assert_eq!(eingabe.len(), self.gewichte.len());
        let summe: f32 = eingabe.iter().zip(&self.gewichte).map(|(x, w)| x * w).sum();
        summe + self.bias
    }

    pub fn vorhersage(&self, eingabe: &[f32]) -> f32 {
        step(self.rohsumme(eingabe))
    }

    /// Passt Gewichte und Bias nach der Perzeptron-Lernregel an.
    pub fn trainiere(&mut self, daten: &[(&[f32], f32)], epochen: usize) {
        for _ in 0..epochen {
            for &(eingabe, ziel) in daten {
                let ausgabe = self.vorhersage(eingabe);
                let fehler = ziel - ausgabe;
                for (w, &x) in self.gewichte.iter_mut().zip(eingabe) {
                    *w += self.lernrate * fehler * x;
                }
                self.bias += self.lernrate * fehler;
            }
        }
    }
}

fn main() {
    let and_daten: [(&[f32], f32); 4] = [
        (&[0.0, 0.0], 0.0),
        (&[0.0, 1.0], 0.0),
        (&[1.0, 0.0], 0.0),
        (&[1.0, 1.0], 1.0),
    ];

    let mut neuron = Perzeptron::neu(2, 0.1);
    neuron.trainiere(&and_daten, 20);

    println!("Gelerntes AND-Perzeptron: gewichte={:?}, bias={:.2}", neuron.gewichte, neuron.bias);
    for (x, ziel) in &and_daten {
        println!("  {:?} -> Vorhersage: {} (Ziel: {})", x, neuron.vorhersage(x), ziel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perzeptron_lernt_and() {
        let daten: [(&[f32], f32); 4] = [
            (&[0.0, 0.0], 0.0),
            (&[0.0, 1.0], 0.0),
            (&[1.0, 0.0], 0.0),
            (&[1.0, 1.0], 1.0),
        ];
        let mut p = Perzeptron::neu(2, 0.1);
        p.trainiere(&daten, 30);
        for (x, ziel) in &daten {
            assert_eq!(p.vorhersage(x), *ziel);
        }
    }

    #[test]
    fn perzeptron_lernt_or() {
        let daten: [(&[f32], f32); 4] = [
            (&[0.0, 0.0], 0.0),
            (&[0.0, 1.0], 1.0),
            (&[1.0, 0.0], 1.0),
            (&[1.0, 1.0], 1.0),
        ];
        let mut p = Perzeptron::neu(2, 0.1);
        p.trainiere(&daten, 30);
        for (x, ziel) in &daten {
            assert_eq!(p.vorhersage(x), *ziel);
        }
    }

    #[test]
    fn aktivierungsfunktionen_im_bereich() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert_eq!(relu(-3.5), 0.0);
        assert_eq!(relu(2.5), 2.5);
    }
}
