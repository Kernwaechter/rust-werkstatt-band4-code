/// Berechnet die Cosinus-Ähnlichkeit zwischen -1.0 (entgegengesetzt) und +1.0 (identische Richtung).
pub fn cosinus_aehnlichkeit(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vektoren müssen gleich lang sein");
    let skalar: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nenner = norm_a * norm_b;
    if nenner < 1e-8 {
        0.0
    } else {
        skalar / nenner
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Dokument {
    pub id: usize,
    pub titel: String,
    pub text: String,
    pub vektor: Vec<f32>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct VektorTresor {
    pub eintraege: Vec<Dokument>,
}

impl VektorTresor {
    pub fn neu() -> Self {
        Self::default()
    }

    pub fn hinzufuegen(&mut self, titel: impl Into<String>, text: impl Into<String>, vektor: Vec<f32>) {
        let id = self.eintraege.len();
        self.eintraege.push(Dokument {
            id,
            titel: titel.into(),
            text: text.into(),
            vektor,
        });
    }

    /// Sucht die `top_k` ähnlichsten Dokumente zur Anfrage.
    pub fn suche(&self, anfrage: &[f32], top_k: usize) -> Vec<(&Dokument, f32)> {
        let mut treffer: Vec<(&Dokument, f32)> = self
            .eintraege
            .iter()
            .map(|dok| (dok, cosinus_aehnlichkeit(&dok.vektor, anfrage)))
            .collect();

        treffer.sort_by(|a, b| b.1.total_cmp(&a.1));
        treffer.truncate(top_k);
        treffer
    }

    pub fn als_text(&self) -> String {
        let mut zeilen = Vec::with_capacity(self.eintraege.len());
        for d in &self.eintraege {
            let vec_str: Vec<String> = d.vektor.iter().map(|v| format!("{v:.6}")).collect();
            let titel_clean = d.titel.replace('|', " ");
            let text_clean = d.text.replace(['|', '\n'], " ");
            zeilen.push(format!("{}|{}|{}|{}", d.id, titel_clean, text_clean, vec_str.join(",")));
        }
        zeilen.join("\n")
    }

    pub fn aus_text(inhalt: &str) -> Result<Self, String> {
        let mut tresor = Self::neu();
        for zeile in inhalt.lines().filter(|z| !z.trim().is_empty()) {
            let teile: Vec<&str> = zeile.splitn(4, '|').collect();
            if teile.len() != 4 {
                return Err("Ungültiges Tresor-Zeilenformat".into());
            }
            let id = teile[0].parse::<usize>().map_err(|e| e.to_string())?;
            let vektor = teile[3]
                .split(',')
                .map(|s| s.parse::<f32>().map_err(|e| e.to_string()))
                .collect::<Result<Vec<f32>, String>>()?;
            tresor.eintraege.push(Dokument {
                id,
                titel: teile[1].to_string(),
                text: teile[2].to_string(),
                vektor,
            });
        }
        Ok(tresor)
    }
}

fn main() {
    // Themen-Achsen: [Rust/System, KI/Mathe, Kochen/Alltag]
    let mut tresor = VektorTresor::neu();
    tresor.hinzufuegen("Ownership", "Speichersicherheit ohne Garbage Collector", vec![0.95, 0.10, 0.0]);
    tresor.hinzufuegen("Backpropagation", "Gradienten durch neuronale Netze leiten", vec![0.20, 0.95, 0.0]);
    tresor.hinzufuegen("Pfannkuchen", "Mehl, Eier und Milch verrühren", vec![0.0, 0.0, 1.0]);

    let anfrage = vec![0.15, 0.90, 0.0]; // Suche nach KI-Mathe
    println!("Top-2 Treffer für KI-Anfrage:");
    for (dok, score) in tresor.suche(&anfrage, 2) {
        println!("  [{:.3}] {} — {}", score, dok.titel, dok.text);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosinus_gleiche_richtung_ist_eins() {
        let sim = cosinus_aehnlichkeit(&[2.0, 4.0], &[1.0, 2.0]);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosinus_senkrecht_ist_null() {
        let sim = cosinus_aehnlichkeit(&[1.0, 0.0], &[0.0, 1.0]);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn tresor_findet_passendstes_dokument_und_speichert() {
        let mut tresor = VektorTresor::neu();
        tresor.hinzufuegen("Rust", "Borrow Checker", vec![1.0, 0.0]);
        tresor.hinzufuegen("KI", "Neuronales Netz", vec![0.0, 1.0]);

        let treffer = tresor.suche(&[0.1, 0.99], 1);
        assert_eq!(treffer[0].0.titel, "KI");

        let serialisiert = tresor.als_text();
        let geladen = VektorTresor::aus_text(&serialisiert).expect("Tresor laden");
        assert_eq!(geladen.eintraege.len(), 2);
        assert_eq!(geladen.eintraege[0].titel, "Rust");
    }
}
