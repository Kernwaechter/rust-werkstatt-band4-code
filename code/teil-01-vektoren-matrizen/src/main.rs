/// Berechnet das Skalarprodukt (Dot Product) zweier gleich langer Vektoren.
pub fn skalarprodukt(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vektoren müssen gleich lang sein");
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Eine kompakte 2D-Matrix im Row-Major-Speicherlayout.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub zeilen: usize,
    pub spalten: usize,
    pub daten: Vec<f32>,
}

impl Matrix {
    pub fn neu(zeilen: usize, spalten: usize, daten: Vec<f32>) -> Self {
        assert_eq!(
            zeilen * spalten,
            daten.len(),
            "Datenlänge muss zeilen * spalten entsprechen"
        );
        Self {
            zeilen,
            spalten,
            daten,
        }
    }

    pub fn nullen(zeilen: usize, spalten: usize) -> Self {
        Self {
            zeilen,
            spalten,
            daten: vec![0.0; zeilen * spalten],
        }
    }

    pub fn get(&self, zeile: usize, spalte: usize) -> f32 {
        self.daten[zeile * self.spalten + spalte]
    }

    pub fn set(&mut self, zeile: usize, spalte: usize, wert: f32) {
        self.daten[zeile * self.spalten + spalte] = wert;
    }

    /// Spiegelt Zeilen und Spalten (Aus m×n wird n×m).
    pub fn transponiere(&self) -> Self {
        let mut ergebnis = Self::nullen(self.spalten, self.zeilen);
        for z in 0..self.zeilen {
            for s in 0..self.spalten {
                ergebnis.set(s, z, self.get(z, s));
            }
        }
        ergebnis
    }

    /// Multipliziert diese Matrix (m×n) mit einer zweiten Matrix (n×k) -> (m×k).
    pub fn matmul(&self, andere: &Self) -> Self {
        assert_eq!(
            self.spalten, andere.zeilen,
            "Spalten von A ({}) müssen Zeilen von B ({}) entsprechen",
            self.spalten, andere.zeilen
        );
        let mut ergebnis = Self::nullen(self.zeilen, andere.spalten);
        for z in 0..self.zeilen {
            for s in 0..andere.spalten {
                let mut summe = 0.0;
                for k in 0..self.spalten {
                    summe += self.get(z, k) * andere.get(k, s);
                }
                ergebnis.set(z, s, summe);
            }
        }
        ergebnis
    }
}

fn main() {
    // Beispiel: Zwei Merkmale (Sonne, Wochenende) werden mit zwei Neuronen gewichtet
    let eingabe = Matrix::neu(1, 2, vec![1.0, 0.5]);
    let gewichte = Matrix::neu(2, 2, vec![0.8, -0.2, 0.4, 0.9]);

    let ausgabe = eingabe.matmul(&gewichte);
    println!("Eingabe:  {:?}", eingabe.daten);
    println!("Ausgabe:  {:?}", ausgabe.daten);

    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 5.0, 6.0];
    println!("Skalarprodukt(a, b) = {}", skalarprodukt(&a, &b));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skalarprodukt_rechnet_korrekt() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert!((skalarprodukt(&a, &b) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn matrix_transponieren() {
        let m = Matrix::neu(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = m.transponiere();
        assert_eq!(t.zeilen, 3);
        assert_eq!(t.spalten, 2);
        assert_eq!(t.daten, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn matrix_multiplikation() {
        let a = Matrix::neu(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::neu(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let c = a.matmul(&b);
        assert_eq!(c.zeilen, 2);
        assert_eq!(c.spalten, 2);
        assert_eq!(c.daten, vec![58.0, 64.0, 139.0, 154.0]);
    }
}
