use candle_core::{Device, Result, Tensor};
use std::collections::HashMap;

/// Berechnet den Durchschnitt aller Token-Vektoren entlang der Sequenz-Achse (Mean Pooling)
/// und normiert das Ergebnis auf Länge 1.0 (L2-Normalisierung).
pub fn satz_embedding(token_vektoren: &Tensor) -> Result<Tensor> {
    let gemittelt = token_vektoren.mean(0)?;
    let norm = gemittelt.sqr()?.sum_all()?.sqrt()?;
    let norm_val = norm.to_scalar::<f32>()?;
    if norm_val < 1e-8 {
        return Ok(gemittelt);
    }
    gemittelt.broadcast_div(&norm)
}

/// Skalarprodukt zweier bereits L2-normierter 1D-Tensoren = Cosinus-Ähnlichkeit.
pub fn tensor_cosinus(a: &Tensor, b: &Tensor) -> Result<f32> {
    a.mul(b)?.sum_all()?.to_scalar::<f32>()
}

pub fn beispiel_vokabular() -> HashMap<&'static str, [f32; 3]> {
    // 3 Themen-Dimensionen: [Rust/Code, KI/Lernen, Küche/Essen]
    HashMap::from([
        ("rust", [0.9, 0.2, 0.0]),
        ("speicher", [0.85, 0.1, 0.0]),
        ("compiler", [0.95, 0.1, 0.0]),
        ("neuron", [0.1, 0.95, 0.0]),
        ("lernen", [0.1, 0.9, 0.0]),
        ("netz", [0.2, 0.9, 0.0]),
        ("brot", [0.0, 0.0, 0.95]),
        ("backen", [0.0, 0.0, 0.9]),
    ])
}

pub fn kodiere_satz(satz: &str, vokabular: &HashMap<&str, [f32; 3]>, dev: &Device) -> Result<Tensor> {
    let mut gesammelt = Vec::new();
    let mut anzahl = 0usize;
    for wort in satz.to_lowercase().split_whitespace() {
        let sauber = wort.trim_matches(|c: char| !c.is_alphanumeric());
        if let Some(vec) = vokabular.get(sauber) {
            gesammelt.extend_from_slice(vec);
            anzahl += 1;
        }
    }
    if anzahl == 0 {
        return Tensor::zeros((3,), candle_core::DType::F32, dev);
    }
    let matrix = Tensor::from_vec(gesammelt, (anzahl, 3), dev)?;
    satz_embedding(&matrix)
}

fn main() -> Result<()> {
    let dev = Device::Cpu;
    let vok = beispiel_vokabular();

    let e1 = kodiere_satz("Rust und der Compiler pruefen Speicher", &vok, &dev)?;
    let e2 = kodiere_satz("Ein Neuron im Netz kann lernen", &vok, &dev)?;
    let e3 = kodiere_satz("Rust hilft beim Lernen", &vok, &dev)?;
    let e4 = kodiere_satz("Frisches Brot backen", &vok, &dev)?;

    println!("Ähnlichkeit (Rust-Satz vs. Rust+Lernen): {:.3}", tensor_cosinus(&e1, &e3)?);
    println!("Ähnlichkeit (KI-Satz   vs. Rust+Lernen): {:.3}", tensor_cosinus(&e2, &e3)?);
    println!("Ähnlichkeit (Rust-Satz vs. Brot backen): {:.3}", tensor_cosinus(&e1, &e4)?);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_pooling_bildet_durchschnitt() -> Result<()> {
        let dev = Device::Cpu;
        let t = Tensor::from_slice(&[2.0_f32, 0.0, 0.0, 2.0], (2, 2), &dev)?;
        let emb = satz_embedding(&t)?;
        let v = emb.to_vec1::<f32>()?;
        // Durchschnitt ist [1.0, 1.0], normiert -> [0.7071, 0.7071]
        assert!((v[0] - v[1]).abs() < 1e-5);
        assert!((v[0] * v[0] + v[1] * v[1] - 1.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn aehnliche_saetze_haben_hoehere_aehnlichkeit() -> Result<()> {
        let dev = Device::Cpu;
        let vok = beispiel_vokabular();
        let s_rust = kodiere_satz("Rust Compiler Speicher", &vok, &dev)?;
        let s_rust2 = kodiere_satz("Rust Speicher", &vok, &dev)?;
        let s_brot = kodiere_satz("Brot backen", &vok, &dev)?;

        let sim_nah = tensor_cosinus(&s_rust, &s_rust2)?;
        let sim_fern = tensor_cosinus(&s_rust, &s_brot)?;
        assert!(sim_nah > 0.95);
        assert!(sim_fern < 0.05);
        Ok(())
    }
}
