use std::io::{self, Write};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufHeader {
    pub version: u32,
    pub tensor_anzahl: u64,
    pub metadata_anzahl: u64,
}

/// Liest die ersten 24 Bytes einer GGUF-Modelldatei (Magic "GGUF" + Version + Zähler).
pub fn lies_gguf_header(bytes: &[u8]) -> Result<GgufHeader, String> {
    if bytes.len() < 24 {
        return Err("Datei zu klein für einen gültigen GGUF-Header".into());
    }
    if &bytes[0..4] != b"GGUF" {
        return Err("Ungültige Magic-Bytes: Keine GGUF-Modelldatei".into());
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    let tensor_anzahl = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
    let metadata_anzahl = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    Ok(GgufHeader {
        version,
        tensor_anzahl,
        metadata_anzahl,
    })
}

pub fn softmax_mit_temperatur(logits: &[f32], temperatur: f32) -> Vec<f32> {
    let t = temperatur.max(1e-5);
    let skaliert: Vec<f32> = logits.iter().map(|&x| x / t).collect();
    let max_w = skaliert.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = skaliert.iter().map(|&x| (x - max_w).exp()).collect();
    let summe: f32 = exp.iter().sum();
    exp.iter().map(|&e| e / summe).collect()
}

/// Filtert nach Top-P (Nucleus Sampling): behält nur die wahrscheinlichsten Tokens bis Summe >= top_p.
pub fn top_p_kandidaten(wahrsch: &[f32], top_p: f32) -> Vec<(usize, f32)> {
    let mut sortiert: Vec<(usize, f32)> = wahrsch.iter().copied().enumerate().collect();
    sortiert.sort_by(|a, b| b.1.total_cmp(&a.1));

    let mut behalten = Vec::new();
    let mut kumuliert = 0.0;
    for (idx, p) in sortiert {
        behalten.push((idx, p));
        kumuliert += p;
        if kumuliert >= top_p {
            break;
        }
    }
    behalten
}

/// Wählt das nächste Token: bei temperatur <= 0.0 deterministisch per Argmax, sonst per Top-P-Sampling.
pub fn waehle_naechstes_token(logits: &[f32], temperatur: f32, top_p: f32, seed: u64) -> usize {
    if temperatur <= 0.0 {
        return logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap_or(0);
    }

    let wahrsch = softmax_mit_temperatur(logits, temperatur);
    let kandidaten = top_p_kandidaten(&wahrsch, top_p);
    let summe: f32 = kandidaten.iter().map(|(_, p)| *p).sum();

    let z = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let zufall = ((z >> 32) as f32 / u32::MAX as f32) * summe;

    let mut akkum = 0.0;
    for (idx, p) in &kandidaten {
        akkum += *p;
        if zufall <= akkum {
            return *idx;
        }
    }
    kandidaten.first().map(|(idx, _)| *idx).unwrap_or(0)
}

fn main() {
    let vokabular = ["Rust", "läuft", "schnell", "und", "sicher", "offline!"];
    let simulierte_logits = [
        vec![0.1, 4.2, 1.0, 0.2, 0.1, 0.0],
        vec![0.1, 0.2, 4.5, 1.2, 0.8, 0.0],
        vec![0.0, 0.1, 0.3, 0.5, 3.8, 4.4],
    ];

    print!("Streaming-Ausgabe (Temperatur=0.0): ");
    for (schritt, logits) in simulierte_logits.iter().enumerate() {
        let id = waehle_naechstes_token(logits, 0.0, 0.9, schritt as u64 + 1);
        print!("{} ", vokabular[id]);
        let _ = io::stdout().flush();
    }
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gguf_header_erkennt_gueltige_datei() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&128_u64.to_le_bytes());
        bytes.extend_from_slice(&24_u64.to_le_bytes());

        let info = lies_gguf_header(&bytes).expect("Gültiger GGUF-Header");
        assert_eq!(info.version, 3);
        assert_eq!(info.tensor_anzahl, 128);
        assert_eq!(info.metadata_anzahl, 24);
    }

    #[test]
    fn temperature_null_ist_deterministisch() {
        let logits = [0.2, 3.9, 1.5, 0.1];
        assert_eq!(waehle_naechstes_token(&logits, 0.0, 0.9, 1), 1);
        assert_eq!(waehle_naechstes_token(&logits, 0.0, 0.9, 999), 1);
    }

    #[test]
    fn top_p_schneidet_unwahrscheinliche_tokens_ab() {
        let wahrsch = [0.05, 0.60, 0.32, 0.03];
        let kandidaten = top_p_kandidaten(&wahrsch, 0.90);
        // 0.60 + 0.32 = 0.92 >= 0.90 -> nur 2 Kandidaten bleiben übrig!
        assert_eq!(kandidaten.len(), 2);
        assert_eq!(kandidaten[0].0, 1);
        assert_eq!(kandidaten[1].0, 2);
    }
}
