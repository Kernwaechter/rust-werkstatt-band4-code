use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    pub merges: Vec<((u32, u32), u32)>,
    pub vocab: HashMap<u32, Vec<u8>>,
}

impl BpeTokenizer {
    /// Trainiert einen Byte-Pair-Encoding-Tokenizer auf Basis-Bytes (0..255).
    pub fn trainiere(text: &str, anzahl_merges: usize) -> Self {
        let mut vocab: HashMap<u32, Vec<u8>> = (0..=255_u32).map(|b| (b, vec![b as u8])).collect();
        let mut tokens: Vec<u32> = text.as_bytes().iter().map(|&b| b as u32).collect();
        let mut merges = Vec::new();
        let mut naechste_id = 256_u32;

        for _ in 0..anzahl_merges {
            let mut paare: HashMap<(u32, u32), usize> = HashMap::new();
            for fenster in tokens.windows(2) {
                *paare.entry((fenster[0], fenster[1])).or_insert(0) += 1;
            }
            let Some((&bestes_paar, &anzahl)) = paare.iter().max_by_key(|(_, &c)| c) else {
                break;
            };
            if anzahl < 2 {
                break;
            }

            let mut kombiniert = vocab[&bestes_paar.0].clone();
            kombiniert.extend_from_slice(&vocab[&bestes_paar.1]);
            vocab.insert(naechste_id, kombiniert);
            merges.push((bestes_paar, naechste_id));
            tokens = wende_merge_an(&tokens, bestes_paar, naechste_id);
            naechste_id += 1;
        }

        Self { merges, vocab }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens: Vec<u32> = text.as_bytes().iter().map(|&b| b as u32).collect();
        for &(paar, neue_id) in &self.merges {
            tokens = wende_merge_an(&tokens, paar, neue_id);
        }
        tokens
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for id in ids {
            if let Some(b) = self.vocab.get(id) {
                bytes.extend_from_slice(b);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

fn wende_merge_an(tokens: &[u32], paar: (u32, u32), neue_id: u32) -> Vec<u32> {
    let mut ergebnis = Vec::with_capacity(tokens.len());
    let mut i = 0;
    while i < tokens.len() {
        if i + 1 < tokens.len() && tokens[i] == paar.0 && tokens[i + 1] == paar.1 {
            ergebnis.push(neue_id);
            i += 2;
        } else {
            ergebnis.push(tokens[i]);
            i += 1;
        }
    }
    ergebnis
}

/// Einfache Embedding-Lookup-Tabelle (Vokabulargröße x Embedding-Dimension).
pub struct EmbeddingTabelle {
    pub dim: usize,
    pub tabelle: HashMap<u32, Vec<f32>>,
}

impl EmbeddingTabelle {
    pub fn neu(vokabular_ids: impl Iterator<Item = u32>, dim: usize, seed: u64) -> Self {
        let mut z = seed;
        let mut next_f32 = || {
            z = z.wrapping_mul(6364136223846793005).wrapping_add(1);
            (((z >> 32) as f32) / (u32::MAX as f32)) * 2.0 - 1.0
        };
        let mut tabelle = HashMap::new();
        for id in vokabular_ids {
            let vektor = (0..dim).map(|_| next_f32()).collect();
            tabelle.insert(id, vektor);
        }
        Self { dim, tabelle }
    }

    pub fn nachschlagen(&self, ids: &[u32]) -> Vec<Vec<f32>> {
        ids.iter()
            .map(|id| {
                self.tabelle
                    .get(id)
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; self.dim])
            })
            .collect()
    }
}

fn main() {
    let korpus = "rust werkstatt rust ki rust lokal werkstatt ki";
    let tok = BpeTokenizer::trainiere(korpus, 10);
    let ids = tok.encode("rust ki");
    let zurueck = tok.decode(&ids);

    println!("Original: 'rust ki' ({} Bytes)", "rust ki".len());
    println!("Tokens:   {:?} ({} Token-IDs)", ids, ids.len());
    println!("Dekodiert: '{}'", zurueck);

    let emb = EmbeddingTabelle::neu(tok.vocab.keys().copied(), 4, 42);
    let vektoren = emb.nachschlagen(&ids);
    println!("Erster Token-Vektor (dim=4): {:?}", vektoren[0]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bpe_lernt_haeufiges_paar() {
        let tok = BpeTokenizer::trainiere("ab ab ab ab", 2);
        let ids = tok.encode("ab");
        assert_eq!(ids.len(), 1);
    }

    #[test]
    fn bpe_encode_decode_roundtrip() {
        let tok = BpeTokenizer::trainiere("Hallo Rust Werkstatt Hallo KI", 8);
        let text = "Hallo KI!";
        let ids = tok.encode(text);
        assert_eq!(tok.decode(&ids), text);
    }

    #[test]
    fn embedding_nachschlagen_hat_richtige_dimension() {
        let tok = BpeTokenizer::trainiere("rust ki rust", 4);
        let emb = EmbeddingTabelle::neu(tok.vocab.keys().copied(), 8, 7);
        let ids = tok.encode("rust");
        let vektoren = emb.nachschlagen(&ids);
        assert_eq!(vektoren.len(), ids.len());
        assert!(vektoren.iter().all(|v| v.len() == 8));
    }
}
