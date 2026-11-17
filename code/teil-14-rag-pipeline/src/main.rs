/// Zerlegt einen langen Text in überlappende Wort-Chunks, damit Sätze an Grenzen nicht verloren gehen.
pub fn teile_in_chunks(text: &str, max_woerter: usize, ueberlappung: usize) -> Vec<String> {
    let woerter: Vec<&str> = text.split_whitespace().collect();
    if woerter.is_empty() || max_woerter == 0 {
        return Vec::new();
    }
    let schritt = max_woerter.saturating_sub(ueberlappung).max(1);
    let mut chunks = Vec::new();
    let mut start = 0;

    while start < woerter.len() {
        let ende = (start + max_woerter).min(woerter.len());
        chunks.push(woerter[start..ende].join(" "));
        if ende == woerter.len() {
            break;
        }
        start += schritt;
    }
    chunks
}

/// Einfaches Hash-Embedding (Feature-Hashing auf 32 Dimensionen) für schnelle Offline-Suche.
pub fn text_zu_vektor(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; dim];
    for wort in text.to_lowercase().split_whitespace() {
        let sauber = wort.trim_matches(|c: char| !c.is_alphanumeric());
        if sauber.is_empty() {
            continue;
        }
        let mut hash: u64 = 5381;
        for b in sauber.as_bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(*b as u64);
        }
        v[(hash as usize) % dim] += 1.0;
    }
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

pub fn skalarprodukt(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[derive(Debug, Clone)]
pub struct ChunkEintrag {
    pub quelle: String,
    pub chunk_id: usize,
    pub text: String,
    pub vektor: Vec<f32>,
}

#[derive(Default)]
pub struct RagIndex {
    pub chunks: Vec<ChunkEintrag>,
}

impl RagIndex {
    pub fn indiziere_dokument(&mut self, quelle: &str, inhalt: &str, chunk_woerter: usize) {
        let teile = teile_in_chunks(inhalt, chunk_woerter, 3);
        for (idx, teil) in teile.into_iter().enumerate() {
            let vektor = text_zu_vektor(&teil, 128);
            self.chunks.push(ChunkEintrag {
                quelle: quelle.to_string(),
                chunk_id: idx,
                text: teil,
                vektor,
            });
        }
    }

    pub fn suche_kontext(&self, frage: &str, top_k: usize) -> Vec<(&ChunkEintrag, f32)> {
        let q_vec = text_zu_vektor(frage, 128);
        let mut treffer: Vec<(&ChunkEintrag, f32)> = self
            .chunks
            .iter()
            .map(|c| (c, skalarprodukt(&c.vektor, &q_vec)))
            .collect();
        treffer.sort_by(|a, b| b.1.total_cmp(&a.1));
        treffer.truncate(top_k);
        treffer
    }
}

pub fn baue_rag_prompt(frage: &str, treffer: &[(&ChunkEintrag, f32)]) -> String {
    let mut prompt = String::from(
        "SYSTEM: Beantworte die Frage ausschließlich anhand der folgenden Quellen. Nenne am Ende die Quelle.\n\n",
    );
    for (chunk, score) in treffer {
        prompt.push_str(&format!(
            "[Quelle: {}#{} | Score: {:.2}]\n{}\n\n",
            chunk.quelle, chunk.chunk_id, score, chunk.text
        ));
    }
    prompt.push_str(&format!("FRAGE: {}\nANTWORT:", frage));
    prompt
}

fn main() {
    let mut index = RagIndex::default();
    index.indiziere_dokument(
        "rust_handbuch.md",
        "Rust garantiert Speichersicherheit durch Ownership und den Borrow Checker zur Kompilierzeit.",
        12,
    );
    index.indiziere_dokument(
        "candle_guide.md",
        "Hugging Face Candle berechnet Tensoren direkt auf CPU, CUDA oder Apple Metal ohne Python.",
        12,
    );

    let frage = "Wie sorgt Rust fuer Speichersicherheit?";
    let treffer = index.suche_kontext(frage, 1);
    let prompt = baue_rag_prompt(frage, &treffer);
    println!("{}", prompt);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunking_mit_ueberlappung_erhaelt_kontext() {
        let text = "eins zwei drei vier fuenf sechs sieben acht";
        let chunks = teile_in_chunks(text, 4, 2);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], "eins zwei drei vier");
        assert_eq!(chunks[1], "drei vier fuenf sechs");
    }

    #[test]
    fn rag_findet_richtige_quelle() {
        let mut index = RagIndex::default();
        index.indiziere_dokument("rust.md", "Ownership und Borrow Checker pruefen Speicher", 10);
        index.indiziere_dokument("kueche.md", "Mehl Eier und Milch fuer Pfannkuchen", 10);

        let treffer = index.suche_kontext("Wie funktioniert der Borrow Checker?", 1);
        assert_eq!(treffer[0].0.quelle, "rust.md");
    }

    #[test]
    fn prompt_enthaelt_quelle_und_frage() {
        let mut index = RagIndex::default();
        index.indiziere_dokument("notiz.md", "Candle nutzt GGUF Modelle", 10);
        let treffer = index.suche_kontext("Was nutzt Candle?", 1);
        let prompt = baue_rag_prompt("Was nutzt Candle?", &treffer);
        assert!(prompt.contains("[Quelle: notiz.md#0"));
        assert!(prompt.contains("FRAGE: Was nutzt Candle?"));
    }
}
