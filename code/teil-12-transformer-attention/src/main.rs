use candle_core::{DType, Device, Result, Tensor};

/// Erzeugt eine kausale Maske (seq x seq): 0.0 erlaubt, -1e9 blockiert zukünftige Tokens.
pub fn kausale_maske(seq_len: usize, dev: &Device) -> Result<Tensor> {
    let mut daten = vec![0.0_f32; seq_len * seq_len];
    for z in 0..seq_len {
        for s in (z + 1)..seq_len {
            daten[z * seq_len + s] = -1e9;
        }
    }
    Tensor::from_vec(daten, (seq_len, seq_len), dev)
}

/// Berechnet Scaled Dot-Product Attention: softmax((Q * K^T) / sqrt(d_k) + Maske) * V
pub fn self_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mit_kausal_maske: bool,
) -> Result<(Tensor, Tensor)> {
    let (seq_q, d_k) = q.dims2()?;
    let k_t = k.t()?;
    let mut scores = q.matmul(&k_t)?;
    let skalar = (d_k as f64).sqrt();
    scores = (scores / skalar)?;

    if mit_kausal_maske {
        let maske = kausale_maske(seq_q, q.device())?;
        scores = scores.add(&maske)?;
    }

    let max_je_zeile = scores.max_keepdim(1)?;
    let verschoben = scores.broadcast_sub(&max_je_zeile)?;
    let exp = verschoben.exp()?;
    let summe = exp.sum_keepdim(1)?;
    let gewichte = exp.broadcast_div(&summe)?;
    let ausgabe = gewichte.matmul(v)?;

    Ok((ausgabe, gewichte))
}

/// Speichert bereits berechnete Key- und Value-Tensoren für schnelle Token-Generierung.
#[derive(Default)]
pub struct KvCache {
    pub keys: Option<Tensor>,
    pub values: Option<Tensor>,
}

impl KvCache {
    pub fn aktualisiere(&mut self, k_neu: &Tensor, v_neu: &Tensor) -> Result<(Tensor, Tensor)> {
        let k_ges = match &self.keys {
            Some(alt) => Tensor::cat(&[alt, k_neu], 0)?,
            None => k_neu.clone(),
        };
        let v_ges = match &self.values {
            Some(alt) => Tensor::cat(&[alt, v_neu], 0)?,
            None => v_neu.clone(),
        };
        self.keys = Some(k_ges.clone());
        self.values = Some(v_ges.clone());
        Ok((k_ges, v_ges))
    }
}

fn main() -> Result<()> {
    let dev = Device::Cpu;
    // 3 Tokens ("Rust", "ist", "schnell") mit d_k = 2
    let q = Tensor::from_slice(&[1.0_f32, 0.0, 0.5, 0.8, 0.9, 0.2], (3, 2), &dev)?;
    let k = Tensor::from_slice(&[1.0_f32, 0.0, 0.2, 0.9, 0.8, 0.3], (3, 2), &dev)?;
    let v = Tensor::from_slice(&[1.0_f32, 0.0, 0.0, 1.0, 0.5, 0.5], (3, 2), &dev)?;

    let (ausgabe, gewichte) = self_attention(&q, &k, &v, true)?;
    println!("Kausale Attention-Gewichte (3x3):\n{}", gewichte);
    println!("Kontext-Vektoren (3x2):\n{}", ausgabe);

    let mut cache = KvCache::default();
    let k1 = Tensor::zeros((1, 2), DType::F32, &dev)?;
    let v1 = Tensor::zeros((1, 2), DType::F32, &dev)?;
    let (k_nach_1, _) = cache.aktualisiere(&k1, &v1)?;
    let (k_nach_2, _) = cache.aktualisiere(&k1, &v1)?;
    println!("KV-Cache-Länge nach 2 Tokens: {} -> {}", k_nach_1.dims()[0], k_nach_2.dims()[0]);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attention_gewichte_summieren_auf_eins() -> Result<()> {
        let dev = Device::Cpu;
        let q = Tensor::from_slice(&[1.0_f32, 0.0, 0.0, 1.0], (2, 2), &dev)?;
        let (_, w) = self_attention(&q, &q, &q, false)?;
        let zeilen = w.to_vec2::<f32>()?;
        for zeile in zeilen {
            let summe: f32 = zeile.iter().sum();
            assert!((summe - 1.0).abs() < 1e-5);
        }
        Ok(())
    }

    #[test]
    fn kausale_maske_verhindert_blick_in_zukunft() -> Result<()> {
        let dev = Device::Cpu;
        let q = Tensor::from_slice(&[1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0], (3, 2), &dev)?;
        let (_, w) = self_attention(&q, &q, &q, true)?;
        let zeilen = w.to_vec2::<f32>()?;
        // Zeile 0 darf nur auf Spalte 0 schauen -> Spalte 1 & 2 müssen ~0.0 sein
        assert!(zeilen[0][1] < 1e-6);
        assert!(zeilen[0][2] < 1e-6);
        assert!(zeilen[1][2] < 1e-6);
        Ok(())
    }

    #[test]
    fn kv_cache_waechst_mit_jedem_token() -> Result<()> {
        let dev = Device::Cpu;
        let mut cache = KvCache::default();
        let k_step = Tensor::zeros((1, 4), DType::F32, &dev)?;
        let v_step = Tensor::zeros((1, 4), DType::F32, &dev)?;

        let (k1, _) = cache.aktualisiere(&k_step, &v_step)?;
        let (k2, _) = cache.aktualisiere(&k_step, &v_step)?;
        let (k3, _) = cache.aktualisiere(&k_step, &v_step)?;
        assert_eq!(k1.dims(), &[1, 4]);
        assert_eq!(k2.dims(), &[2, 4]);
        assert_eq!(k3.dims(), &[3, 4]);
        Ok(())
    }
}
