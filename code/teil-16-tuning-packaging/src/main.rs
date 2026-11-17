use std::time::Instant;

/// Klassisches Skalarprodukt zum Vergleich.
pub fn skalarprodukt_klassisch(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// 4-fach entrolltes Skalarprodukt (erleichtert der CPU Auto-Vektorisierung / SIMD).
pub fn skalarprodukt_unrolled(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut s0 = 0.0_f32;
    let mut s1 = 0.0_f32;
    let mut s2 = 0.0_f32;
    let mut s3 = 0.0_f32;

    let mut i = 0;
    while i + 4 <= a.len() {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
        i += 4;
    }
    let mut rest = 0.0_f32;
    while i < a.len() {
        rest += a[i] * b[i];
        i += 1;
    }
    (s0 + s1) + (s2 + s3) + rest
}

/// Schätzt den benötigten Arbeitsspeicher in Megabyte (Gewichte + ca. 20 % KV-Cache-Reserve).
pub fn schaetze_ram_mb(milliarden_params: f32, bits_pro_param: f32) -> f32 {
    let gewichte_mb = (milliarden_params * 1_000.0 * bits_pro_param) / 8.0;
    gewichte_mb * 1.20
}

#[derive(Debug, Clone)]
pub struct ReleaseKonfig {
    pub modell_datei: String,
    pub temperatur: f32,
    pub max_tokens: usize,
    pub verfuegbar_ram_mb: f32,
}

pub fn pruefe_release_konfig(k: &ReleaseKonfig) -> Result<(), String> {
    if !k.modell_datei.ends_with(".gguf") && !k.modell_datei.ends_with(".safetensors") {
        return Err("Nur sichere Modellformate (.gguf oder .safetensors) erlaubt!".into());
    }
    if k.max_tokens == 0 || k.max_tokens > 4096 {
        return Err("max_tokens muss zwischen 1 und 4096 liegen".into());
    }
    if !(0.0..=1.5).contains(&k.temperatur) {
        return Err("Temperatur sollte zwischen 0.0 und 1.5 liegen".into());
    }
    if k.verfuegbar_ram_mb < 512.0 {
        return Err("Mindestens 512 MB freier RAM erforderlich".into());
    }
    Ok(())
}

fn main() {
    let n = 4096;
    let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
    let b: Vec<f32> = (0..n).map(|i| ((n - i) as f32) * 0.001).collect();

    let t1 = Instant::now();
    let mut sum1 = 0.0;
    for _ in 0..1000 {
        sum1 += skalarprodukt_klassisch(&a, &b);
    }
    let d1 = t1.elapsed();

    let t2 = Instant::now();
    let mut sum2 = 0.0;
    for _ in 0..1000 {
        sum2 += skalarprodukt_unrolled(&a, &b);
    }
    let d2 = t2.elapsed();

    println!("1000x Skalarprodukt (dim=4096): klassisch={:?} ({:.1}), unrolled={:?} ({:.1})", d1, sum1, d2, sum2);
    println!("RAM-Bedarf 3B-Modell in FP32:   {:.0} MB", schaetze_ram_mb(3.0, 32.0));
    println!("RAM-Bedarf 3B-Modell in Q4_K_M: {:.0} MB", schaetze_ram_mb(3.0, 4.5));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unrolled_entspricht_skalar() {
        let a = [1.0, 2.5, -3.0, 4.0, 0.5, 1.2];
        let b = [0.5, -1.0, 2.0, 3.0, 4.0, -0.5];
        let k = skalarprodukt_klassisch(&a, &b);
        let u = skalarprodukt_unrolled(&a, &b);
        assert!((k - u).abs() < 1e-5);
    }

    #[test]
    fn ram_budget_q4_kleiner_als_fp32() {
        let fp32 = schaetze_ram_mb(3.0, 32.0);
        let q4 = schaetze_ram_mb(3.0, 4.5);
        assert!(q4 < fp32 / 6.0);
    }

    #[test]
    fn release_konfig_validiert() {
        let ok = ReleaseKonfig {
            modell_datei: "phi-3-mini-q4.gguf".into(),
            temperatur: 0.2,
            max_tokens: 512,
            verfuegbar_ram_mb: 4096.0,
        };
        assert!(pruefe_release_konfig(&ok).is_ok());

        let pkl = ReleaseKonfig {
            modell_datei: "gefaehrlich.pkl".into(),
            ..ok
        };
        assert!(pruefe_release_konfig(&pkl).is_err());
    }
}
