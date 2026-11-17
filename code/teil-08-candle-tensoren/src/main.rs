use candle_core::{Device, Result, Tensor, Var};

/// Berechnet y = relu(x * w + b) mit Candle-Tensoren.
pub fn lineare_schicht_relu(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    x.matmul(w)?.broadcast_add(b)?.relu()
}

/// Führt einen automatischen Gradienten-Schritt für f(x) = (x - 3)^2 aus.
pub fn autograd_beispiel(startwert: f32) -> Result<(f32, f32)> {
    let device = Device::Cpu;
    let x = Var::from_slice(&[startwert], (1,), &device)?;
    let ziel = Tensor::from_slice(&[3.0_f32], (1,), &device)?;

    let diff = x.as_tensor().sub(&ziel)?;
    let loss = diff.sqr()?.sum_all()?;
    let grads = loss.backward()?;
    let grad_x = grads.get(x.as_tensor()).expect("Gradient für x muss existieren");

    Ok((loss.to_scalar::<f32>()?, grad_x.to_vec1::<f32>()?[0]))
}

fn main() -> Result<()> {
    let device = Device::Cpu;

    let eingabe = Tensor::from_slice(&[1.0_f32, 0.5, -1.0, 2.0], (2, 2), &device)?;
    let gewichte = Tensor::from_slice(&[0.8_f32, -0.2, 0.4, 0.9], (2, 2), &device)?;
    let bias = Tensor::from_slice(&[0.1_f32, -0.3], (1, 2), &device)?;

    let ausgabe = lineare_schicht_relu(&eingabe, &gewichte, &bias)?;
    println!("Candle-Ausgabe (2x2):\n{}", ausgabe);

    let (loss, grad) = autograd_beispiel(5.0)?;
    println!("Autograd bei x=5.0: loss={:.2}, df/dx={:.2}", loss, grad);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candle_matmul_und_broadcast() -> Result<()> {
        let dev = Device::Cpu;
        let x = Tensor::from_slice(&[1.0_f32, 2.0], (1, 2), &dev)?;
        let w = Tensor::from_slice(&[3.0_f32, 4.0], (2, 1), &dev)?;
        let b = Tensor::from_slice(&[1.0_f32], (1, 1), &dev)?;
        let y = lineare_schicht_relu(&x, &w, &b)?;
        // 1*3 + 2*4 + 1 = 12
        let werte = y.to_vec2::<f32>()?;
        assert!((werte[0][0] - 12.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn candle_relu_kappt_negative() -> Result<()> {
        let dev = Device::Cpu;
        let x = Tensor::from_slice(&[-2.0_f32, 1.0], (1, 2), &dev)?;
        let w = Tensor::from_slice(&[1.0_f32, 0.0, 0.0, 1.0], (2, 2), &dev)?;
        let b = Tensor::zeros((1, 2), candle_core::DType::F32, &dev)?;
        let y = lineare_schicht_relu(&x, &w, &b)?;
        let werte = y.to_vec2::<f32>()?;
        assert_eq!(werte[0], vec![0.0, 1.0]);
        Ok(())
    }

    #[test]
    fn candle_autograd_berechnet_gradient() -> Result<()> {
        // f(x) = (x - 3)^2 -> f(5) = 4, f'(5) = 2*(5-3) = 4
        let (loss, grad) = autograd_beispiel(5.0)?;
        assert!((loss - 4.0).abs() < 1e-5);
        assert!((grad - 4.0).abs() < 1e-5);
        Ok(())
    }
}
