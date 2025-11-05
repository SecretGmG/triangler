use num_complex::Complex64;
use triangler::{LVec, Vec3, integrands::TriangleIntegrandBuilder};

#[test]
fn sample(){
    let integrand_builder = TriangleIntegrandBuilder::new(
        LVec::new(0.005, 0.0, 0.0,  0.005),
        LVec::new(0.005, 0.0, 0.0, -0.005),
        Complex64::from(0.02));

     let val = integrand_builder.basic_ltd()(Vec3::new(0.1, 0.2, 0.3));
     
     println!("value: {val}")

}