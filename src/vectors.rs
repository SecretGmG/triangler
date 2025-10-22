use std::ops::{Add, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}
impl Vec3{
    pub const ZERO : Vec3 = Vec3{x:0.0,y:0.0,z:0.0};
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn squared(&self) -> f64 {
        self.dot(self)
    }
}

// Rust operator overloading
impl Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}
impl Mul<f64> for Vec3{
    type Output = Self;
    fn mul(self, rhs: f64) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LVec {
    pub t: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}
impl LVec{
    pub const ZERO : LVec = LVec{t:0.0,x:0.0,y:0.0,z:0.0};
}

impl LVec {
    pub fn new(t: f64, x: f64, y: f64, z: f64) -> Self {
        Self { t, x, y, z }
    }

    pub fn spatial(&self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }

    pub fn dot(&self, other: &Self) -> f64 {
        self.t * other.t - (self.x * other.x + self.y * other.y + self.z * other.z)
    }

    pub fn squared(&self) -> f64 {
        self.dot(self)
    }

    pub fn t(&self) -> f64 {
        self.t
    }
}

// Rust operator overloading
impl Add for LVec {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            t: self.t + rhs.t,
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl Sub for LVec {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            t: self.t - rhs.t,
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Neg for LVec {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self {
            t: -self.t,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}
impl Mul<f64> for LVec{
    type Output = Self;
    fn mul(self, rhs: f64) -> Self::Output {
        Self {
            t: self.t * rhs,
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}