#[cfg(feature = "python")]
use pyo3::prelude::*;
use std::ops::{Add, Sub, Neg};

#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone, Copy, Debug)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}
impl Vec3{
    pub fn zero()->Self{
        Self::new(0.0,0.0,0.0)
    }
}

#[cfg_attr(feature = "python", pymethods)]
impl Vec3 {
    #[new]
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

#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone, Copy, Debug)]
pub struct LVec {
    pub t: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}
impl LVec{
    pub fn zero()->Self{
        Self::new(0.0,0.0,0.0,0.0)
    }
}

#[cfg_attr(feature = "python", pymethods)]
impl LVec {
    #[new]
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

    // Python dunder methods
    fn __add__(&self, other: &Self) -> Self {
        *self + *other
    }

    fn __sub__(&self, other: &Self) -> Self {
        *self - *other
    }

    fn __neg__(&self) -> Self {
        -*self
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
