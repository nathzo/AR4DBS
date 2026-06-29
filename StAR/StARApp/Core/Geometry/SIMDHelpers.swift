import simd

/// Small simd conveniences shared by the geometry/registration layer.
/// Owned by WP1.

public extension SIMD4 {
    /// The first three lanes as a SIMD3 (drops `w`).
    @inlinable var xyz: SIMD3<Scalar> { SIMD3(x, y, z) }
}

public extension simd_float4x4 {
    /// The upper-left 3×3 rotation block.
    @inlinable var upperLeft3x3: simd_float3x3 {
        simd_float3x3(columns: (columns.0.xyz, columns.1.xyz, columns.2.xyz))
    }

    /// The translation (4th column, xyz).
    @inlinable var translation: SIMD3<Float> { columns.3.xyz }

    /// Build a rigid transform from a rotation quaternion and a translation.
    init(rotation q: simd_quatf, translation t: SIMD3<Float>) {
        let r = simd_float3x3(q)
        self.init(columns: (
            SIMD4<Float>(r.columns.0.x, r.columns.0.y, r.columns.0.z, 0),
            SIMD4<Float>(r.columns.1.x, r.columns.1.y, r.columns.1.z, 0),
            SIMD4<Float>(r.columns.2.x, r.columns.2.y, r.columns.2.z, 0),
            SIMD4<Float>(t.x, t.y, t.z, 1)
        ))
    }
}

public enum Rotations {
    /// Rotation of `angle` radians about the +Y axis (legacy `Ry(θ)`).
    public static func aboutY(_ angle: Float) -> simd_quatf {
        simd_quatf(angle: angle, axis: SIMD3(0, 1, 0))
    }
}
