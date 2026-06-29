//  RealityKitOverlay.swift
//  WP3 — RealityKit overlay & occlusion (implements OverlayRendering).
//
//  Ported from the legacy C++ overlay/occlusion pipeline:
//    - core/rendering/OverlayRenderer.{h,cpp}      (drawSegment/drawTargetMarker/drawIncisionMarker, Style)
//    - app/AppController.cpp                        (renderOverlayOnto, renderWithOcclusion,
//                                                     findIncisionPoint, checkIncisionQuality,
//                                                     the LockedIncision streak)
//    - app/AppController.h                          (kIncisionLockFrames = 5, kIncisionLockRadius = 0.003 m)
//
//  The legacy code projected 3-D Leksell geometry onto a 2-D cv::Mat with QPainter
//  and sampled a MiDaS/LiDAR depth map by hand to occlude geometry behind the head
//  and to find the incision point. Here that is replaced entirely:
//    * Geometry is real RealityKit entities placed under an AnchorEntity at
//      world_T_leksell, so ARKit handles projection.
//    * Occlusion is done by the LiDAR scene mesh + RealityKit occlusion
//      (sceneUnderstanding.options.insert(.occlusion)); the reconstructed mesh
//      occludes overlay geometry behind the physical head. This REPLACES the
//      manual depth-sampling occlusion (findIncisionPoint's depth walk / `visible`).
//    * The incision point is found by raycasting the LiDAR scene mesh from the deep
//      target along the trajectory direction (the line target→skull exit), then the
//      LockedIncision streak/lock is ported verbatim (5 frames within 3 mm).
//
//  Single module "StAR": all Core/* types are visible. Geometry is METRES.

import RealityKit
import ARKit
import simd
import Combine   // Cancellable (scene event subscription)
import SwiftUI   // Color
import UIKit     // UIColor

// MARK: - Bridge to WP7

/// WP7 conforms its ARView-handle class to this so WP3 can reach the live `ARView`
/// without importing WP7. There is ONE ARSession (the ARView's); WP3 never makes one.
@MainActor
public protocol ARViewProviding: ARViewRepresentableHandle {
    var arView: ARView { get }
}

// MARK: - Overlay

@MainActor
public final class RealityKitOverlay: OverlayRendering {

    // Legacy constants (app/AppController.h).
    private static let kIncisionLockFrames = 5      // consecutive qualifying frames to lock
    private static let kIncisionLockRadius: Float = 0.003   // 3 mm coherence window (metres)

    private static let targetSphereRadius:   Float = 0.004  // ~legacy target marker
    private static let incisionSphereRadius: Float = 0.0035

    private weak var arView: ARView?
    private var rootAnchor: AnchorEntity?

    // Mesh caches: rebuild children only when geometry or style changed.
    private var lastGeometry: PlanGeometry?
    private var lastStyle: RenderStyle?
    /// Last locked pose, cached so the per-frame tick can re-run the incision logic.
    private var lastWorldToLeksell: simd_float4x4?
    /// Subscription to the RealityKit render loop. ARScreen only calls update() on a
    /// registration-state CHANGE, so without a per-frame tick the incision raycast
    /// would fire exactly once at lock and the 5-frame lock streak could never
    /// complete (audit S3-01). This drives updateSide() every rendered frame.
    private var sceneUpdateSub: (any Cancellable)?

    // Per-side rendering entities (reused; only their mesh/material/transform swap).
    private struct SideEntities {
        let lineEntity:     ModelEntity
        let targetEntity:   ModelEntity
        let incisionEntity: ModelEntity
    }
    private var leftSide:  SideEntities?
    private var rightSide: SideEntities?

    // Per-side incision lock state (ports app/AppController.h `LockedIncision`).
    // Stored in LEKSELL space (under rootAnchor) so it stays fixed across frames.
    private struct LockedIncision {
        var leksellPt: SIMD3<Float> = .zero
        var locked: Bool = false
        var streakCount: Int = 0
        var streakSum: SIMD3<Float> = .zero
    }
    private var leftLock  = LockedIncision()
    private var rightLock = LockedIncision()

    public init() {}

    // MARK: OverlayRendering

    public func attach(to arView: ARViewRepresentableHandle) {
        guard let provider = arView as? ARViewProviding else { return }
        let view = provider.arView
        self.arView = view
        // Enable mesh occlusion: the LiDAR scene mesh (sceneReconstruction=.mesh set
        // by WP2) now occludes overlay geometry behind the head. Replaces manual
        // depth-sampling occlusion.
        view.environment.sceneUnderstanding.options.insert(.occlusion)

        // Re-run the per-frame side logic (incision raycast + lock streak + line
        // clamp) every rendered frame. RealityKit scene events fire on the main
        // thread, so MainActor.assumeIsolated is sound here.
        sceneUpdateSub?.cancel()
        sceneUpdateSub = view.scene.subscribe(to: SceneEvents.Update.self) { [weak self] _ in
            MainActor.assumeIsolated { self?.onSceneUpdate() }
        }
    }

    /// Per-frame tick from the RealityKit render loop. Re-runs the incision
    /// raycast/lock and line clamp from the last locked inputs so the lock streak
    /// accumulates across frames (audit S3-01). No-ops until update() supplied inputs
    /// and while there is no active anchor (e.g. after clear()).
    private func onSceneUpdate() {
        guard let w = lastWorldToLeksell,
              let g = lastGeometry,
              let s = lastStyle,
              rootAnchor != nil else { return }
        updateSide(.left,  trajectory: g.left,  entities: leftSide,
                   lock: &leftLock,  worldToLeksell: w, style: s)
        updateSide(.right, trajectory: g.right, entities: rightSide,
                   lock: &rightLock, worldToLeksell: w, style: s)
    }

    public func update(worldToLeksell: simd_float4x4,
                       geometry: PlanGeometry,
                       style: RenderStyle) {
        guard let arView else { return }

        // Ensure the Leksell root anchor exists (added to the scene once).
        let root: AnchorEntity
        if let existing = rootAnchor {
            root = existing
        } else {
            let a = AnchorEntity(world: .zero)
            arView.scene.addAnchor(a)
            rootAnchor = a
            root = a
        }

        // Cheap pose update every frame: child entities live in Leksell metres under
        // this anchor, so child positions == Leksell coordinates directly.
        root.transform = Transform(matrix: worldToLeksell)

        // Rebuild meshes only when geometry or style changed; otherwise reuse.
        let needsRebuild = (geometry != lastGeometry) || (style != lastStyle)
        if needsRebuild {
            rebuildSides(in: root, geometry: geometry, style: style)
            lastGeometry = geometry
            lastStyle = style
        }

        // Per-side incision raycast + lock + line clamp (every frame).
        updateSide(.left,  trajectory: geometry.left,  entities: leftSide,
                   lock: &leftLock,  worldToLeksell: worldToLeksell, style: style)
        updateSide(.right, trajectory: geometry.right, entities: rightSide,
                   lock: &rightLock, worldToLeksell: worldToLeksell, style: style)

        // Cache for the per-frame tick (onSceneUpdate) so the incision lock streak
        // keeps accumulating between registration-change-driven update() calls.
        lastWorldToLeksell = worldToLeksell
    }

    public func clear() {
        if let root = rootAnchor, let arView { arView.scene.removeAnchor(root) }
        rootAnchor = nil
        leftSide = nil
        rightSide = nil
        lastGeometry = nil
        lastStyle = nil
        leftLock  = LockedIncision()  // Recalibrer path: reset incision locks.
        rightLock = LockedIncision()
    }

    // MARK: - Mesh build (only on geometry/style change)

    private enum Side { case left, right }

    private func rebuildSides(in root: AnchorEntity,
                              geometry: PlanGeometry,
                              style: RenderStyle) {
        // Tear down old child entities, then rebuild present sides.
        if let s = leftSide  { detach(s) }
        if let s = rightSide { detach(s) }
        leftSide  = geometry.left  != nil ? buildSide(in: root, style: style) : nil
        rightSide = geometry.right != nil ? buildSide(in: root, style: style) : nil
    }

    private func detach(_ s: SideEntities) {
        s.lineEntity.removeFromParent()
        s.targetEntity.removeFromParent()
        s.incisionEntity.removeFromParent()
    }

    private func buildSide(in root: AnchorEntity, style: RenderStyle) -> SideEntities {
        let lineColor     = UIColor(Color(hex: style.lineColorHex))
        let targetColor   = UIColor(Color(hex: style.targetColorHex))
        let incisionColor = UIColor(Color(hex: style.incisionColorHex))

        // Trajectory line: a unit-height cylinder centred on local +Y; per-frame
        // we re-scale/orient/position it (so a 1 m cylinder is the canonical mesh).
        let radius = (style.lineWidthMM / 1000) / 2
        let lineMesh = MeshResource.generateCylinder(height: 1, radius: max(radius, 1e-4))
        let lineEntity = ModelEntity(mesh: lineMesh,
                                     materials: [SimpleMaterial(color: lineColor, isMetallic: false)])

        let targetEntity = ModelEntity(
            mesh: .generateSphere(radius: Self.targetSphereRadius),
            materials: [SimpleMaterial(color: targetColor, isMetallic: false)])

        let incisionEntity = ModelEntity(
            mesh: .generateSphere(radius: Self.incisionSphereRadius),
            materials: [SimpleMaterial(color: incisionColor, isMetallic: false)])
        incisionEntity.isEnabled = false   // hidden until an incision point is found

        root.addChild(lineEntity)
        root.addChild(targetEntity)
        root.addChild(incisionEntity)

        return SideEntities(lineEntity: lineEntity,
                            targetEntity: targetEntity,
                            incisionEntity: incisionEntity)
    }

    // MARK: - Per-frame side update

    private func updateSide(_ side: Side,
                            trajectory: Trajectory?,
                            entities: SideEntities?,
                            lock: inout LockedIncision,
                            worldToLeksell: simd_float4x4,
                            style: RenderStyle) {
        guard let trajectory, let entities else { return }

        // Target sphere sits at the deep target (Leksell).
        entities.targetEntity.position = trajectory.target

        // Incision raycast + lock streak (ports renderWithOcclusion's iOS branch).
        if !lock.locked {
            if let hitLeksell = raycastIncision(trajectory: trajectory,
                                                worldToLeksell: worldToLeksell) {
                accumulateLock(&lock, hit: hitLeksell)
                placeIncision(entities, at: hitLeksell)
            } else {
                lock.streakCount = 0
                entities.incisionEntity.isEnabled = false
            }
        } else {
            // Locked: keep the frozen point shown.
            placeIncision(entities, at: lock.leksellPt)
        }

        // Line geometry: clamp to the incision point when locked (legacy t_inc clamp
        // in renderWithOcclusion), else draw the full lineEnd→target segment.
        let drawnEnd: SIMD3<Float>
        if lock.locked {
            let t = IncisionGeometry.parameter(of: lock.leksellPt, on: trajectory)
            let tc = min(max(t, 0), 1)
            // parameter t is measured lineEnd(0)→target(1); the visible part is
            // lineEnd → incision point, so the line runs lineEnd .. clampedPoint.
            drawnEnd = trajectory.lineEnd + (trajectory.target - trajectory.lineEnd) * tc
        } else {
            drawnEnd = trajectory.target
        }
        orientLine(entities.lineEntity,
                   from: trajectory.lineEnd, to: drawnEnd, style: style)
    }

    private func placeIncision(_ entities: SideEntities, at leksellPt: SIMD3<Float>) {
        entities.incisionEntity.position = leksellPt
        entities.incisionEntity.isEnabled = true
    }

    /// Orient/scale the canonical unit cylinder so it spans `a → b` (Leksell metres).
    private func orientLine(_ entity: ModelEntity,
                            from a: SIMD3<Float>,
                            to b: SIMD3<Float>,
                            style: RenderStyle) {
        let delta = b - a
        let height = simd_length(delta)
        guard height > 1e-6 else { entity.isEnabled = false; return }
        entity.isEnabled = true

        let dir = delta / height
        let orientation = simd_quatf(from: SIMD3<Float>(0, 1, 0), to: dir)

        // Canonical mesh is a 1 m cylinder of the configured radius; scale Y to the
        // required height (radius already baked into the mesh, leave X/Z at 1).
        entity.transform = Transform(
            scale: SIMD3<Float>(1, height, 1),
            rotation: orientation,
            translation: (a + b) * 0.5)
    }

    // MARK: - Incision raycast (WORLD space) + lock streak

    /// Raycast the LiDAR scene mesh from the deep target along the trajectory
    /// direction (target→skull) to find where the line meets the head surface.
    /// Returns the nearest hit converted back to LEKSELL space, or nil (sim / no mesh).
    private func raycastIncision(trajectory: Trajectory,
                                 worldToLeksell: simd_float4x4) -> SIMD3<Float>? {
        guard let arView else { return nil }

        // origin_world = worldToLeksell * target ; dir_world = R * direction (normalised).
        let originWorld = (worldToLeksell * SIMD4<Float>(trajectory.target, 1)).xyz
        let dirWorld    = simd_normalize(worldToLeksell.upperLeft3x3 * trajectory.direction)
        guard dirWorld.x.isFinite, dirWorld.y.isFinite, dirWorld.z.isFinite else { return nil }

        // The line from target toward skull is at most |lineEnd - target| long.
        let maxLen = simd_length(trajectory.lineEnd - trajectory.target)
        let length = max(maxLen, 0.001)

        let hits = arView.scene.raycast(origin: originWorld,
                                        direction: dirWorld,
                                        length: length,
                                        query: .nearest,
                                        mask: .sceneUnderstanding,
                                        relativeTo: nil)
        // Graceful no-op when the mesh isn't there yet (simulator / before scan).
        guard let nearest = hits.first else { return nil }

        // Convert the world hit back to Leksell space to store side-locally.
        let hitLeksell = (worldToLeksell.inverse * SIMD4<Float>(nearest.position, 1)).xyz
        return hitLeksell
    }

    /// Ports the LockedIncision streak from renderWithOcclusion: average N consecutive
    /// hits within kIncisionLockRadius, freeze after kIncisionLockFrames.
    private func accumulateLock(_ lock: inout LockedIncision, hit: SIMD3<Float>) {
        if lock.streakCount == 0 {
            lock.streakSum = hit
            lock.streakCount = 1
        } else {
            let avg = lock.streakSum / Float(lock.streakCount)
            let dist = simd_length(hit - avg)
            if dist < Self.kIncisionLockRadius {
                lock.streakSum += hit
                lock.streakCount += 1
            } else {
                lock.streakSum = hit
                lock.streakCount = 1
            }
        }
        if lock.streakCount >= Self.kIncisionLockFrames {
            lock.leksellPt = lock.streakSum / Float(lock.streakCount)
            lock.locked = true
        }
    }
}
