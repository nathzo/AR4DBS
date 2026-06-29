//  ArucoDictionary.swift
//  B2 — native DICT_4X4_50 marker dictionary (no OpenCV on device).
//
//  The 50 canonical 4x4 marker codes, extracted bit-exact from OpenCV's
//  cv2.aruco.DICT_4X4_50 so the on-device decoder matches the surgeon's
//  physical ArUco/AprilTags. Each code packs the 4x4 data cells row-major
//  into the low 16 bits of a UInt16 (bit r*4+c; white cell = 1). The black
//  border and quiet zone are validated by the decoder, not stored here.
//
//  Generated from OpenCV 4.13.0. Do not hand-edit.

enum ArucoDictionary {
    /// Data region side (cells).
    static let gridSize = 4
    /// Max bit errors tolerated when matching (OpenCV DICT_4X4_50 value).
    static let maxCorrectionBits = 1

    /// Canonical (rotation-0) code per marker id, packed row-major (white=1).
    static let codes: [UInt16] = [
        0x4CAD, 0x59F0, 0xB4CC, 0x6299, 0x792A, 0xB39E, 0x7479, 0x4F23, 0x5B7F, 0x6AF3,
        0x899F, 0xE588, 0xED70, 0xF054, 0x8D24, 0x7C64, 0xA662, 0x0066, 0x7A36, 0xF56E,
        0xD161, 0xD40D, 0xAB33, 0x41BB, 0xE27F, 0x8E29, 0x2735, 0x2AA5, 0xC484, 0xF62C,
        0xA822, 0x4DEA, 0xF379, 0xD30F, 0x7510, 0x9490, 0xAE18, 0xFF20, 0x6FB0, 0x5A38,
        0x18E8, 0x1454, 0x314C, 0x4D1C, 0x1724, 0xD774, 0xFCB4, 0x26D2, 0x740A, 0xC80A,
    ]
}
