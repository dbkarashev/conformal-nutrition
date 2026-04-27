// NutritionInferencePipeline.swift
//
// Гибридный inference: визуальный и текстовый энкодеры через onnxruntime-objc
// (CoreML EP включает ANE), CQR-голова через нативный CoreML.
// Возвращает 90% интервал по 4 нутриентам для одного блюда.
//
// Зависимости (SPM):
//   onnxruntime-objc:  https://github.com/microsoft/onnxruntime
//   swift-transformers: https://github.com/huggingface/swift-transformers
//
// Ресурсы в Bundle:
//   visual_dinov2_small.onnx     — из output ноутбука 05
//   text_minilm_l6_v2.onnx       — из output ноутбука 05
//   cqr_head.mlpackage           — из output скрипта convert_to_coreml
//   normalization.json           — target_norm.json из ноутбука 02
//   conformal_quantiles.json     — из ноутбука 04 (поле cqr_q)
//   tokenizer/                   — содержимое HF tokenizer для MiniLM-L6-v2

import CoreML
import Foundation
import UIKit
import Vision
import onnxruntime_objc
import Tokenizers

public struct NutritionInterval {
    public let target: String
    public let lower: Float
    public let upper: Float
}

public enum InferenceError: Error {
    case missingResource(String)
    case preprocessingFailed
    case modelOutputMissing(String)
}

public final class NutritionInferencePipeline {
    public static let targets = ["total_calories", "total_fat", "total_carb", "total_protein"]
    private static let imageSize = 224
    private static let textMaxLen = 64

    // ImageNet нормализация — DINOv2 учился именно с такими mean/std.
    private static let imageMean: [Float] = [0.485, 0.456, 0.406]
    private static let imageStd:  [Float] = [0.229, 0.224, 0.225]

    private let ortEnv: OrtEnv
    private let visualSession: OrtSession
    private let textSession: OrtSession
    private let cqrHead: MLModel
    private let tokenizer: any Tokenizer

    private let targetMean: [Float]   // shape [4]
    private let targetStd: [Float]    // shape [4]
    private let conformalQ: [Float]   // shape [4]

    public init(bundle: Bundle = .main) async throws {
        guard
            let visualURL = bundle.url(forResource: "visual_dinov2_small", withExtension: "onnx"),
            let textURL = bundle.url(forResource: "text_minilm_l6_v2", withExtension: "onnx"),
            let cqrURL = bundle.url(forResource: "cqr_head", withExtension: "mlpackage"),
            let normURL = bundle.url(forResource: "normalization", withExtension: "json"),
            let qURL = bundle.url(forResource: "conformal_quantiles", withExtension: "json")
        else { throw InferenceError.missingResource("одного из onnx/mlpackage/json") }

        ortEnv = try OrtEnv(loggingLevel: .warning)

        let opts = try OrtSessionOptions()
        try opts.appendCoreMLExecutionProvider(with: OrtCoreMLProviderOptions())
        visualSession = try OrtSession(env: ortEnv, modelPath: visualURL.path, sessionOptions: opts)
        textSession = try OrtSession(env: ortEnv, modelPath: textURL.path, sessionOptions: opts)

        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all
        cqrHead = try MLModel(contentsOf: cqrURL, configuration: cfg)

        tokenizer = try await AutoTokenizer.from(modelFolder: bundle.url(forResource: "tokenizer", withExtension: nil)!)

        let normData = try Data(contentsOf: normURL)
        let norm = try JSONDecoder().decode([String: [Float]].self, from: normData)
        guard let m = norm["mean"], let s = norm["std"], m.count == 4, s.count == 4 else {
            throw InferenceError.missingResource("normalization.json {mean,std} ожидаются 4 числа")
        }
        targetMean = m
        targetStd = s

        let qData = try Data(contentsOf: qURL)
        let qDict = try JSONDecoder().decode([String: Float].self, from: qData)
        conformalQ = Self.targets.map { qDict[$0] ?? 0 }
    }

    public func predict(image: UIImage, ingredients: String) throws -> [NutritionInterval] {
        let visEmbedding = try encodeImage(image)
        let txtEmbedding = try encodeText(ingredients)
        let quantiles = try runCqrHead(v: visEmbedding, t: txtEmbedding)
        return assemble(quantiles: quantiles)
    }

    // MARK: - image

    private func encodeImage(_ image: UIImage) throws -> [Float] {
        guard let pixels = preprocessToFloatBuffer(image) else {
            throw InferenceError.preprocessingFailed
        }
        let shape: [NSNumber] = [1, 3, NSNumber(value: Self.imageSize), NSNumber(value: Self.imageSize)]
        let data = NSMutableData(bytes: pixels, length: pixels.count * MemoryLayout<Float>.size)
        let input = try OrtValue(tensorData: data, elementType: .float, shape: shape)
        let outputs = try visualSession.run(
            withInputs: ["pixel_values": input],
            outputNames: Set(["embedding"]),
            runOptions: nil
        )
        guard let out = outputs["embedding"] else {
            throw InferenceError.modelOutputMissing("visual.embedding")
        }
        return try out.tensorDataAsArray(elementType: .float) as [Float]
    }

    private func preprocessToFloatBuffer(_ image: UIImage) -> [Float]? {
        let size = CGSize(width: Self.imageSize, height: Self.imageSize)
        UIGraphicsBeginImageContextWithOptions(size, true, 1.0)
        image.draw(in: CGRect(origin: .zero, size: size))
        guard let resized = UIGraphicsGetImageFromCurrentImageContext()?.cgImage else {
            UIGraphicsEndImageContext(); return nil
        }
        UIGraphicsEndImageContext()

        let w = Self.imageSize, h = Self.imageSize
        var raw = [UInt8](repeating: 0, count: w * h * 4)
        let space = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(data: &raw, width: w, height: h,
                            bitsPerComponent: 8, bytesPerRow: w * 4,
                            space: space,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        ctx?.draw(resized, in: CGRect(x: 0, y: 0, width: w, height: h))

        // CHW float32, нормализация ImageNet.
        var out = [Float](repeating: 0, count: 3 * h * w)
        for y in 0..<h {
            for x in 0..<w {
                let p = (y * w + x) * 4
                for c in 0..<3 {
                    let v = Float(raw[p + c]) / 255.0
                    out[c * h * w + y * w + x] = (v - Self.imageMean[c]) / Self.imageStd[c]
                }
            }
        }
        return out
    }

    // MARK: - text

    private func encodeText(_ text: String) throws -> [Float] {
        var ids = tokenizer.encode(text: text)
        if ids.count > Self.textMaxLen { ids = Array(ids.prefix(Self.textMaxLen)) }
        var mask = Array(repeating: Int64(1), count: ids.count)
        while ids.count < Self.textMaxLen {
            ids.append(tokenizer.padTokenId ?? 0)
            mask.append(0)
        }
        let ids64 = ids.map { Int64($0) }

        let shape: [NSNumber] = [1, NSNumber(value: Self.textMaxLen)]
        let idData = NSMutableData(bytes: ids64, length: ids64.count * MemoryLayout<Int64>.size)
        let mkData = NSMutableData(bytes: mask, length: mask.count * MemoryLayout<Int64>.size)
        let inIds = try OrtValue(tensorData: idData, elementType: .int64, shape: shape)
        let inMask = try OrtValue(tensorData: mkData, elementType: .int64, shape: shape)

        let outputs = try textSession.run(
            withInputs: ["input_ids": inIds, "attention_mask": inMask],
            outputNames: Set(["embedding"]),
            runOptions: nil
        )
        guard let out = outputs["embedding"] else {
            throw InferenceError.modelOutputMissing("text.embedding")
        }
        return try out.tensorDataAsArray(elementType: .float) as [Float]
    }

    // MARK: - head

    private func runCqrHead(v: [Float], t: [Float]) throws -> [[Float]] {
        let vArr = try MLMultiArray(shape: [1, 384], dataType: .float32)
        let tArr = try MLMultiArray(shape: [1, 384], dataType: .float32)
        for i in 0..<384 {
            vArr[i] = NSNumber(value: v[i])
            tArr[i] = NSNumber(value: t[i])
        }
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "v": MLFeatureValue(multiArray: vArr),
            "t": MLFeatureValue(multiArray: tArr),
        ])
        let prediction = try cqrHead.prediction(from: input)
        let outName = prediction.featureNames.first { $0.contains("var") || $0 == "quantiles" }
            ?? prediction.featureNames.first!
        guard let q = prediction.featureValue(for: outName)?.multiArrayValue else {
            throw InferenceError.modelOutputMissing("cqr.\(outName)")
        }
        // shape (1, 4, 3) — переводим в [[lo, med, hi]] на 4 цели.
        var rows: [[Float]] = []
        for tIdx in 0..<4 {
            var trio = [Float](repeating: 0, count: 3)
            for qIdx in 0..<3 {
                let flat = tIdx * 3 + qIdx
                trio[qIdx] = q[flat].floatValue
            }
            rows.append(trio)
        }
        return rows
    }

    // MARK: - assemble

    private func assemble(quantiles: [[Float]]) -> [NutritionInterval] {
        var out: [NutritionInterval] = []
        for (i, t) in Self.targets.enumerated() {
            let lo = quantiles[i][0] * targetStd[i] + targetMean[i] - conformalQ[i]
            let hi = quantiles[i][2] * targetStd[i] + targetMean[i] + conformalQ[i]
            out.append(NutritionInterval(target: t, lower: lo, upper: hi))
        }
        return out
    }
}
