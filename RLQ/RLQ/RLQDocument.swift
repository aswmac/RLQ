//
//  RLQDocument.swift
//  RLQ
//
//  Created 2025.06.27.140857
//

import SwiftUI
import UniformTypeIdentifiers
import MLX

nonisolated struct RLQDocument: FileDocument {
	var mat: MLXArray
	
	
	
	init() {
		self.mat = MLXArray.eye(30, m: 31, k: 0, dtype: .int32)
	}
	
	static let readableContentTypes = [
		UTType(importedAs: "com.example.plain-text")
	]
	
	init(configuration: ReadConfiguration) throws {
		if let data = configuration.file.regularFileContents {
			let matData = try JSONDecoder().decode(Array<Int>.self, from: data)
			let flat = MLXArray(Array(matData[2...]))
			self.mat = flat.reshaped([matData[0], matData[1]])
		} else {
			throw CocoaError(.fileReadCorruptFile)
		}
	}
		
	func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
		let shape = self.mat.shape
		guard shape.count == 2 else {
			fatalError("Unsupported matrix shape: \(shape)")
		}
		
		let rows = shape[0]
		let cols = shape[1]
		var data: [Int] = [rows, cols]
		for row in 0..<rows {
			for col in 0..<cols {
				let element: Int = self.mat[row, col].item()
				data.append(element)
			}
		}
		let cdata = try JSONEncoder().encode(data)
		return FileWrapper(regularFileWithContents: cdata)
	}
}
