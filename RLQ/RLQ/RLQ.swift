//
//  RLQ.swift
//  RLQ
//
//  Created 2025.06.27.140857
//

import MLX
import MLXLinalg

struct RLQ {
	var grid = MLXArray([])
	let machineEpsilon: Float32 = 1.0e-15 // If just Float, then Float64...MLX ERROR
	init(rows: Int, cols: Int) {
		self.grid = MLXArray.eye(rows, m: cols, k: 0, dtype: .int32) // set the diagonal elements to unity
	}
	
	init(_ input: MLXArray) {
		self.grid = input
	}
	
	func intArray() -> [[Int]] {
		var integers: [[Int]] = []
		let shape = self.grid.shape
		let rows = Int(shape[0])
		let cols = Int(shape[1])
		for row in 0..<rows {
			integers.append(Array(repeating: 0, count: cols))
			for col in 0..<cols {
				let element: Int = self.grid[row][col].item()
				integers[row][col] = element
			}
		}
		return integers
	}
		
	func colzeroPass(col: Int = 0, row: Int = 0) {
		//print("I just print!")
		// reduce using no divides per-se down the column from rm index if given, from index 0 if rm not given
		guard row < self.grid.shape[0] else {
			return
		}
		guard col < self.grid.shape[1] else {
			return
		}
		if abs(self.grid[row][col].item()) < machineEpsilon {
			return
		}
		let pivotRow = self.grid[row]
		for r in (row+1)..<self.grid.shape[0] {
			let num: Float16 = self.grid[r][col].item()
			let den: Float16 = self.grid[row][col].item()
			let k: Int = Int((num / den).rounded(.toNearestOrAwayFromZero))
			let newRow = self.grid[r] - k * pivotRow
			self.grid[r] = newRow
			//debugPrint("row \(r) reduced by \(k), newRow: \(newRow)")
		}
		
	}
	
}
