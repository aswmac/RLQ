//
//  RLQ.swift
//  RLQ
//
//  Created 2025.06.27.140857
//

import MLX
import MLXLinalg
internal import RealModule // for Float32.sqrt(nn)

struct RLQ {
	var pid: MLXArray // Int
	var row: MLXArray // Float
	var corow: MLXArray // SQUARE matrix, float
	var reddim: Int = 0
	let machineEpsilon: Float32 = 1.0e-15 // If just Float, then Float64...MLX ERROR
	init(rows: Int, cols: Int) {
		self.pid = MLXArray.eye(rows, m: cols, k: 0, dtype: .int32) // set the diagonal elements to unity
		self.row = MLXArray.eye(rows, m: cols, k: 0, dtype: .float32) // set the diagonal elements to unity
		self.corow = MLXArray.eye(cols, m: cols, k: 0, dtype: .float32)
	}
	
	var rows: Int { return Int(self.pid.shape[0]) }
	var cols: Int { return Int(self.pid.shape[1]) }
	
	init(_ input: MLXArray) {
		self.pid = input
		let cols = input.shape[1] // "self not available yet"
		//self.row = self.pid.asMLXArray(dtype: .float32) // _DOES NOT_ COPY-CONVERT the integers to floats -- THEY STAY INT!!!!
		self.row = self.pid.asType(.float32)
		self.corow = MLXArray.eye(cols, m: cols, k: 0, dtype: .float32)
		assert(self.corow.dtype == .float32, "corow dtype is not float32")
		assert(self.row.dtype == .float32, "row dtype is not float32")
	}
	
	mutating func reset(reorder: Bool = true) {
		self.row = self.pid.asMLXArray(dtype: .float32) // COPY-CONVERT the integers to floats
		self.corow = MLXArray.eye(cols, m: cols, k: 0, dtype: .float32)
		if reorder {
			self.lq()
		}
	}
	
	mutating func lq(dim: Int? = nil) {
		let dim = dim ?? self.reddim
		for i in 0..<dim {
			var mn: Int?
			var mni: Int?
			for j in i..<dim {
				let t: Int = (sum(pid[i..<dim]*pid[i..<dim])).item() // sum returns MLXArray, item() because want Int
				if mn == nil || t < mn! {
					mn = t
					mni = j
				}
			}
			if mni != nil {
				self.rowswap(i, mni!)
			}
			self.houseRow(i, i)
		}
		for i in dim..<self.rows { // now do the rest of the rows as well
			self.houseRow(i, dim)
		}
	}
	
	/// zero across the row of the row matrix
	mutating func houseRow(_ ir: Int, _ ic: Int) {
		let kx = self.row[ir][ic..<self.cols]
	
		//let nn = rmsNorm(kx, weight: .ones(like: kx), eps: machineEpsilon) // NO! this is x broadcast-divided by (L_2 / sqrt(dim))
		let nn = MLXLinalg.norm(kx, ord: 2)
		let squaredim = self.cols - ic
		let e0 = eye(1, m: squaredim, k: 0, dtype: .float32)
		let vk = kx - nn*e0
		let vn = MLXLinalg.norm(vk, ord: 2, stream: .cpu) // MLX error: [linalg::svd] This op is not yet supported on the GPU. Explicitly pass a CPU stream to run it.
		let v = vk/vn
		let xx = outer(v, v) // outer product, (self.cols - ic - 1) square
		let e = eye(squaredim, m: squaredim, k: 0, dtype: .float32)
		let q = e - 2*xx
		self.row[0..<self.rows,ic..<self.cols] = self.row[0..<self.rows,ic..<self.cols].matmul(q)
		self.corow[0..<self.cols,ic..<self.cols] = self.corow[0..<self.cols,ic..<self.cols].matmul(q) // corow is cols by cols
	}
	
	func dot(row1 r1: Int, row2 r2: Int, from c1: Int, to c2: Int) -> Float32 {
		let urow_i: Float32 = (sum(self.row[r1][c1..<c2]*self.row[r2][c1..<c2])).item()
		return urow_i
	}
	
	func intArray() -> [[Int]] {
		var integers: [[Int]] = []
		let shape = self.pid.shape
		let rows = Int(shape[0])
		let cols = Int(shape[1])
		for row in 0..<rows {
			integers.append(Array(repeating: 0, count: cols))
			for col in 0..<cols {
				let element: Int = self.pid[row][col].item()
				integers[row][col] = element
			}
		}
		return integers
	}
		
	mutating func colzeroPass(col: Int = 0, row: Int = 0) -> Bool {
		// reduce using no divides per-se down the column
//		guard row < self.pid.shape[0] else { // index bounds check
//			return
//		}
//		guard col < self.pid.shape[1] else { // index bounds check
//			return
//		}
		// get the minimum non-zero element at or below the row in the column
		//let mnz = self.pid[row..<self.pid.shape[0]].min() // this does what...?
		
		// find the index for the minimum non-zero element at or below the row in the column
		var mn: Int32 = Int32.max // big number, testing if 999999999 or Int32.max gives an error...
		var mni: Int?
		for r in row..<self.pid.shape[0] {
			let p: Int32 = abs(self.pid[r][col].item())
			if p > 0 && (mni == nil || p < mn) {
				mn = p
				mni = r
			}
		}
		if mni == nil { return false }
		let minIndex = mni!
		if minIndex != row { self.rowswap(row, minIndex) }
		var change: Bool = false

		for r in (row+1)..<self.pid.shape[0] {
			let num: Int32 = self.pid[r][col].item()
			let den: Int32 = self.pid[row][col].item()
			let kf = Float(2*num + den)/Float(2*den)
			let k = Int32(kf.rounded(.down))
			//let k = (2*num + den).di/(2*den) // TODO: num=-1 and den=1 should give 1 but does not...
			print("num=\(num), den=\(den), k=\(k)")
			if k == 0 { print("K HOW? r=\(r)") }
			guard k != 0 else { continue }
			change = true
			//if r == minIndex { print("HOW!?!?!?") }
			self.rowSubPlace(r, minusRow: row, times: k) // TODO: confirm rounding is to the neares for k = num/den
		}
		return change
	}
	
	mutating func rowSubPlace(_ r1: Int, minusRow r2: Int, times k: Int32) {
		// row r1 gets minus k times r2, i.e.
		// row r1 becomes r1 - k*r2, corow r2 becomes r2 + k*r1
		if r1 == r2 { return }
		if k == 0 { return }
		self.pid[r1] = self.pid[r1] - k*self.pid[r2]
		
	}
	mutating func rowswap(_ r1: Int, _ r2: Int, preserveLQ: Bool = true) {
		let temprow = self.pid[r1]
		self.pid[r1] = self.pid[r2]
		self.pid[r2] = temprow
	}
	
	mutating func rowslide(_ r1: Int, _ r2: Int, preserveLQ: Bool = true) {
		if r1 < r2 {
			for i in stride(from: r1, to: r2, by: 1){
				self.rowswap(i, i+1)
			}
		} else if r2 < r1 {
			for i in stride(from: r1, to: r2, by: -1){
				self.rowswap(i, i-1)
			}
		}
	}
}
