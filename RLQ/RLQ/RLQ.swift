//
//  RLQ.swift
//  RLQ
//
//  Created 2025.06.27.140857
//
//  File: RLQ.swift (see rlq.py for original)
//  Integer lattice (row) reduction.
//  self.pid has integer values
//  Row operations done on self.pid are also done on self.row, as self.row is
//  meant to store the L form of the lower triangle in LQ factoring. ie L*L^T == P*P^T
//  (where P is self.pid). That way any unitary column operations on L do not
//  affect that invariant of L*L^T --> (LQ)*(LQ)^T == L(Q*Q^T)*L^T == L*L^T
//  Q ---- self.corow stores the Q, but keeps it as Q^T so that the same operations
//  can be done to self.corow without transposing concerns
//
//  TLDR: ••••••••••••    R = P*Q    •••••••••••• R and P KEEP SAME dot products between rows
//  -----------------------------------------------------------------------------------------
//  Row operations are integer only and unimodular--applied to pid and row only
//  Col operations are Unitary only--applied to row and corow only
//  corow stores the Q matrix of unitary, so that self.row = self.pid * self.corow, or R = P*Q


import MLX
import MLXLinalg
internal import RealModule // for Float32.sqrt(nn)
import Foundation // for hypot()

// TODO: use driftCount, or drift (error estimations on row) to batch re-alignments like row[i] = pid[i].matmul(corow)
// TODO: also (like if take in high bits truncated of large numbers) can do error analysis for the pid...
struct RLQ {
	var pid: MLXArray // Int
	var row: MLXArray // Float
	var corow: MLXArray // SQUARE matrix, float
	var reddim: Int = 0
	private let machineEpsilon: Float32 = 1.0e-15 // If just Float, then Float64...MLX ERROR
	private let driftThreshold: Float32 = 1.0e-4 // tolerance of error aggregated, triggers a re-alignment doing R = L*Q
	private var driftRow: Float32 = 1.0e-7 // The track of row operations errors that might want a re-alignment like R = LQ or R_i = ...
	init(rows: Int, cols: Int) {
		self.pid = MLXArray.eye(rows, m: cols, k: 0, dtype: .int32) // set the diagonal elements to unity
		self.row = MLXArray.eye(rows, m: cols, k: 0, dtype: .float32) // set the diagonal elements to unity
		self.corow = MLXArray.eye(cols, m: cols, k: 0, dtype: .float32)
		driftRow = machineEpsilon  // instead of always doing row[i] = pid[i] @ Q, batch them with a count
	}
	
	var rows: Int { return Int(self.pid.shape[0]) }
	var cols: Int { return Int(self.pid.shape[1]) }
	
	init(_ input: MLXArray) {
		self.pid = input
		let cols = input.shape[1] // "self not available yet"
		//self.row = self.pid.asMLXArray(dtype: .float32) // _DOES NOT_ COPY-CONVERT the integers to floats -- THEY STAY INT!!!!
		self.row = self.pid.asType(.float32)
		self.corow = MLXArray.eye(cols, m: cols, k: 0, dtype: .float32)
		driftRow = machineEpsilon // Use the count of operations on a row (or maybe a float representing err?)
		assert(self.corow.dtype == .float32, "corow dtype is not float32")
		assert(self.row.dtype == .float32, "row dtype is not float32")
	}
	
	mutating func reset(reorder: Bool = true) {
		self.row = self.pid.asMLXArray(dtype: .float32) // COPY-CONVERT the integers to floats
		self.driftRow = machineEpsilon
		self.corow = MLXArray.eye(cols, m: cols, k: 0, dtype: .float32)
		debugPrint("reset() gives pid[0,1] = \(self.pid[0,1])")
		if reorder { self.lq() }
	}
	
	mutating func lq(dim: Int? = nil) {
		let dim = dim ?? self.reddim
		for i in 0..<dim { // For each row going down, find the smallest norm
			var mn: Int?
			var mni: Int?
			for j in i..<dim { // TODO: want to do linalg norm like houserow's vn = (..., stream: .cpu)...?
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
	
	mutating func digest(start: Int = 0) -> Bool {
		// look at adjacent diagonal values of the LQ form and mix for larger values at the lower end.
		// Assumption: row is in lq form
		let diagEpsilon: Float32 = 1e-6 // or machineEpsilon? or variable input to func?... used to test changes
		var change: Bool = false
		var i = start + 1
		while i < self.reddim {
			let a: Float32 = self.row[i-1][i-1].item()    //<---  | a 0 |
			let e: Float32 = self.row[i][i-1].item()      //<---  | e f |
			let f: Float32 = self.row[i][i].item()        //<--- variables renamed only for reading and typing convenience.
			if abs(a) < diagEpsilon { i += 1; continue }
			let t = e/a
			if abs(abs(t) - 0.5) < diagEpsilon { i += 1; continue }
			let k = Int32(t.rounded(.toNearestOrAwayFromZero))
			if k == 0 { i += 1; continue }
			change = true
			self.rowSubPlace(i, minusRow: i - 1, times: k) // if e gets reduced by a
			for j in stride(from: i - 2, to: -1, by: -1) {
				let aj: Float32 = self.row[j][j].item()      //<---  | a 0 |
				let ej: Float32 = self.row[i][j].item()      //<---  | e f |
				let tj = ej/aj
				if abs(abs(tj) - 0.5) < diagEpsilon { continue } // if the reduction is negligible
				let kj: Int32 = Int32(tj.rounded(.toNearestOrAwayFromZero))
				if kj == 0 { continue } // if there is no reduction at all
				self.rowSubPlace(i, minusRow: j, times: kj)
			}
			let new_e: Float32 = self.row[i][i-1].item()
			if a*a - (new_e*new_e + f*f) > diagEpsilon { // if the reductions found a significantly smaller "upper-low" value
				change = true
				self.givens(row: i, Col0: i-1, col1: i) // inlline of rowSlide which preserves the LQ form
				self.rowswap(i, i - 1)
			} else {
				i += 1
			}
		}
		return change
	}
	
	mutating func digall(_ quality: Float32 = 1.65) -> Int {
		guard self.reddim > 1 else { return 0 } // otherwise dratio would be empty and max would be nil!
		let digallEpsilon: Float32 = 1e-6
		var count = 0
		var rred = true
		debugPrint("Starting rred while loop...")
		while rred {
			rred = false
			//self.reset(reorder: true) // reset(true) includes lq()
			var dratio: [Float32] = []
			for i in 1..<self.reddim {
				let a: Float32 = self.row[i-1][i-1].item()    //<---  | a 0 |
				//let e: Float32 = self.row[i][i-1].item()    //<---  | e f |
				let f: Float32 = self.row[i][i].item()        //<--- variables renamed for convenience.
				dratio.append(abs(f) < digallEpsilon ? 0.0 : abs(a)/abs(f))
			}
			var dmax = dratio.max()!
			debugPrint("dmax initial: \(dmax)")
			while dmax > quality {
				while self.digest() { continue }
				self.reset(reorder: true)
				for i in 1..<self.reddim {
					if abs(self.row[i][i].item()) < digallEpsilon { dratio[i - 1] = 0.0 }
					else { dratio[i - 1] = abs(self.row[i-1, i-1].item()/self.row[i, i].item()) }
				}
				count += 1
				dmax = dratio.max()!
				debugPrint("at count \(count) dmax: \(dmax)")
			}
			debugPrint("rred iteration \(count), dratio: \(dratio), rred: \(rred)")
		}
		return count
	}
	
	mutating func houseDiag(_ ir: Int, _ ic: Int) {
		var r = ir
		var c = ic
		while r < self.rows && c < self.cols {
			self.houseRow(r, c)
			r += 1
			c += 1
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
		let rows = Int(self.rows)
		let cols = Int(self.cols)
		for row in 0..<rows {
			integers.append(Array(repeating: 0, count: cols))
			for col in 0..<cols {
				let element: Int = self.pid[row][col].item()
				integers[row][col] = element
			}
		}
		return integers
	}

	func floatArray() -> [[Float]] {
		var floaters: [[Float]] = []
		let rows = Int(self.rows)
		let cols = Int(self.cols)
		for row in 0..<rows {
			floaters.append(Array(repeating: 0, count: cols))
			for col in 0..<cols {
				let element: Float = self.row[row][col].item()
				floaters[row][col] = element
			}
		}
		return floaters
	}
	
	mutating func colzero(col cm: Int, row rm: Int) {
		while self.colzeroPass(col: cm, row: rm) { continue }
		if self.pid[rm, cm].item() < 0 { self.rowneg(rm) }
	}
	
	// like colzeroPass, but looking at row not pid, and keeping the row not moving it
	mutating func rowDown(from row: Int, col: Int) -> Bool {
		// no search for minimum, just use this element and reduce below
		let den: Float32 = self.row[row][col].item()
		guard abs(den) > machineEpsilon else { return false }
		var change: Bool = false
		for r in row+1..<self.rows {
			//debugPrint("checking \(r)")
			let num: Float32 = self.row[r][col].item()
			let kf: Float32 = (2*num + den)/(2*den)
			let k: Int32 = Int32(kf.rounded(.down))
			//debugPrint("num \(num), den \(den), k \(k)")
			guard k != 0 else { continue }
			self.rowSubPlace(r, minusRow: row, times: k)
			//debugPrint("Did rowSubPlace")
			change = true
		}
		return change
	}
	
	mutating func reduceL(to diag: Int) -> [Int] {
		// matrix view guards index limits
		var reduceIndices: [Int] = []
		for i in stride(from: self.rows - 1, to: -1, by: -1) { // for i in self.rows where i != diag { // can do where in a for loop...?
			if self.rowDown(from: i, col: i) {
				reduceIndices.append(i)
			}
		}
		return reduceIndices
	}

	mutating func colzeroPass(col: Int = 0, row: Int = 0) -> Bool {
		// reduce using no divides per-se down the column
		
		// find the index for the minimum non-zero element at or below the row in the column
		var mn: Int32 = Int32.max // big number, testing if 999999999 or Int32.max gives an error...
		var mni: Int?
		for r in row..<self.rows {
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

		for r in (row+1)..<self.rows {
			let num: Int32 = self.pid[r][col].item()
			let den: Int32 = self.pid[row][col].item()
			let kf = Float(2*num + den)/Float(2*den)
			let k = Int32(kf.rounded(.down))
			guard k != 0 else { continue }
			change = true
			self.rowSubPlace(r, minusRow: row, times: k) // TODO: confirm rounding is to the neares for k = num/den
		}
		return change
	}
	
	mutating func rowneg(_ r: Int) {
		self.pid[r] = -self.pid[r]
		self.row[r] = -self.row[r]
	}
	
	mutating func rowSubPlace(_ r1: Int, minusRow r2: Int, times k: Int32) {
		// row r1 gets minus k times r2, i.e.
		// row r1 becomes r1 - k*r2, corow r2 becomes r2 + k*r1
		if r1 == r2 { return }
		if k == 0 { return }
		self.pid[r1] = self.pid[r1] - k*self.pid[r2]
		// now use the comatrix to recalculate the row (to keep precision in the face of lots of integer ops)
		self.row[r1] = self.pid[r1].matmul(self.corow)//.transposed()) // transposed or not,not sure yet...
		self.driftRow += abs(Float32(k)*self.driftRow) // keep track of how errors are multiplying (aggregated for all rows, as batch update ...)
		if self.driftRow > self.driftThreshold {
			self.row = self.pid.matmul(self.corow) // re-align/restore for errors
			self.driftRow = machineEpsilon // and reset the error counter
		}
	}
	
	mutating func rowswap(_ r1: Int, _ r2: Int) { // }, preserveLQ: Bool = true) {
		if r1 == r2 { return }
		let temp_pid = self.pid[r1]
		self.pid[r1] = self.pid[r2]
		self.pid[r2] = temp_pid
		let temp_row = self.row[r1]
		self.row[r1] = self.row[r2]
		self.row[r2] = temp_row
	}
	
	mutating func rowslide(_ r1: Int, _ r2: Int, preserveLQ: Bool = true) {
		if r1 < r2 {
			for i in stride(from: r1, to: r2, by: 1) {
				self.rowswap(i, i+1)
				if preserveLQ {
					self.givens(row: i, Col0: i, col1: i + 1)
				}
			}
		} else if r2 < r1 {
			for i in stride(from: r1, to: r2, by: -1) {
				if preserveLQ {
					self.givens(row: i, Col0: i - 1, col1: i)
				}
				self.rowswap(i, i-1)
			}
		}
	}
	
	mutating func givens(row r: Int, Col0 c0: Int, col1 c1: Int) {
		// do givens on the column between columns c0 and c1 so that the value in row r column c1 is zero
		assert(c1 == c0 + 1, "For now this function only works for adjacent columns (due to slicing knowledge of myself)")
		let clam: Float = self.row[r, c0].item()
		let slam: Float = self.row[r, c1].item()
		let dis = hypot(clam, slam) // need import Foundation for hypot()
		let c = clam/dis
		let s = slam/dis
		let q = MLXArray([c, -s, s, c], [2, 2])
		self.row[0..<self.row.shape[0],c0..<c0+2] = self.row[0..<self.row.shape[0],c0..<c0+2].matmul(q)
		self.corow[0..<self.corow.shape[0],c0..<c0+2] = self.corow[0..<self.corow.shape[0],c0..<c0+2].matmul(q)
	}
}
