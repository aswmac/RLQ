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

//# CALL DEPENDENCY
//# ----------------------------------------------------------------------
//# -----------main functions---------------------------------------------
//# ----------------------------------------------------------------------
//# init()      --> NONE
//# house_row()     --> NONE
//# setnull()       --> align()
//# digallinc()     --> digall(), zrow()
//# digall()        --> digest(), reset(), lq() (lq implied: input form)
//# zrow()          --> house_row()
//# digest()        --> givens()
//# reset()         --> NONE
//# lq()            --> house_row()
//# givens()        --> NONE
//# ----------------------------------------------------------------------
//# -----------finishing graph enumeration functions----------------------
//# ----------------------------------------------------------------------
//# ered()          --> enum(), rowswap(), rownorm()
//# enum()          --> house_row(), row_sub_place(), rowswap(), colswap()
//# row_sub_place() --> NONE
//# rowswap()       --> NONE
//# colswap()       --> NONE
//# ----------------------------------------------------------------------
//# ------------other useful functions------------------------------------
//# ----------------------------------------------------------------------
//# rownorm()       --> NONE
//# crestmax()      --> house_row(), rowswap(), row_sub_place(), givens()
//# xrows()         --> house_row(), xcol()
//# xcol()          --> NONE

import MLX
import MLXLinalg
import MLXRandom
internal import RealModule // for Float32.sqrt(nn)
import Foundation // for hypot()


// TODO: also (like if take in high bits truncated of large numbers) can do error analysis for the pid...
struct RLQ {
	var pid: MLXArray // Int
	var row: MLXArray // Float
	var corow: MLXArray // SQUARE matrix, float
	var reddim: Int = 0
	private let machineEpsilon: Float32 = 1.0e-15 // If just Float, then Float64...MLX ERROR
	private let driftThreshold: Float32 = 1.0e-4 // tolerance of error aggregated, triggers a re-alignment doing R = L*Q
	private let qualityMIN: Float32 = 1.1548 // min for all assumption possible is 1.15470025 ie 2/√3
	private var driftRow: Float32 = 1.0e-7 // The track of row operations errors that might want a re-alignment like R = LQ or R_i = ...
	
	init(rows: Int, cols: Int) {
		self.pid = MLXArray.eye(rows, m: cols, k: 0, dtype: .int32) // set the diagonal elements to unity
		self.row = MLXArray.eye(rows, m: cols, k: 0, dtype: .float32) // set the diagonal elements to unity
		self.corow = MLXArray.eye(cols, m: cols, k: 0, dtype: .float32)
		driftRow = machineEpsilon  // instead of always doing row[i] = pid[i] @ Q, batch them with a count
		assert(self.pid.dtype == .int32, "main init(): pid dtype is not int32")
		assert(self.corow.dtype == .float32, "main init(): corow dtype is not float32")
		assert(self.row.dtype == .float32, "main init(): row dtype is not float32")
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
		assert(self.pid.dtype == .int32, "MLX array init(): pid dtype is not int32")
		assert(self.corow.dtype == .float32, "MLX array init(): corow dtype is not float32")
		assert(self.row.dtype == .float32, "rMLX array init(): ow dtype is not float32")
	}
	
	mutating func reset(reorder: Bool = true) {
		self.row = self.pid.asType(.float32) // COPY-CONVERT the integers to floats, self.pid.asMLXArray(dtype: .float32) NOT WORK
		assert(self.row.dtype == .float32, "reset(): row dtype is not float32")
		//debugPrint("Should contain identity: row = \(self.row)")
		self.driftRow = machineEpsilon
		self.corow = MLXArray.eye(cols, m: cols, k: 0, dtype: .float32)
		//debugPrint("reset() before reorder = \(reorder) gives corow = \(self.corow)")
		if reorder { self.lq() }
		assert(self.pid.dtype == .int32, "reset(): pid dtype is not int32")
		assert(self.corow.dtype == .float32, "reset(): corow dtype is not float32")
		
	}
	
	mutating func setnull(x: MLXArray) {
		assert(x.count <= self.rows, "RLQ.setnull(x): x too long to fit")
		let e = MLXArray.eye(rows, m: rows, k: 0, dtype: .int32)
		if x.count < self.rows { // pad the x input if necessary
			let xx = concatenated([x.reshaped(-1,1), MLXArray.zeros([-1,self.rows-x.count])], axis: 0) // xx a column vector now
			self.pid = concatenated([xx,e], axis: 1) // set pid to the column vector augmented with the identity matrix
		} else {
			self.pid = concatenated([x.reshaped([-1,1]),e], axis: 1)
			//debugPrint("Should contain identity: pic = \(self.pid)")
		}
		self.reset(reorder: false) // reset and perform the LQ factoring
	}
	
	mutating func randNull() {
		let x = MLXRandom.randInt(low: -100000000, high: 100000000, [self.rows]) // could do [-1, self.rows] I think to make column vector, but setnull takes row vec
		//debugPrint("setting random values with x = \(x)")
		self.setnull(x: x)
		//debugPrint("randNull(): pid = \(self.pid)")
		//debugPrint("randNull(): row = \(self.row)")
	}
	
	mutating func lq(dim: Int? = nil) {
		let dim = dim ?? self.reddim
		for i in 0..<dim { // For each row going down, find the smallest norm
			var mn: Int?
			var mni: Int?
			for j in i..<dim { // TODO: ? want to do linalg norm like houserow's vn = (..., stream: .cpu)...? (TEST IT SOMETIME...)
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
	
	mutating func digest(_ rm: Int, start: Int = 0) -> Bool {
		// look at adjacent diagonal values of the LQ form and mix for larger values at the lower end.
		// Assumption: row is in lq form
		// max value if want always possible is √3/2 = 0.8660254037844386, ie like   | 1    0    |  or  | 1.155    0  |
		// 2/√3 = 1.1547005383792517                                                | •  0.866  |      |   •       1  |
		let diagEpsilon: Float = 1e-6 // or machineEpsilon? or variable input to func?... used to test changes TODO: FLOAT32 gave blank-err?
		var change: Bool = false
		if rm >= self.reddim {
			//debugPrint("digest() called with low reddim value, adjusting reddim from \(self.reddim) to \(rm+1)")
			self.reddim = rm+1
		}
		if self.reddim > rm + 1 {
			//debugPrint("digest() called with high reddim value, adjusting reddim from \(self.reddim) to \(rm+1)")
			//debugPrint("If want to do something with a loop using a higher reddim, stay tuned for whatever changes will come....")
			self.reddim = rm+1
		}
		var i = start + 1 //
		while i < self.reddim {
			let a: Float32 = self.row[i-1, i-1].item()    //<---  | a 0 |
			let e: Float32 = self.row[i, i-1].item()    //<---  | e f |
			let f: Float32 = self.row[i, i].item()        //<--- variables renamed only for reading and typing convenience.
			if abs(a) < diagEpsilon { i += 1; continue }  // way small like epsilon, a zero row likely
			let test = abs(a/f)
			if test < self.qualityMIN { i += 1; continue } // here the diagonal is reduced up to quality already
			let zchange = self.zrow(rm: i) // This will reduce e, but also those to the left of e--Good for loop on larger block
			if zchange == false {
				self.givens(row: i, Col0: i-1, col1: i) // inline of rowSlide which preserves the LQ form
				self.rowswap(i, i - 1)
				let z2 = self.zrow(rm: i)
				if z2 == false && abs(e/a) > 0.5 {
					//debugPrint("digest(): The diagonal values (at i = \(i)) looked good for reduction")
					//debugPrint("----because test = \(test) and ratio = \(e/a), but no such reduction was found!!")
					i += 1;
					continue
				}
				if abs(e/a) <= 0.5 { // I think this could be a stick in the loop otherwise...
					i += 1
					continue
				}
			}
			change = true
			let new_e: Float32 = self.row[i, i-1].item()
			if a*a - (new_e*new_e + f*f) > diagEpsilon { // if the reductions found a significantly smaller "upper-low" value
				change = true
				self.givens(row: i, Col0: i-1, col1: i) // inlline of rowSlide which preserves the LQ form
				self.rowswap(i, i - 1)
				continue // do it again at this index since it did a reduction and swapped rows
			} else {
				i += 1
			}
		}
		return change
	}
	
	mutating func digall() -> Int {
		guard self.reddim > 1 else { return 0 } // otherwise dratio would be empty and max would be nil!
		let digallEpsilon: Float32 = 1e-6
		var count = 0
		var rred = true
		//debugPrint("Starting rred while loop...")
		while rred {
			rred = false
			//self.reset(reorder: true) // reset(true) includes lq()
			var dratio: [Float32] = []
			for i in 1..<self.reddim {
				let a: Float32 = self.row[i-1, i-1].item()    //<---  | a 0 |
				//let e: Float32 = self.row[i, i-1].item()    //<---  | e f |
				let f: Float32 = self.row[i, i].item()        //<--- variables renamed for convenience.
				dratio.append(abs(f) < digallEpsilon ? 0.0 : abs(a)/abs(f))
			}
			var dmax = dratio.max()!
			//debugPrint("dmax initial: \(dmax)")
			while dmax > self.qualityMIN {
				while self.digest(self.reddim - 1) { continue }
				self.reset(reorder: true)
				for i in 1..<self.reddim {
					if abs(self.row[i, i].item()) < digallEpsilon { dratio[i - 1] = 0.0 }
					else { dratio[i - 1] = abs(self.row[i-1, i-1].item()/self.row[i, i].item()) }
				}
				count += 1
				dmax = dratio.max()!
				//debugPrint("at count \(count) dmax: \(dmax)")
			}
			//debugPrint("rred iteration \(count), dratio: \(dratio), rred: \(rred)")
		}
		return count
	}
	
	mutating func digallinc() -> Int {
		let saveDim = self.reddim
		self.reddim = 2
		var count = self.digall()
		self.reddim = 3
		while self.reddim <= self.rows {
			_ = self.zrow(rm: self.reddim - 1) // call once, it does not iterate anything
			count += self.digall()
			self.reddim += 1
		}
		self.reddim = saveDim
		return count
	}
	
	mutating func zrow(rm: Int, start: Int = 0) -> Bool {
		// target reduce the row. Use start to limit which rows to use for reducing
		// ASSUMPTION: row[] in in lq form.
		let zedEpsilon: Float = 1e-6 // avoid small components and zero division
		let d0: Float32 = inner(self.row[rm, 0..<rm], self.row[rm, 0..<rm]).item() // get norm squared of L-row
		var change: Bool = false
		for i in stride(from: rm - 1, to: start - 1, by: -1) { // look at diagonal elements of row going backwards/countdown
			let den: Float32 = self.row[i,i].item()
			if abs(den) < zedEpsilon { continue }  // ignore small components
			let num: Float32 = self.row[rm, i].item()
			let t = num/den
			if abs(abs(t) - 0.5) < zedEpsilon { continue } // ignore small reduction amounts
			let k: Int32 = Int32(t.rounded(.toNearestOrAwayFromZero))
			if k == 0 { continue } // if there is no reduction
			self.rowSubPlace(rm, minusRow: i, times: k)
			change = true
		}
		if change {
			let d1: Float32 = inner(self.row[rm], self.row[rm]).item()
			return d1 < d0
		} else {
			return false
		}
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
		//debugPrint("start: row[0] = \(self.row[0])")
		//debugPrint("start: pid[0] = \(self.pid[0])")
		let kx = self.row[ir, ic..<self.cols]
		//debugPrint("shape of kx is \(kx.shape)")
		//let nn = rmsNorm(kx, weight: .ones(like: kx), eps: machineEpsilon) // NO! this is x broadcast-divided by (L_2 / sqrt(dim))
		let nn = MLXLinalg.norm(kx, ord: 2)
		let squaredim = self.cols - ic
		let e0 = eye(1, m: squaredim, k: 0, dtype: .float32)
		let vk = kx - nn*e0
		//debugPrint("01: row[0] = \(self.row[0])")
		//debugPrint("01: pid[0] = \(self.pid[0])")
		let vn = MLXLinalg.norm(vk, ord: 2, stream: .cpu) // MLX error: [linalg::svd] This op is not yet supported on the GPU. Explicitly pass a CPU stream to run it.
		let v = vk/vn
		let xx = outer(v, v) // outer product, (self.cols - ic - 1) square
		//debugPrint("02: row[0] = \(self.row[0])")
		//debugPrint("02: pid[0] = \(self.pid[0])")
		let e = eye(squaredim, m: squaredim, k: 0, dtype: .float32)
		let q = e - 2*xx
		//debugPrint("03: row[0] = \(self.row[0])")
		//debugPrint("03: pid[0] = \(self.pid[0])")
		//debugPrint("shape of q is \(q.shape)")
		let temprow = self.row[0..<self.rows,ic..<self.cols]
		//debugPrint("temprow is \(temprow)")
		let newrow = temprow.matmul(q)
		//debugPrint("newrow is \(newrow)")
		//debugPrint("shape of newrow is \(newrow.shape)")
		//debugPrint("row[0] (again) is \(self.row[0])")
		self.row[0..<self.rows,ic..<self.cols] = newrow
		//debugPrint("04: row[0] = \(self.row[0])")
		//debugPrint("04: pid[0] = \(self.pid[0])")
		self.corow[0..<self.cols,ic..<self.cols] = self.corow[0..<self.cols,ic..<self.cols].matmul(q) // corow is cols by cols
		//debugPrint("end: row[0] = \(self.row[0])")
		//debugPrint("end: pid[0] = \(self.pid[0])")
	}
	
	func dot(row1 r1: Int, row2 r2: Int, from c1: Int, to c2: Int) -> Float32 {
		let urow_i: Float32 = (sum(self.row[r1, c1..<c2]*self.row[r2, c1..<c2])).item()
		return urow_i
	}
	
	func intArray() -> [[Int]] {
		var integers: [[Int]] = []
		let rrows = Int(self.rows)
		let ccols = Int(self.cols)
		for row in 0..<rrows {
			integers.append(Array(repeating: 0, count: ccols))
			for col in 0..<ccols {
				let element: Int = self.pid[row, col].item()
				integers[row][col] = element
			}
		}
		return integers
	}

	func floatArray() -> [[Float32]] {
		var floaters: [[Float32]] = []
		let rrows = Int(self.rows)
		let ccols = Int(self.cols)
		for row in 0..<rrows {
			floaters.append(Array(repeating: 0, count: ccols))
			for col in 0..<ccols {
				let element: Float32 = self.row[row, col].item()
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
	mutating func zcol(from row: Int, col: Int) -> Bool {
		// reduce down using the row elements in the column as the guide
		// no search for minimum, just use this element and reduce below
		let den: Float32 = self.row[row, col].item()
		guard abs(den) > machineEpsilon else { return false }
		var change: Bool = false
		for r in row+1..<self.rows {
			//debugPrint("checking \(r)")
			let num: Float32 = self.row[r, col].item()
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
	
	mutating func reduceUnderL(to diag: Int) -> [Int] {
		/// Do all zrow like actions but for all under the diagonal up to index diag
		// matrix view guards index limits
		assert(diag < self.rows && diag < self.cols, "reduceUnderL called with bad index \(diag)")
		var reduceIndices: [Int] = []
		for i in stride(from: diag, to: -1, by: -1) { // for i in self.rows where i != diag { // can do where in a for loop...?
			if self.zcol(from: i, col: i) {
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
			let p: Int32 = abs(self.pid[r, col].item())
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
			let num: Int32 = self.pid[r, col].item()
			let den: Int32 = self.pid[row, col].item()
			let kf = Float(2*num + den)/Float(2*den) // Seems to work--it zeros the column
			let k = Int32(kf.rounded(.down)) // with this round down of course
			guard k != 0 else { continue }
			change = true
			self.rowSubPlace(r, minusRow: row, times: k)
		}
		return change
	}
	
	mutating func rowneg(_ r: Int) {
		self.pid[r] = -self.pid[r]
		self.row[r] = -self.row[r]
	}
	
	// MARK: row[0..<m][c] = -row[0..<m][c] seems not to do anything, need to index like [0..<m, c]!!!
	mutating func colneg(_ c: Int) {
		self.row[0..<self.rows, c] = -self.row[0..<self.rows, c]
		self.corow[0..<self.rows, c] = -self.corow[0..<self.rows, c]
	}
	
	mutating func rowSubPlace(_ r1: Int, minusRow r2: Int, times k: Int32) {
		// row r1 gets minus k times r2, i.e.
		// row r1 becomes r1 - k*r2, corow r2 becomes r2 + k*r1
		if r1 == r2 { return }
		if k == 0 { return }
		self.pid[r1] = self.pid[r1] - k*self.pid[r2]
		self.row[r1] = self.row[r1] - k*self.row[r2]
		self.driftRow += abs(Float32(k)*self.driftRow) // keep track of how errors are multiplying (aggregated for all rows, as batch update ...)
		if self.driftRow > self.driftThreshold {
			self.row = self.pid.asType(.float32)
			self.row = self.row.matmul(self.corow) // re-align/restore for errors
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
	
	mutating func diagSwap(_ r1: Int, _ r2: Int) {
		if r1 == r2 { return }
		self.diagSlide(from: r1, to: r2) // now r2 is in r2 - 1
		if r1 - r2 > 1 { // if the slide is down but needs more than one to complete the swap
			self.diagSlide(from: r1 + 1, to: r2) // otherwise put it in r1 to complete the swap
		}
		if r2 - r1 > 1 { // if the slide is up but needs more than one to complete the swap
			self.diagSlide(from: r2 - 1, to: r1) // otherwise put it in r1 to complete the swap
		}
	}
	
	mutating func diagSlide(from r1: Int, to r2: Int) {
		if r1 == r2 { return }
		if r1 > r2 {
			for r in stride(from: r1, to: r2, by: -1) {
				self.rowswap(r, r - 1)
				self.givens(row: r - 1, Col0: r - 1, col1: r)
			}
		} else { // here r1 < r2
			for r in r1..<r2 { // I wonder if the swift or mlx compiler is smart enough to batch
				self.givens(row: r + 1, Col0: r, col1: r + 1) // the rowstuff separate from the colstuff
				self.rowswap(r + 1, r) // TODO: consider batching row separate from colstuff and speedtest it
			}
		}
	}
	
	mutating func colswap(_ c1: Int, _ c2: Int) {
		if c1 == c2 { return }
		let temp_pid = self.pid[0..<self.rows, c1]
		self.pid[0..<self.rows, c1] = self.pid[0..<self.rows, c2]
		self.pid[0..<self.rows, c2] = temp_pid
		let temp_col = self.row[0..<self.rows, c1]
		self.row[0..<self.rows, c1] = self.row[0..<self.rows, c2]
		self.row[0..<self.rows, c2] = temp_col
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
	
	private func unitDot(_ r1: Int, _ r2: Int) -> Float {
		let n1 = MLXLinalg.norm(self.row[r1], ord: 2)
		let n2 = MLXLinalg.norm(self.row[r2], ord: 2)
		let u1 = self.row[r1]/n1
		let u2 = self.row[r2]/n2
		return inner(u1, u2).item()
	}
	
	func unitDotValues() -> MLXArray {
		let N = MLXLinalg.norm(self.row, ord: 2, axis: 1) // The norms of the rows
		debugPrint("Row norms: \(N)")
		let UR = self.row/N.reshaped([-1, 1]) // The rows as unit vectors
		//debugPrint("UR: \(UR)")
		let UUT = inner(UR, UR)
		let e = MLXArray.eye(self.rows, k: 0, dtype: UUT.dtype)
		let OffDiag = UUT - e
		//debugPrint("UUT: \(UUT)")
		let maxIndex = argMax(abs(OffDiag)) // the index in the flattened array of the largest value
		print("Max off diagonal: \(OffDiag.flattened()[maxIndex])")
		print("at (\(maxIndex)))")
		print("maxIndex/self.row.shape[0]: \(maxIndex/self.row.shape[0])")
		print("maxIndex%self.row.shape[0]: \(maxIndex%self.row.shape[0])")
		print("sum is \(sum(N))")
		let nnnn = sum(diag(self.row.matmul(transposed(self.row))))
		//let nnnn = MLXLinalg.norm(self.row, ord: 'fro') // Cannot convert value of type 'String' to expected argument type 'Double'
		print("nnnn: \(nnnn)")
		return UUT
	}
}
