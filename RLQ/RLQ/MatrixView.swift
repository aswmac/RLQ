import SwiftUI
import MLX

struct MatrixView: View {
	// MARK: Properties
	@State var matrix: RLQ
	@State var integers: [[Int]] // The copy of the matrix for printing/viewing
	@State var floaters: [[Float]] // The copy of the matrix for printing/viewing
	@State private var hoverIndex: (Int, Int)? = nil
	
	// MARK: Body
	init(input_matrix: MLXArray) {
		let im = RLQ(input_matrix)
		self.matrix = im
		self.integers = im.intArray()
		self.floaters = im.floatArray()
	}
	
	func resync() {
		self.integers = self.matrix.intArray()
		self.floaters = self.matrix.floatArray()
	}
	
	func colzeroPass(cm: Int, rm: Int) {
		_ = self.matrix.colzeroPass(col: cm, row: rm)
		
	}
	
	var body: some View {
		VStack {
			Grid(alignment: .center) {
				ForEach(0..<floaters.count, id: \.self) { row in
					GridRow {
						ForEach(0..<floaters[row].count, id: \.self) { col in
							let t: Float = floaters[row][col]
							//let c: String = "\(t, format: .scientific, significantDigits: 5)"
							Text("\(t)") //t.formatted(.number.notation(.scientific).significantDigits(5))) GIVES PROBLEM!!
								.onHover { isHovered in
									if isHovered {
										hoverIndex = (row, col)
									} else {
										hoverIndex = nil
									}
								}
								.contextMenu {
									Button(action: {
										//guard row > 0 else { return } // The input is dimension checked by the view
										self.matrix.houseRow(row, col)
										//self.matrix.lq() // test it out, it just changes the corow...so no visual on that yet...
										self.resync() // refresh the integers view
									}) {
										Text("House Row zero to the right")
									}
									Button(action: {
										//guard row > 0 else { return } // The input is dimension checked by the view
										self.matrix.houseDiag(row, col)
										//self.matrix.lq() // test it out, it just changes the corow...so no visual on that yet...
										self.resync() // refresh the integers view
									}) {
										Text("House Row Diagonal")
									}
									Button(action: {
										guard row + 1 < self.matrix.rows else { return } // could say not equal to zero...
										self.matrix.diagSwap(row, row + 1)
										//self.matrix.lq() // test it out, it just changes the corow...so no visual on that yet...
										self.resync() // refresh the integers view
									}) {
										Text("Diag-swap down")
									}
									Button(action: {
										self.matrix.reddim = row
										self.matrix.lq()
										self.resync() // refresh the integers/row view
									}) {
										Text("LQ")
									}
									Button(action: {
										self.matrix.smithDiagROW(row: row, col: col)
										//self.matrix.smithDiagROW(row: row, col: col)
										self.resync() // refresh the integers/row view
									}) {
										Text("smith-diag-down")
									}
									Button(action: {
										_ = self.matrix.reduceUnderL(to: row)
										self.resync()
									}) {
										Text("Reduce diagonal")
									}
									Button(action: {
										_ = self.matrix.digallinc()
										self.resync()
									}) {
										Text("dig-loop")
									}
									Button(action: {
										self.matrix.rowsort()
										self.resync()
									}) {
										Text("rowsort")
									}
									Button(action: {
										self.matrix.reset(reorder: false)
										self.resync()
									}) {
										Text("reset")
									}
								}
						}
					}
				}
			}
			Grid(alignment: .center) {
				ForEach(0..<integers.count, id: \.self) { row in
					GridRow {
						ForEach(0..<integers[row].count, id: \.self) { col in
							Text("\(integers[row][col])")
							//.frame(width: 300, height: 300)
								.monospaced()
								.background(Color.gray.opacity(0.2))
								.cornerRadius(4)
								.multilineTextAlignment(.leading)
								.onHover { isHovered in
									if isHovered {
										hoverIndex = (row, col)
									} else {
										hoverIndex = nil
									}
								}
								.contextMenu {
									Button(action: {
										self.matrix.enumerate(for: row)
										self.resync()
									}) {
										Text("ENUMERATE!")
									}
									Button(action: {
										self.colzeroPass(cm: col, rm: row)
										self.resync() // refresh the integers view
									}) {
										Text("colzeroPass")
									}
									Button(action: {
										self.matrix.rowneg(row)
										self.resync()
									}) {
										Text("Negate Row")
									}
									Button(action: {
										guard row - 1 >= 0 else { return }
										self.matrix.rowswap(row, row-1)
										self.resync() // refresh the integers view
									}) {
										Text("Swap row up")
									}
									Button(action: {
										//guard row > 0 else { return } // The input is dimension checked by the view
										self.matrix.rowslide(row, 0)
										self.matrix.lq() // test it out, it just changes the corow...so no visual on that yet...
										self.resync() // refresh the integers view
									}) {
										Text("Put row to top")
									}
									Button(action: {
										self.matrix.randNull()
										self.resync()
									}) {
										Text("Randomize entire matrix null column")
									}
									Button(action: {
										_ = self.matrix.zrow(rm: row)
										self.resync()
									}) {
										Text("Reduce row using LQ form")
									}
									Button(action: {
										_ = self.matrix.minPairAngle()
										_ = self.matrix.zrow(rm: 1) 
										self.resync()
									}) {
										Text("Print unit dot")
									}
									Button(action: {
										//let e = MLXArray.eye(1, m: 31, k: 2, dtype: .float32)
										self.matrix.nearest(to: col)
										//self.resync()
									}) {
										Text("Nearest test")
									}
								}
						}
					}
				}
			} // end Grid
			if let index = hoverIndex {
				let pi = self.integers[index.0][index.1]
				Text("Row: \(index.0), Column: \(index.1) ------- \(pi)")
//				Text("Row: \(index.0), Column: \(index.1)")
			} else {
				Text("Row: *, Column: *")
			}
		} // end VStack
	} // end body
}

//// Preview Provider
//struct MatrixView_Previews: PreviewProvider {
//	static var previews: some View {
////		let matrix = MLXArray.eye(3, m: 4, k: 0, dtype: .int64) // set the diagonal elements to unity
//		let matrix: [[Int]] = [
//			[11, 2, 3, 7],
//			[4, 5, 6, 9],
//			[7, 8, 9, 5]
//		]
//		MatrixView(matrix: matrix)
//	}
//}
