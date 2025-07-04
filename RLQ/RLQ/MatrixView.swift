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
	
	func align() {
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
								.contextMenu {
									Button(action: {
										//guard row > 0 else { return } // The input is dimension checked by the view
										self.matrix.houseRow(row, col)
										//self.matrix.lq() // test it out, it just changes the corow...so no visual on that yet...
										self.align() // refresh the integers view
									}) {
										Text("House Row zero to the right")
									}
									Button(action: {
										//guard row > 0 else { return } // The input is dimension checked by the view
										self.matrix.houseDiag(row, col)
										//self.matrix.lq() // test it out, it just changes the corow...so no visual on that yet...
										self.align() // refresh the integers view
									}) {
										Text("House Row Diagonal")
									}
									Button(action: {
										guard col - 1 >= 0 else { return } // The input is dimension checked by the view
										guard row - 1 >= 0 else { return }
										self.matrix.rowswap(row, row - 1)
										self.matrix.givens(row: row - 1, Col0: col - 1, col1: col)
										//self.matrix.lq() // test it out, it just changes the corow...so no visual on that yet...
										self.align() // refresh the integers view
									}) {
										Text("Diag-up")
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
										self.colzeroPass(cm: col, rm: row)
										self.align() // refresh the integers view
									}) {
										Text("colzeroPass")
									}
									Button(action: {
										self.matrix.rowneg(row)
										self.align()
									}) {
										Text("Negate Row")
									}
									Button(action: {
										guard row - 1 <= 0 else { return }
										self.matrix.rowswap(row, row-1)
										self.align() // refresh the integers view
									}) {
										Text("Swap row up")
									}
									Button(action: {
										//guard row > 0 else { return } // The input is dimension checked by the view
										self.matrix.rowslide(row, 0)
										self.matrix.lq() // test it out, it just changes the corow...so no visual on that yet...
										self.align() // refresh the integers view
									}) {
										Text("Put row to top")
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
		.padding()
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
