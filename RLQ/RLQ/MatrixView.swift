import SwiftUI
import MLX

struct MatrixView: View {
	// MARK: Properties
	var matrix: RLQ
	@State var integers: [[Int]] // The copy of the matrix for printing/viewing
	@State private var hoverIndex: (Int, Int)? = nil
	
	// MARK: Body
	init(input_matrix: MLXArray) {
		let im = RLQ(input_matrix)
		self.matrix = im
		self.integers = im.intArray()
	}
	
	func align() {
		self.integers = self.matrix.intArray()
	}
	
	func colzeroPass(cm: Int, rm: Int) {
		self.matrix.colzeroPass(col: cm, row: rm)
		self.integers[rm][cm] = 0
		
	}
	
	var body: some View {
		VStack {
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
										self.align()
									}) {
										Text("colzeroPass")
									}
									Button(action: {
										self.integers[row][col] = 0
									}) {
										Text("Make Zero")
									}
									Button(action: {
										()
									}) {
										Text("Function 2")
									}
								}
						}
					}
				}
			} // end Grid
			if let index = hoverIndex {
				Text("Row: \(index.0), Column: \(index.1)")
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
