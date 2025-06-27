import SwiftUI
import MLX

struct MatrixView: View {
	// MARK: Properties
	let matrix: MLXArray
	var integers: [[Int]]
	@State private var hoverIndex: (Int, Int)? = nil
	
	// MARK: Body
	init(matrix: MLXArray) {
		self.matrix = matrix
		self.integers = []
		let shape = matrix.shape
		let rows = Int(shape[0])
		let cols = Int(shape[1])
		for row in 0..<rows {
			integers.append(Array(repeating: 0, count: cols))
			for col in 0..<cols {
				let element: Int = matrix[row][col].item()
				self.integers[row][col] = element
			}
		}
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
										()
									}) {
										Text("Function 1")
									}
									Button(action: {
										()
									}) {
										Text("Function 2")
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
