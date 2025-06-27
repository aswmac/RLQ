//
//  ContentView.swift
//  RLQ
//
//  Created 2025.06.27.140857
//

import SwiftUI

struct ContentView: View {
    @Binding var document: RLQDocument

    var body: some View {
        MatrixView(input_matrix: document.mat)
    }
}

//#Preview {
//    ContentView(document: .constant(RLQDocument()))
//}
